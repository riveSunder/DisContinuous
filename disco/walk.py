import os
import sys
import argparse
import subprocess

import time
import json

import numpy as np
import torch
import torch.nn.functional as F

import yuca
from yuca.zoo.librarian import Librarian
from yuca.ca.continuous import CCA
from yuca.ca.neural import NCA
from yuca.ca.reaction_diffusion import RxnDfn
from yuca.utils import query_kwargs

import skimage
import skimage.io as sio
import skimage.transform

import matplotlib.pyplot as plt

CA_DICT = {"CCA": CCA, \
        "NCA": NCA, \
        "RxnDfn": RxnDfn}

def seed_both(my_seed):

    np.random.seed(my_seed)
    torch.manual_seed(my_seed)

def walk(**kwargs):

    hidden_channels = 256
    save_images = kwargs["save_images"]
    time_limit = kwargs["time_limit"]
    number_evaluations = kwargs["number_evaluations"]
    my_seed = kwargs["random_seed"]
    system_type = kwargs["system_type"]
    alpha = kwargs["alpha"]
    epsilon = 0.8

    my_device = kwargs["device"]
    if torch.cuda.is_available() and "cuda" in my_device.lower():
        my_device = my_device
    elif "cpu" in my_device:
        my_device = "cpu"
    else:
        print("cuda not available, falling back to cpu device")
        my_device = "cpu"

    max_dt = max(kwargs["time_step_bounds"])
    min_dt = min(kwargs["time_step_bounds"])

    min_kr = min(kwargs["kernel_radius_bounds"])
    max_kr = max(kwargs["kernel_radius_bounds"])

    # how far we can step in our random walk
    max_scale_dt = max_dt * 0.25
    max_scale_kr = max_kr * 0.25
    min_scale_dt = max_dt / 1000.0
    min_scale_kr = 1
    
    bits = kwargs["numerical_precision"]

    max_steps = kwargs["max_steps"]
    glider_patterns = kwargs["glider_pattern"]

    # clean up floating point precision list
    remove_list = []
    for gg in range(len(bits)):
        if bits[gg] != 64:
            if bits[gg] != 32:
                if bits[gg] == 16 and "cuda" not in my_device:
                    print(f" {bits[gg]}-bit floats only supported on cuda devices" \
                            f" Dropping {bits[gg]}-bit floats from experiment.")
                    remove_list.append(numerical_precision[gg])
                elif "cuda" in my_device:
                    pass
                else:
                    print(f" {bits[gg]}-bit floats not supported in torch {torch.__version__}" \
                            f" Dropping {bits[gg]}-bit floats from experiment.")

    for remove_element in remove_list:
        numerical_precision.remove(remove_element)
    probability_bits = [1/len(bits) for elem in bits]

    results = {}
    
    lib = Librarian(verbose=False) 

    t00 = time.time()
    exp_names = []
    for pattern_name in glider_patterns:

        time_stamp = str(int(time.time()*1000))
        exp_folder = os.path.join("results", f"exp_{time_stamp}")
        log_results_path = f"metadata.json"
        log_results_path = os.path.join(exp_folder, log_results_path)
        results_path = os.path.join(exp_folder, f"results_{pattern_name}.npy")

        if os.path.exists(exp_folder):
            pass
        else:
            os.mkdir(exp_folder)

        # save results for this pattern_name
        if save_images:
            images_folder = os.path.join(exp_folder, "images")
            if os.path.exists(images_folder):
                pass
            else:
                os.mkdir(images_folder)

        log_results = {}
        log_results["entry_point"] = query_kwargs("entry_point", "none", **kwargs)
        log_results["git_hash"] = query_kwargs("git_hash", "none", **kwargs)
        log_results["time_limit"] = time_limit
        log_results["random_seed"] = my_seed
        log_results["system_type"] = system_type
        log_results["epsilon"] = epsilon
        log_results["device"] = my_device
        log_results["max_dt"] = max_dt 
        log_results["min_dt"] = min_dt 
        log_results["max_kr"] = max_kr 
        log_results["min_kr"] = min_kr
        log_results["max_scale_dt"] = max_scale_dt 
        log_results["min_scale_dt"] = min_scale_dt 
        log_results["max_scale_kr"] = max_scale_kr 
        log_results["min_scale_kr"] = min_scale_kr 
        log_results["numerical_precision"] = bits
        log_results["glider_pattern"] = pattern_name
        log_results["max_steps"] = max_steps

        t0 = time.time()
        elapsed = 0.0

        mean_dt = 0.5 * (max_dt + min_dt) 
        mean_kr = 0.5 * (max_kr + min_kr) 

        pattern, metadata = lib.load(pattern_name)
        ca = CA_DICT[system_type](hidden_channels=hidden_channels)
        ca.restore_config(metadata["ca_config"], verbose=False)

        kr = ca.kernel_radius #(max_kr - min_kr) // 2 
        dt = ca.dt #(max_dt - min_dt) / 2.

        kr_scale = min_scale_kr
        dt_scale = min_scale_dt

        last_persistence = 0
        persistence = 1

        persistence_list = []
        evaluation_count = 0

        while elapsed <= time_limit and evaluation_count <= number_evaluations:
            # select discretization parameters 


            bit = np.random.choice(bits, p=probability_bits)

            if bit == 16:
                torch.set_default_dtype(torch.float16)
            elif bit == 32:
                torch.set_default_dtype(torch.float32)
            else:
                torch.set_default_dtype(torch.float64)
            
            ## persistence run
            # initialize system
            ca = CA_DICT[system_type](hidden_channels=hidden_channels)
            ca.restore_config(metadata["ca_config"], verbose=False)
            native_config = ca.make_config()

            native_radius = native_config["neighborhood_kernel_config"]["radius"]

            # set temporal step size and kernel radius
            ca.dt = dt
            ca.change_kernel_radius(kr)
            ca.to_device(my_device)
            ca.no_grad()
            #turn off dropout!
            ca.eval()


            # set up and scale pattern
            my_scale = kr / native_radius
            use_anti_aliasing = True if my_scale < 1.0 else False

            while len(pattern.shape) < 4:
                pattern = pattern[None,...]

            scaled_pattern = skimage.transform.rescale(pattern, \
                    scale=(1,1, my_scale, my_scale), \
                    anti_aliasing=use_anti_aliasing)

            # set up grid
            grid_size = min([min([int(10 * kr), kr + max(scaled_pattern.shape)]), 1024])
            mid_grid = 0 #grid_size // 2 

            grid = torch.zeros(1, 1, grid_size, grid_size)

            crop_x, crop_y = scaled_pattern.shape[-2], scaled_pattern.shape[-1]


            # send pattern to the intended fp precision dtype
            scaled_pattern = torch.tensor(scaled_pattern, dtype=torch.get_default_dtype())

            # add pattern to grid
            grid[:,:,mid_grid:mid_grid+crop_x, mid_grid:mid_grid+crop_y] = scaled_pattern
            grid = grid.to(my_device)

            pattern_sum = grid.sum()
            starting_correlation = F.conv2d(\
                    grid.detach().cpu().to(torch.float32), \
                    scaled_pattern.detach().to(torch.float32)); 
            starting_correlation = starting_correlation / starting_correlation.std()

            for step in range(max_steps):

                grid = ca(grid)

#                current_correlation = F.conv2d(\
#                        grid.detach().cpu().to(torch.float32), \
#                        scaled_pattern.detach().to(torch.float32)); 
#                current_correlation = current_correlation / current_correlation.std()
#                relative_correlation = current_correlation.max() / starting_correlation.max()
                current_grid_sum = grid.sum()

                relative_correlation = 0.1
                gain = current_grid_sum / pattern_sum


                if (gain > 1.25 or gain < 0.75) \
                        and (relative_correlation > 1.5 or relative_correlation < 0.5):
                    print(f"\nloss of pattern homeostasis at step {step} assumed")
                    print(f"     pattern: {pattern_name}, dt {dt:.4f}, "\
                            f"radius {kr}, {bit}-bit precision")
                    print(f"     kr_scale: {kr_scale:.2f}, dt_scale: {dt_scale:.4f}")
                    print(f"     t_count: {ca.t_count:.3f}")
                    print(relative_correlation)
                    break
                elif step == (max_steps-1):
                    print(f"\npattern persisted to step {step}")
                    print(f"     pattern: {pattern_name}, dt {dt:.4f}, "\
                            f"radius {kr}, {bit}-bit precision")
                    print(f"     kr_scale: {kr_scale:.2f}, dt_scale: {dt_scale:.4f}")
                    print(f"     t_count: {ca.t_count:.3f}")


            if save_images:
                kr_string = f"{kr}"
                dt_string = f"{dt:.6f}".replace(".","_")
                fp_string = f"{int(bit)}"
                image_fn = f"final_grid_{pattern_name}_step{step}_"\
                        f"kr{kr_string}_dt{dt_string}_fp{fp_string}.png"
                save_image_path = os.path.join(images_folder, image_fn)
                image_to_save = np.array(255*grid.squeeze().cpu().numpy(), dtype=np.uint8)
                print(f"kr: {kr}, grid size = {grid.shape}")
                sio.imsave(save_image_path, image_to_save)

            # update random walk scale based on how different persistence was
            persistence = step

            m = (persistence + last_persistence) / (1e-1 + 2 * np.abs(persistence - last_persistence))
            scale_modifier = 1. + (m-1)*2
            scale_modifier = max([min([scale_modifier, 1.5]), 0.5])
#            1. \
#                    / (np.sqrt((persistence - last_persistence)**2) \
#                    / (persistence + last_persistence+1e-6) + epsilon)

            if persistence == 0 and last_persistence == 0: 
                scale_modifier = 1.5

            print(f"scale modifier {scale_modifier}", m, persistence, last_persistence)
            last_persistence = persistence

            kr_scale *= scale_modifier #alpha * kr_scale + (1-alpha) * scale_modifier
            dt_scale *= scale_modifier #alpha * dt_scale + (1-alpha) * scale_modifier

            kr_scale = min([max_scale_kr, max([min_scale_kr, kr_scale])])
            dt_scale = min([max_scale_dt, max([min_scale_dt, dt_scale])])

            elapsed = time.time() - t0

            persistence_list.append([dt, kr, bit, persistence])

            kr_shift = kr_scale * np.random.randn()
            dt_shift = dt_scale * np.random.randn()

            # if a parameter is up against a max/min boundary
            # ensure that the parameter is changed in the direction away 
            # from the boundary
            if np.isclose(kr, max_kr, rtol=0.001) and np.sign(kr_shift) > 0:
                kr_shift *= -1.
            elif np.isclose(kr, min_kr, rtol=0.001) and np.sign(kr_shift) < 0:
                kr_shift *= -1.

            if np.isclose(dt, max_dt, rtol=0.001) and np.sign(dt_shift) > 0:
                dt_shift *= -1.
            elif np.isclose(dt, min_dt, rtol=0.001) and np.sign(dt_shift) < 0:
                dt_shift *= -1.

            kr = int(kr + kr_shift)
            kr = min([max([kr, min_kr]), max_kr])

            dt = dt + dt_shift
            dt = min([max([dt, min_dt]), max_dt])

            evaluation_count +=1 

        log_results["wall_time"] = (time.time() - t0)

        if os.path.exists(exp_folder):
            pass
        else:
            os.mkdir(exp_folder)

        log_results["results_file"] = results_path #f"results_{pattern_name}.npy"

        print(f"saving results to {results_path}")
        np.save(results_path, np.array(persistence_list)) #results[pattern_name])

        print(f"saving metadata to {log_results_path}")
        with open(log_results_path, "w") as f:
            json.dump(log_results, f)

        exp_names.append(results_path)

    # print a summary statement
    print(f"\nRun completed. {len(exp_names)} experiments in {time.time() - t00:.2f} seconds")
    print("Results saved to:")
    for jj in range(len(exp_names)):
        print(f"          {exp_names[jj]}")

    print("")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--alpha", type=float, default=0.75,\
            help="exponential averaging variable R[0.0, 1.0], default=0.75")
    parser.add_argument("-c", "--system_type", type=str, default="CCA", \
            help="complex system, options: CCA, NCA, RxnDfn (default CCA)")
    parser.add_argument("-d", "--device", type=str, \
            default="cuda", help="device to use (cpu or cuda)")
    parser.add_argument("-g", "--glider_pattern", type=str, default=["orbium_orbium000"], nargs="+", \
            help="name of the glider pattern(s) to investigate")
    parser.add_argument("-k", "--kernel_radius_bounds", type=int, default=[3, 111], nargs="+",\
            help="neighborhood kernel radius(es) in pixels. k_r = 1/dk_r")
    parser.add_argument("-l", "--time_limit", type=int, default=3600,\
            help="maximum wall-clock run time in seconds (default 3600 = 1 hour)")
    parser.add_argument("-m", "--max_steps", type=int, default=2048,\
            help="maximum number of simulation steps ")
    parser.add_argument("-n", "--number_evaluations", type=float, default=float("Inf"),\
            help="evaluation limit (default \inf, not limited by number of evaluations)")
    parser.add_argument("-p", "--numerical_precision", type=int,\
            default=[16, 32, 64], nargs="+",\
            help="floating point precision options to be use (default 16, 32, 64)")
    parser.add_argument("-r", "--random_seed", type=int,\
            default=42, \
            help="seed for pseudorandom number generation")
    parser.add_argument("-s", "--save_images", type=int, default=0, \
            help="0: don't save final grid images (any other integer): do")
    parser.add_argument("-t", "--time_step_bounds", type=float, default=[0.001, 1.0], nargs="+",\
            help="min and max time step dt = 1/T (default [0.001, 1.0]")


    args = parser.parse_args()

    kwargs = dict(args._get_kwargs())

    entry_point = []
    entry_point.append(os.path.split(sys.argv[0])[1])
    args_list = sys.argv[1:]

    sorted_args = []
    for aa in range(0, len(args_list)):

        if "-" in args_list[aa]:
            sorted_args.append([args_list[aa]])
        else: 
            sorted_args[-1].append(args_list[aa])

            sorted_args.sort()
            entry_point.extend(sorted_args)

    kwargs["entry_point"] = entry_point

    # use subprocess to get the current git hash, store
    hash_command = ["git", "rev-parse", "--verify", "HEAD"]
    git_hash = subprocess.check_output(hash_command)
    # check_output returns bytes, convert to utf8 encoding before storing
    kwargs["git_hash"] = git_hash.decode("utf8")

    walk(**kwargs)
