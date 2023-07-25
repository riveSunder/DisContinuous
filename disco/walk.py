import os
import sys
import argparse
import subprocess

import uuid

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

def rxn_dfn_walk(**kwargs):

    hidden_channels = 256
    if kwargs["yuca_hash"] is not None:
        simulator_hash = kwargs["yuca_hash"] 
    else:
        os.system("cd yuca")
        sim_hash_command = ["git", "rev-parse", "--verify", "HEAD"]
        sim_hash = subprocess.check_output(hash_command)
        # check_output returns bytes, convert to utf8 encoding before storing
        simulator_hash = sim_hash.decode("utf8")
        os.system("../")

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

    max_dx = max(kwargs["dx_bounds"])
    min_dx = min(kwargs["dx_bounds"])

    min_kr = min(kwargs["kernel_radius_bounds"])
    max_kr = max(kwargs["kernel_radius_bounds"])

    # these are stop conditions that determine thresholds at which
    # pattern is considered lost
    min_gain = min(kwargs["gain_limits"])
    max_gain = max(kwargs["gain_limits"])

    min_cell_value = min(kwargs["cell_value_limits"])
    max_cell_value = max(kwargs["cell_value_limits"])

    min_correlation = min(kwargs["correlation_limits"])
    max_correlation = max(kwargs["correlation_limits"])

    if (min_gain + max_gain) < 0:
        use_gain = False
    else: 
        use_gain = True

    if (min_cell_value + max_cell_value) < 0:
        use_cell_value = False
    else:
        use_cell_value = True

    if (min_correlation + max_correlation) < 0:
        use_correlation = False
    else:
        use_correlation = True

    # how far we can step in our random walk
    max_scale_dx = max_dx * 0.25
    max_scale_dt = max_dt * 0.25
    max_scale_kr = max_kr * 0.25
    min_scale_dx = max_dx / 1000.0
    min_scale_dt = max_dt / 50.0
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

        my_uuid = str(uuid.uuid4()).replace("-","_")
        exp_folder = os.path.join("results", f"exp_{my_uuid}_{system_type}_{pattern_name}")
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
        log_results["simulator_hash"] = simulator_hash
        log_results["time_limit"] = time_limit
        log_results["random_seed"] = my_seed
        log_results["system_type"] = system_type
        log_results["epsilon"] = epsilon
        log_results["device"] = my_device
        log_results["max_dt"] = max_dt 
        log_results["min_dt"] = min_dt 
        log_results["max_dx"] = max_dx 
        log_results["min_dx"] = min_dx 
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

        kr = ca.get_kernel_radius()
        dt = ca.get_dt()
        dx = ca.get_dx()

        kr_scale = min_scale_kr
        dt_scale = min_scale_dt
        dx_scale = min_scale_dt

        last_persistence = 0
        persistence = 1

        results_list = ["uuid", "persistence", "max_persistence", \
                "bit", "dt", "kr", "dx", "image_filename"]
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

            ca.to_device(my_device)
            ca.no_grad()

            # set up and scale pattern
            my_scale = kr / native_radius
            use_anti_aliasing = True if my_scale < 1.0 else False

            while len(pattern.shape) < 4:
                pattern = pattern[None,...]

            if system_type == "RxnDfn":
                scaled_pattern = pattern
            else:
                scaled_pattern = skimage.transform.rescale(pattern, \
                        scale=(1,1, my_scale, my_scale), \
                        anti_aliasing=use_anti_aliasing,\
                        mode="constant", cval=0.0)

            # set up grid
            max_pattern_dim = max(scaled_pattern.shape)
            if max_pattern_dim >= 256:
                grid_size = max_pattern_dim + kr*2
                print(f"warning, scaled pattern is large (max dim: {max_pattern_dim}"\
                        f"using non-standard grid size {grid_size}")
            else:
                #standard grid size
                grid_size = 256

            mid_grid = (grid_size - max(scaled_pattern.shape)) // 2 

            grid = ca.initialize_grid(dim=grid_size)

            crop_x, crop_y = scaled_pattern.shape[-2], scaled_pattern.shape[-1]

            # send pattern to the intended fp precision dtype
            scaled_pattern = torch.tensor(scaled_pattern, dtype=torch.get_default_dtype())

            # add pattern to grid
            grid[:,:,mid_grid:mid_grid+crop_x, mid_grid:mid_grid+crop_y] = scaled_pattern
            grid = grid.to(my_device)

            # correlation and pattern pixel values will be used jointly to detect pattern dissolution
            starting_correlation = F.conv2d(\
                    grid.detach().cpu().to(torch.float32), \
                    scaled_pattern.detach().to(torch.float32)); 
            starting_correlation = starting_correlation / starting_correlation.std()

            starting_grid = 1.0 * grid
            starting_pattern = (1.0 * scaled_pattern).to(my_device)

            if system_type == "RxnDfn":
                setup_steps = 1000

                slope_dt = (ca.get_dt() - dt)  / setup_steps
                slope_dx = (ca.get_dx() - dx) / setup_steps


                for setup_step in range(setup_steps):

                    grid = ca(grid)

                    
                    new_dx = ca.get_dx() + slope_dx
                    new_dt = ca.get_dt() + slope_dt

                    ca.set_dx(new_dx)
                    ca.set_dt(new_dt)

                # set temporal step size and kernel radius
                ca.set_dt(dt)
                ca.set_dx(dx)
            ca.set_kernel_radius(kr)

            pattern_sum = grid.cpu().sum()

            t1 = time.time()
            failed = False
            for my_step in range(max_steps):

                grid = ca(grid)

                if use_correlation and (my_step % (max_steps // 32) == 0 \
                        or my_step == max_steps -1):

                    padded_g = F.pad(grid[0:1,0:1], (grid_size,0, grid_size, 0), \
                            mode="constant", value=starting_grid[0,0,-1,-1])

                    padded_s = 1.0 * starting_pattern[0:1,0:1] #starting_grid[0:1,0:1] 

                    padded_g -= padded_g.mean()
                    padded_s -= padded_s.mean()

                    correlate = F.conv2d(padded_g, padded_s[0:1,0:1], padding="same")
                    autocorrelate = F.conv2d(padded_g, padded_g, padding="same")

                    c_max = correlate.max() / correlate.std()

                    corr_autocorr = correlate.max() / autocorrelate.max()

                    print(f"step {my_step} \n"\
                            f"correlation.max / autocorrelation.max {corr_autocorr:.4f}")

                    print(f" correlation max / correlation std. dev. with grid_0 {c_max.item():3f}")

                    if corr_autocorr < min_correlation:
                        print("failed correlation test")
                        failed = True

                elif use_correlation:
                    pass
                else:
                    corr_autocorr = "na"
                    c_max = "na"

                if use_gain:
                    current_grid_sum = grid.sum()
                    gain = current_grid_sum.cpu() / pattern_sum

                    if (np.abs(1. - gain) > max_gain):
                        failed = True
                else:
                    gain = "na"

                if use_cell_value:
                    if grid[0,0].mean() <= min_cell_value or grid[0,0].mean() >= max_cell_value:
                        print("failed mean cell value test")
                        failed = True

                if np.isfinite(grid.mean().item()):
                    pass
                else:
                    print("failed finite test")
                    failed = True
                if failed:
                    break
                    


            kr_string = f"{kr}"
            dt_string = f"{dt:.6f}".replace(".","_")

            if system_type == "RxnDfn":
                dx_str = f"{dx:.6f}".replace(".","_")
            else:
                dx_str = "na"

            if use_correlation:
                corr_str = f"{c_max:.3f}".replace(".","_")
                autocorr_str = f"{corr_autocorr:.3f}".replace(".","_")
            else:
                corr_str = "na" 
                autocorr_str = "na" 

            if use_gain:
                gain_str = f"{gain:.3f}".replace(".","_")
            else:
                gain_str = "na"

            fp_string = f"{int(bit)}"

            t2 = time.time()
            just_elapsed = t2 - t1

            
            if failed:
                print(f"\nloss of pattern homeostasis at step {my_step} assumed")

            print(f"\n{pattern_name} persisted up to to step {my_step} uuid: {my_uuid}")

            print(f"     gain: {gain_str}")
            print(f"     normalized correlation: {autocorr_str}")
            print(f"     dt {dt:.4f}, dx {dx_str}, radius {kr}, {bit}-bit precision")
            print(f"     t_count: {np.array(ca.t_count):.3f}")

            print(f"     kr_scale: {kr_scale:.2f}, dt_scale: {dt_scale:.4f}")
            print(f"elapsed {just_elapsed:.3f} s {(my_step / just_elapsed):.3f} steps/s")

            if save_images:

                image_fn = f"final_grid_{pattern_name}_step{my_step}_"\
                        f"kr{kr_string}_dt{dt_string}_fp{fp_string}_dx{dx_str}_"\
                        f"autocorr{autocorr_str}_corr{corr_str}_gain{gain_str}.png"

                save_image_path = os.path.join(images_folder, image_fn)
                dgrid = starting_grid - grid
                dgrid = dgrid - dgrid.min()
                dgrid = (dgrid / dgrid.max())#.cpu()

                image_grid = torch.cat([starting_grid, grid, dgrid], -1)
                print(starting_grid[0,0].mean(), grid[0,0].mean())
                image_to_save = np.array(255*image_grid[0,0].squeeze().cpu().numpy(), dtype=np.uint8)
                print(f"kr: {kr}, grid size = {grid.shape}")
                sio.imsave(save_image_path, image_to_save)
            else:
                save_image_path = "no_image"



            # update random walk scale based on how different persistence was
            persistence = my_step

            m = (persistence + last_persistence) / (1 + 2 * np.abs(persistence - last_persistence))
            scale_modifier = 1. + (m-1)*2
            scale_modifier = max([min([scale_modifier, 1.5]), 0.5])

            if persistence == 0 and last_persistence == 0: 
                scale_modifier = 1.5

            print(f"scale modifier {scale_modifier}", m, persistence, last_persistence)
            last_persistence = persistence

            kr_scale *= scale_modifier #alpha * kr_scale + (1-alpha) * scale_modifier
            dt_scale *= scale_modifier #alpha * dt_scale + (1-alpha) * scale_modifier
            dx_scale *= scale_modifier #alpha * dt_scale + (1-alpha) * scale_modifier

            kr_scale = min([max_scale_kr, max([min_scale_kr, kr_scale])])
            dt_scale = min([max_scale_dt, max([min_scale_dt, dt_scale])])
            dx_scale = min([max_scale_dt, max([min_scale_dt, dt_scale])])

            elapsed = time.time() - t0

            results_list.append([my_uuid, persistence, max_steps, save_image_path, \
                    bit, dt, kr, dx])

            kr_shift = kr_scale * np.random.randn()
            dt_shift = dt_scale * np.random.randn()
            dx_shift = dx_scale * np.random.randn()

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

            if system_type == "RxnDfn":
                if np.isclose(dx, max_dx, rtol=0.001) and np.sign(dx_shift) > 0:
                    dx_shift *= -1.
                    dx = dx + dx_shift
                    dx = min([max([dx, min_dx]), max_dx])
                elif np.isclose(dx, min_dx, rtol=0.001) and np.sign(dx_shift) < 0:
                    dx_shift *= -1.
            else:
                dx = "na"

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

        log_results["results_file"] = results_path 

        print(f"saving results to {results_path}")
        np.save(results_path, np.array(results_list)) 

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
    parser.add_argument("-b", "--gain_bounds", type=float, default=[0.75, 1.25], nargs="+",\
            help="gain limits before assuming dissolution")
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
    parser.add_argument("-x", "--dx_bounds", type=float, default=[0.00150, 0.0145], nargs="+",\
            help="min and max dx, corresponding to spatial resolution scaler")
    parser.add_argument("-y", "--yuca_hash", type=str, default=None,\
            help="git commit hash for yuca ca simulator")

    parser.add_argument("--gain_limits", type=float, default=[-1.0, -1.0], nargs="+",
            help="stop condition: "\
                    "maximum absolute gain/loss in mean cell value, e.g. .25 is 25% growth/loss")
    parser.add_argument("--cell_value_limits", type=float, default=[-1.0, -1.0], nargs="+",
            help="stop condition: "\
                    "minimum and maximum mean cell values")
    parser.add_argument("--correlation_limits", type=float, default=[-1.0, -1.0], nargs="+",
            help="stop condition: "\
                    "minimum and maximum peak correlation,"\
                    "normalized to peak autocorrelation of initial pattern with itself")


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

    rxn_dfn_walk(**kwargs)
