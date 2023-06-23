import os
import sys
import argparse
import subprocess

import time
import json

import numpy as np
import torch

import yuca
from yuca.zoo.librarian import Librarian
from yuca.ca.continuous import CCA
from yuca.ca.neural import NCA
from yuca.utils import query_kwargs

import skimage
import skimage.transform

import matplotlib.pyplot as plt

def sweep(**kwargs):

    time_step = query_kwargs("time_step", [10], **kwargs)
    kernel_radius = query_kwargs("kernel_radius", [13], **kwargs)
    numerical_precision = query_kwargs("numerical_precision", [32], **kwargs)
    pattern_names = query_kwargs("pattern", ["orbium_orbium000"], **kwargs)
    max_steps = query_kwargs("max_steps", 128, **kwargs)
    my_device =  query_kwargs("device", "cpu", **kwargs)



    # get lists in order
    time_step.sort()
    kernel_radius.sort()
    numerical_precision.sort()
    remove_list = []

    for gg in range(len(numerical_precision)):
        if numerical_precision[gg] != 32:
            if numerical_precision[gg] != 64:
                print(f" floating point precision of {numerical_precision[gg]}" \
                        f" not supported. Dropping from experiment.")
                remove_list.append(numerical_precision[gg])

    for remove_element in remove_list:
        numerical_precision.remove(remove_element)

    results = {}
    log_results = {}
    log_results["entry_point"] = query_kwargs("entry_point", "none", **kwargs)
    log_results["git_hash"] = query_kwargs("git_hash", "none", **kwargs)
    time_stamp = str(int(time.time()*1000))[-4:]

    log_results_path = f"metadata.json"

    # lists for axes labels
    names = []

    lib = Librarian()

    for hh, pattern_name in enumerate(pattern_names):
        names.append(pattern_name)
        results[pattern_name] = np.zeros((\
                len(numerical_precision),\
                len(kernel_radius),\
                len(time_step)))

        prec = []
        for ii, precision in enumerate(numerical_precision):
            prec.append(precision)

            if precision == 16:
                assert False, "float16 support not implemented yet"
                torch.set_default_dtype(torch.float16)
            elif precision == 32:
                torch.set_default_dtype(torch.float32)
                float16_workaround = False
            elif precision == 64:
                torch.set_default_dtype(torch.float64)
                float16_workaround = False

            pattern, metadata = lib.load(pattern_name)
            
            ca = CCA()
            ca.restore_config(metadata["ca_config"])
            ca.no_grad()
            native_config = ca.make_config()

            native_radius = native_config["neighborhood_kernel_config"]["radius"]

            kr = []
            for jj, radius in enumerate(kernel_radius):
                kr.append(radius)

                T = []
                for kk, dt in enumerate([1/elem for elem in time_step]):
                    T.append(1/dt)

                    grid_size = int(5 * radius)
                    mid_grid = grid_size // 2 

                    my_scale = radius / native_radius
                    use_anti_aliasing = True if my_scale < 1.0 else False
                    while len(pattern.shape) < 4:
                        pattern = pattern[None,...]
                    scaled_pattern = skimage.transform.rescale(pattern, \
                            scale=(1,1, my_scale, my_scale), \
                            anti_aliasing=use_anti_aliasing)

                    crop_x, crop_y = scaled_pattern.shape[-2], scaled_pattern.shape[-1]

                    grid = torch.zeros(1,1,grid_size, grid_size)

                    scaled_pattern = torch.tensor(scaled_pattern, dtype=torch.get_default_dtype())
                    grid[:,:,mid_grid:mid_grid+crop_x, mid_grid:mid_grid+crop_y] = scaled_pattern


                    if float16_workaround:
                        grid = grid.to(torch.float16)

                    pattern_sum = grid.sum()

                    ca.change_kernel_radius(radius=radius)
                    ca.dt = dt

                    ca.to_device(my_device)
                    grid = grid.to(my_device)
                    for step in range(max_steps):

                        grid = ca(grid)

                        gain = grid.sum() / pattern_sum

                        if gain > 1.25 or gain < 0.75:
                            print(f"loss of pattern homeostasis at step {step} assumed")
                            print(f"     pattern: {pattern_name}, dt {dt}, "\
                                    f"radius {radius}, precision {precision}")
                            break

                    results[pattern_name][ii,jj,kk] = step

    log_results["kernel_radii"] = kr
    log_results["precision"] = prec
    log_results["time_steps"] = T

    exp_folder = os.path.join("results", f"exp_{time_stamp}")
    if os.path.exists(exp_folder):
        pass
    else:
        os.mkdir(exp_folder)

    for pattern_name in results.keys():
        results_path = os.path.join(exp_folder, f"results_{pattern_name}.npy")
        np.save(results_path, results[pattern_name])

    with open(os.path.join(exp_folder, log_results_path), "w") as f:
        
        json.dump(log_results, f)
        




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-T", "--time_step", type=float, default=[10.], nargs="+",\
            help="number of simulation steps per time unit. dt = 1/T")
    parser.add_argument("-K", "--kernel_radius", type=int, default=[13], nargs="+",\
            help="neighborhood kernel radius(es) in pixels. dxy = 1/K")

    parser.add_argument("-d", "--device", type=str, \
            default="cpu", help="device to use (cpu or cuda)")
    parser.add_argument("-m", "--max_steps", type=int, default=2048,\
            help="maximum number of simulation steps ")

    parser.add_argument("-P", "--numerical_precision", type=int, default=[32], nargs="+",\
            help="numerical precision of floating points, options: 32 and 64 bits.")

    parser.add_argument("-p", "--pattern", type=str, default=["orbium_orbium000"], nargs="+", \
            help="name of the pattern(s) to use in discretization precision sweep")

    parser.add_argument("-s", "--save_video", type=bool, default=False)

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

    sweep(**kwargs)
