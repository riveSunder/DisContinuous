import os

script_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
log_directory = os.path.join(script_path, "logs", "exp_0x00")
log_filepath_gol = os.path.join(log_directory, "exp_0x00_ca_gol_parameter_walk_log.txt")

if os.path.exists(log_directory):
    pass
else:
    os.mkdir(log_directory)

time_limit = 7200 #14400
system_type = "CCA"
save_images = "0"
max_steps = "1024"
device = "cuda"
kernel_radius_bounds = "1 31"
glider_patterns = f" gol_reflex_glider000 gol_small_spaceship000 gol_medium_spaceship000 gol_large_spaceship000 "

my_cmd_gol = f"python -m disco.walk -l {time_limit} -c {system_type} "\
        f"-g {glider_patterns}"\
        f"-s {save_images} -m {max_steps} -d {device} -k {kernel_radius_bounds} > {log_filepath_gol}"

my_cmd_gol_images = f"python -m disco.walk -l {900} -c {system_type} "\
        f"-g {glider_patterns}"\
        f"-s 1 -m {max_steps} -d {device} -k {kernel_radius_bounds} > {log_filepath_gol}"

os.system(my_cmd_gol_images)
os.system(my_cmd_gol)
