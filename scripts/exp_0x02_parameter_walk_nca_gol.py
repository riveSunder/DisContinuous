import os

script_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
log_directory = os.path.join(script_path, "logs", "exp_0x02")
log_filepath_nca = os.path.join(log_directory, "exp_0x02_nca_parameter_walk_log.txt")

if os.path.exists(log_directory):
    pass
else:
    os.mkdir(log_directory)

time_limit = 7200
system_type = "NCA"
save_images = "0"
max_steps = "1024"
device = "cuda"
kernel_radius_bounds = "1 31"
glider_patterns = f" neuroreflex_glider000 "

my_cmd_nca_gol = f"python -m disco.walk -l {time_limit} -c {system_type} "\
        f"-g {glider_patterns}"\
        f"-s {save_images} -m {max_steps} -d {device} -k {kernel_radius_bounds} > {log_filepath_nca}_gol"

os.system(my_cmd_nca_gol)
