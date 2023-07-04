import os

script_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
log_directory = os.path.join(script_path, "logs", "exp_0x02")
log_filepath_nca = os.path.join(log_directory, "exp_0x02_nca_parameter_walk_log.txt")
log_filepath_cca = os.path.join(log_directory, "exp_0x02_cca_parameter_walk_log.txt")

if os.path.exists(log_directory):
    pass
else:
    os.mkdir(log_directory)

time_limit = 7200
system_type = "NCA"
save_images = "0"
max_steps = "1024"
device = "cuda"
kernel_radius_bounds = "1 111"
glider_patterns = f" neurowobble_glider000 "

my_cmd_nca = f"python -m disco.walk -l {time_limit} -c {system_type} "\
        f"-g {glider_patterns}"\
        f"-s {save_images} -m {max_steps} -d {device} -k {kernel_radius_bounds} > {log_filepath_nca}"

my_cmd_nca_audit = f"python -m disco.walk -l 900 -c {system_type} "\
        f"-g {glider_patterns}"\
        f"-s 1 -m {max_steps} -d {device} -k {kernel_radius_bounds} > {log_filepath_nca}_images"

time_limit = 7200
system_type = "CCA"
save_images = "0"
max_steps = "1024"
device = "cuda"
kernel_radius_bounds = "1 111"
glider_patterns = f"s613_fast_wobble_glider000 s643_s643_3wide_wobble_glider000 geminium2_wobble_glider "

my_cmd_cca = f"python -m disco.walk -l {time_limit} -c {system_type} "\
        f"-g {glider_patterns}"\
        f"-s {save_images} -m {max_steps} -d {device} -k {kernel_radius_bounds} > {log_filepath_cca}"

my_cmd_cca_audit = f"python -m disco.walk -l 900 -c {system_type} "\
        f"-g {glider_patterns}"\
        f"-s 1 -m {max_steps} -d {device} -k {kernel_radius_bounds} > {log_filepath_cca}_images"

os.system(my_cmd_cca)
os.system(my_cmd_nca)

print("Now running sims and saving final images")

os.system(my_cmd_cca_audit)
os.system(my_cmd_nca_audit)
