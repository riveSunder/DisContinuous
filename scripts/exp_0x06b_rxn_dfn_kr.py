import os

script_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
log_directory = os.path.join(script_path, "logs", "exp_0x06b_rxn_dfn_kr")
log_filepath_rxn = os.path.join(log_directory, "exp_0x0r06b_rxn.txt")

if os.path.exists(log_directory):
    pass
else:
    os.mkdir(log_directory)

# u-skate patterns
glider_patterns = f"uskate_glider001 uskate_daedalus001 uskate_berrycup001 "

time_limit = 14400
system_type = "RxnDfn"
save_images = "1"
max_steps = "65536"
device = "cuda"

kernel_radius_bounds = "1 31"
dx_bounds = "0.0033 0.0033"

# gliders in u-skate world travel in straight lines, 
# the failure condition is based max correlation with the starting pattern normalized to 
# max autocorrelation of the starting pattern with itself
correlation_limits = "0.8 5.0"

my_cmd_rxn = f"python -m disco.walk -l {time_limit} -c {system_type} "\
        f"-g {glider_patterns}"\
        f"-s {save_images} -m {max_steps} -d {device}  -x {dx_bounds} "\
        f"-y a8f88c459999e23a3fabc8c8ffa3f27b4173a13b "\
        f"--correlation_limits {correlation_limits} "\
        f"-k {kernel_radius_bounds} > {log_filepath_rxn}"

print("Running experiment in U-Skate World framework")
print(my_cmd_rxn)
os.system(my_cmd_rxn)
