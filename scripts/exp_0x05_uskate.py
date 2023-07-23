import os

script_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
log_directory = os.path.join(script_path, "logs", "exp_0x05")
log_filepath_rxn = os.path.join(log_directory, "exp_0x0r05_rxn.txt")
log_filepath_rxn_audit = os.path.join(log_directory, "exp_0x05_rxn_audit.txt")

if os.path.exists(log_directory):
    pass
else:
    os.mkdir(log_directory)

time_limit = 7200
system_type = "RxnDfn"
save_images = "0"
max_steps = "131072"
device = "cuda"
kernel_radius_bounds = "1 1"
dx_bounds = "0.001 0.015"
glider_patterns = f"uskate_glider001 uskate_daedalus001 uskate_berrycup001 "

my_cmd_rxn = f"python -m disco.rxn_dfn_walk -l {time_limit} -c {system_type} "\
        f"-g {glider_patterns}"\
        f"-s {save_images} -m {max_steps} -d {device}  -x {dx_bounds} "\
        f"-y db738125121545cfe58e5e62db7b51634c36350f "\
        f"-k {kernel_radius_bounds} > {log_filepath_rxn}"

my_cmd_rxn_audit = f"python -m disco.rxn_dfn_walk -l 900 -c {system_type} "\
        f"-g {glider_patterns} -b 0.6 1.4 "\
        f"-s 1 -m {max_steps} -d {device} -x {dx_bounds} "\
        f"-y db738125121545cfe58e5e62db7b51634c36350f "\
        f"-k {kernel_radius_bounds} > {log_filepath_rxn_audit}"

print("Now running sims and saving final images")
os.system(my_cmd_rxn_audit)

os.system(my_cmd_rxn)
