import os

script_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
log_directory = os.path.join(script_path, "logs", "exp_0x02")
log_filepath_rxn = os.path.join(log_directory, "exp_0x04_rxn.txt")
log_filepath_rxn_audit = os.path.join(log_directory, "exp_0x04_rxn_audit.txt")

if os.path.exists(log_directory):
    pass
else:
    os.mkdir(log_directory)

time_limit = 3600
system_type = "RxnDfn"
save_images = "0"
max_steps = "65536"
device = "cuda"
kernel_radius_bounds = "1 1"
glider_patterns = f"uskate_glider001 uskate_daedalus001 uskate_berrycup001 "

my_cmd_rxn = f"python -m disco.walk -l {time_limit} -c {system_type} "\
        f"-g {glider_patterns}"\
        f"-s {save_images} -m {max_steps} -d {device} -k {kernel_radius_bounds} > {log_filepath_rxn}"

my_cmd_rxn_audit = f"python -m disco.walk -l 900 -c {system_type} "\
        f"-g {glider_patterns} -b 0.6 1.4 "\
        f"-s 1 -m {max_steps} -d {device} -k {kernel_radius_bounds} > {log_filepath_rxn_audit}"

os.system(my_cmd_rxn)
print("Now running sims and saving final images")
os.system(my_cmd_rxn_audit)
