import os

script_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
log_directory = os.path.join(script_path, "logs", "exp_0x00")
log_filepath_nca = os.path.join(log_directory, "exp_0x00_nca_save_image_log.txt")
log_filepath_cca = os.path.join(log_directory, "exp_0x00_cca_save_image_log.txt")

if os.path.exists(log_directory):
    pass
else:
    os.mkdir(log_directory)

time_limit = 14400
system_type = "NCA"
save_images = "0"
max_steps = "1024"
device = "cuda"
kernel_radius_bounds = "3 111"
glider_patterns = f" neurosingle_glider000 neurosynorbium000 neuroo_bicaudatus_ignis000 "

my_cmd_nca = f"python -m disco.walk -l {time_limit} -c {system_type} "\
        f"-g {glider_patterns}"\
        f"-s {save_images} -m {max_steps} -d {device} -k {kernel_radius_bounds} > {log_filepath_nca}"

system_type_cca = "CCA"
glider_patterns_cca= f" exp_local_glaberish_geminium_simple_halt_1649264299_seed11_config_evolved_ca_slow_glider000 "\
        f"exp_local_glaberish_geminium_simple_halt_1649264299_seed7_config_spiky_frog_egg000 "\
        f"s613_s613_frog000 "\
        f"s643_s643_frog000 "\
        f"smoothlife_single_glider000 "\
        f"scutium_gravidus_single000 "\
        f"geminium_hydrogeminium_natans000 "\
        f"orbium_orbium000 "\
        f"synorbium_orbium000 "\
        f"triscutium_solidus_triscutium000 "

my_cmd_cca = f"python -m disco.walk -l {time_limit} -c {system_type_cca} "\
        f"-g {glider_patterns_cca}"\
        f"-s {save_images} -m {max_steps} -d {device} -k {kernel_radius_bounds} > {log_filepath_cca}"

os.system(my_cmd_nca)
os.system(my_cmd_cca)
