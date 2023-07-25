import os

script_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
log_directory = os.path.join(script_path, "logs", "exp_0x06_dry_run")
log_filepath_rxn = os.path.join(log_directory, "exp_0x0r06_rxn.txt")

if os.path.exists(log_directory):
    pass
else:
    os.mkdir(log_directory)

# u-skate patterns
glider_patterns = f"uskate_glider001 uskate_daedalus001 uskate_berrycup001 "

time_limit = 7200
system_type = "RxnDfn"
save_images = "1"
max_steps = "131072"
device = "cuda"

kernel_radius_bounds = "1 1"
dx_bounds = "0.001 0.015"

# gliders in u-skate world travel in straight lines, 
# the failure condition is based max correlation with the starting pattern normalized to 
# max autocorrelation of the starting pattern with itself
correlation_limits = "0.8 5.0"

my_cmd_rxn = f"python -m disco.walk -l {time_limit} -c {system_type} "\
        f"-g {glider_patterns}"\
        f"-s {save_images} -m {max_steps} -d {device}  -x {dx_bounds} "\
        f"-y 5c5cf4861bfa955477694f887fef2a6e1905ffdd "\
        f"--correlation_limits {correlation_limits} "\
        f"-k {kernel_radius_bounds} > {log_filepath_rxn}"

# patterns from Lenia
log_filepath_lenia = os.path.join(log_directory, "exp_0x0r06_lenia.txt")
glider_patterns = f" orbium_orbium000 geminium_hydrogeminium_natans000 scutium_gravidus_single000 "\
        f" triscutium_solidus_triscutium000 geminium2_wobble_glider000 "
system_type = "CCA"
max_steps = "4096"
device = "cuda"
kernel_radius_bounds = "1 101"

# gliders in Lenia tend to rotate a bit, so gain in mean cell value is used instead of correlation, 
gain_limits = "0.30"

my_cmd_lenia = f"python -m disco.walk -l {time_limit} -c {system_type} "\
        f"-g {glider_patterns}"\
        f"-s {save_images} -m {max_steps} -d {device}  -x {dx_bounds} "\
        f"-y 5c5cf4861bfa955477694f887fef2a6e1905ffdd "\
        f"--gain_limits {gain_limits} " \
        f"-k {kernel_radius_bounds} > {log_filepath_lenia}"

log_filepath_smoothlife = os.path.join(log_directory, "exp_0x0r06_smoothlife.txt")
glider_patterns = f" smoothlife_single_glider000 "

# gliders in SmoothLife tend to rotate a bit, so gain in mean cell value is used instead of correlation, 
gain_limits = "0.30"

my_cmd_smoothlife = f"python -m disco.walk -l {time_limit} -c {system_type} "\
        f"-g {glider_patterns}"\
        f"-s {save_images} -m {max_steps} -d {device}  -x {dx_bounds} "\
        f"-y 5c5cf4861bfa955477694f887fef2a6e1905ffdd "\
        f"--gain_limits {gain_limits} " \
        f"-k {kernel_radius_bounds} > {log_filepath_smoothlife}"

# Experiment applied to gliders from Conway's Game of Life
log_filepath_gol = os.path.join(log_directory, "exp_0x0r06_game_of_life.txt")
glider_patterns = f" gol_25P3H1V0_1000 gol_reflex_glider000 gol_small_spaceship000 "

# gliders in Life mostly maintain orientation, but can take different shapes from step to step
# therefore, gain is used as the dissolution threshold
kernel_radius_bounds = "1 31"
gain_limits = "0.30"

my_cmd_gol = f"python -m disco.walk -l {time_limit} -c {system_type} "\
        f"-g {glider_patterns}"\
        f"-s {save_images} -m {max_steps} -d {device}  -x {dx_bounds} "\
        f"-y 5c5cf4861bfa955477694f887fef2a6e1905ffdd "\
        f"--gain_limits {gain_limits} " \
        f"-k {kernel_radius_bounds} > {log_filepath_gol}"

log_filepath_glaberish = os.path.join(log_directory, "exp_0x0r06_glaberish.txt")
glider_patterns = f" s11_config_evolved_ca_slow_glider000 s613_s613_frog000 s643_s643_frog000 "\
        " s613_fast_wobble_glider000 s643_s643_3wide_wobble_glider000 "

# gliders in Life mostly maintain orientation, but can take different shapes from step to step
# therefore, gain is used as the dissolution threshold
gain_limits = "0.30"

my_cmd_glaberish = f"python -m disco.walk -l {time_limit} -c {system_type} "\
        f"-g {glider_patterns}"\
        f"-s {save_images} -m {max_steps} -d {device}  -x {dx_bounds} "\
        f"-y 5c5cf4861bfa955477694f887fef2a6e1905ffdd "\
        f"--gain_limits {gain_limits} " \
        f"-k {kernel_radius_bounds} > {log_filepath_glaberish}"

# patterns from NCA
log_filepath_nca = os.path.join(log_directory, "exp_0x0r06_nca.txt")
glider_patterns = " neurosingle_glider000 neurosynorbium000 neurorbium000 neurowobble_glider000 "
system_type = "NCA"
max_steps = "4096"
device = "cuda"
kernel_radius_bounds = "1 101"

# NCA gliders exist under rules that are essentially (and poorly) cloned from Lenia and glaberish
gain_limits = "0.30"

my_cmd_nca = f"python -m disco.walk -l {time_limit} -c {system_type} "\
        f"-g {glider_patterns}"\
        f"-s {save_images} -m {max_steps} -d {device}  -x {dx_bounds} "\
        f"-y 5c5cf4861bfa955477694f887fef2a6e1905ffdd "\
        f"--gain_limits {gain_limits} " \
        f"-k {kernel_radius_bounds} > {log_filepath_nca}"

print("Running experiment in U-Skate World framework")
print(my_cmd_rxn)
os.system(my_cmd_rxn)
print("Running experiment in Lenia framework")
print(my_cmd_lenia)
os.system(my_cmd_lenia)
print("Running experiment in SmoothLife framework")
print(my_cmd_smoothlife)
os.system(my_cmd_smoothlife)
print("Running experiment in Game of Life")
print(my_cmd_gol)
os.system(my_cmd_gol)
print("Running experiment in Glaberish framework")
print(my_cmd_glaberish)
os.system(my_cmd_glaberish)
print("Running experiment in Neural Celllular Automata framework")
print(my_cmd_nca)
os.system(my_cmd_nca)
