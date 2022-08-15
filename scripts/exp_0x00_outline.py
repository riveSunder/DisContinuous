import os

import time

from disca.rule_pattern import RulePattern
# RulePattern combines CPPN synthesis pattern generation 
# with universe rules (e.g. CA parameters) 

from disca.universe_glider_wrapper import UniverseGliderWrapper
# UniverseGliderWrapper wraps the complex system (Lenia, Glaberish, NCA, Gray-Scott, .... etc.)
# and provides glider based reward as a result of adjusting the universe rules and the initial patterns 

from disca.coevo import coevolve
# coevolve(template, algo, universe_type=CA, **kwargs)
# template is a population individual with methods:
#   get_parameters, set_parameters, get_action
# kwargs includes population size, generations, change rate, etc.  

from disca.nes import NES
# NES is a natural evolution strategy algo class, handling the selection and update steps 

from disca.map_support import map_support
# map_support(pattern_name, kernel_sizes, dtypes, dts, exp_tag, growth_threshold, smooth=False)

# Lenia framework
from yuca.lenia import Lenia
# Glaberish framework
from yuca.multiverse import CA
# neural cellular automata
from yuca.nca import NCA
# Gray-Scott system
from disca.gray_scott import GrayScott

# discretization parameters for mapping support
my_kernel_sizes = [3, 5, 9, 13, 27, 39, 53, 65, 79, 91, 103, 117, 131]
my_dtypes = [torch.float64, torch.float32, torch.float16]
my_dts = [0.01, 0.03, 0.09, 0.15, 0.21, 0.27, 0.33, 0.39, 0.45, 0.51, 0.57, 0.62, 0.70, 0.8, 0.9, 0.93, 0.96, 0.98, 0.99]

# previously described patterns (support to be mapped)
# _Scutium gravidus_ and primordial _Orbium_ are both in Lenia's 'Exokernel' class
# _Hydrogeminium natans_ is a Mesokernel rule
my_patterns = ["scutium_gravidus_single000", \
        "orbium_orbium000", \
        "hydrogeminium_natans_000"]

# I haven't captured these patterns yet, so they don't have entries in the librarian index
# aim for two pattern/rule pairs from each Class (so if I can't catch one pair in each class, that's OK)

# * _Pyroscutium ambiguus_ (from class Exokernel)
# * _Aerogeminium volitans_ (from class Mesokernel)
# * _Aerogeminium ambiguus_ (from class Mesokernel)
# * _Kronium vagus_ (from class Endokernel)
# * _Kronium saliens_ (from class Endokernel)
# * _Ferrokronium solidus_ (from class Endokernel)

# coevolve rule/pattern pairs
# aim for three new pairs for each complex system

kwargs = {"max_generations": 64,\
        "population_size": 64, \
        "save_directory": os.path.join("parameters", "coevolved")}

# in Lenia
coevolve(template=RulePair, algo=NES, universe_type=Lenia, **kwargs)
# in Glaberish
coevolve(template=RulePair, algo=NES, universe_type=CA, **kwargs)
# With Neural Cellular Automata
coevolve(template=RulePair, algo=NES, universe_type=NCA, **kwargs)
# in Gray-Scott reaction diffusion system
coevolve(template=RulePair, algo=NES, universe_type=GrayScott, **kwargs)

# map discretization support for known pattern/rule pairs 
# growth threshold: how much can a glider grow before it's considered to lose it's identity?
growth_threshold = 0.5

for pattern in my_patterns:

    timestap = str(int(time.time()))[-4:]
    exp_tag = f"{pattern}_support_{timestamp}"

    my_support = map_support(pattern, my_kernel_sizes, my_dtypes, my_dts, exp_tag, growth_threshold, smooth=False)

    # save support results, write to csv
    write_filename = f"{exp_tag}.csv"
    write_name = os.path.join("parameters","support",write_filename)

    with open(write_name, "w") as f:
        for line in my_support:
            f.writelines(line)

# map discretization support for newly evolved pattern/rule pairs 

my_dir_list = os.path.listdir(os.path.join("parameters", "coevolved"))

for element in my_dir_list:

    pattern = os.path.splitext(element)

    timestap = str(int(time.time()))[-4:]
    exp_tag = f"{pattern}_support_{timestamp}"

    my_support = map_support(pattern, my_kernel_sizes, my_dtypes, my_dts, exp_tag, growth_threshold, smooth=False)

    # save support results, write to csv
    write_filename = f"{exp_tag}.csv"
    write_name = os.path.join("parameters","support",write_filename)

    with open(write_name, "w") as f:
        for line in my_support:
            f.writelines(line)

