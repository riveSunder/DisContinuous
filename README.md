# Self-organization experiments in discrete approximations of complex continuous systems

Checkout [the project page](https://rivesunder.github.io/DisContinuous) for an outline of the project, links to code illustrating discretization effects on _Scutium gravidus_ and _Orbium_ Lenia gliders, and preliminary results (animations).

## Quick setup

This repo uses the [yuca](https://github.com/rivesunder/yuca) cellular automata simulator. 

To get started you can clone this repo and run `clone_install.sh` from within your virtual environment. This will install `yuca`. 

```
git clone https://github.com/riveSunder/DisContinuous.git disco
cd disco

virtualenv disco_env --python=python3.8
source disco_env/bin/activate

sh clone_install.sh

# try and run yuca tests
pytohn -m testing.test_all
```

Afterward you should be able to run the [Notebook](https://github.com/riveSunder/DisContinuous/blob/master/notebooks/disc_demo.ipynb). But you can also try it on: [mybinder](https://mybinder.org/v2/gh/rivesunder/DisContinuous/master?labpath=notebooks%2Fdisco_demo.ipynb) -> [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rivesunder/DisContinuous/master?labpath=notebooks%2Fdisco_demo.ipynb), or [colab](https://colab.research.google.com/github/rivesunder/DisContinuous/blob/master/notebooks/disco_demo.ipynb) -> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rivesunder/DisContinuous/blob/master/notebooks/disco_demo.ipynb) 
