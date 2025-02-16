from setuptools import setup, find_packages

long_description = """

# Rocket Learn

Learning!

"""

setup(
   name='rocket_learn',
   version='0.2.8',
   description='Rocket Learn',
   author='Rolv-Arild Braaten, Daniel Downs, Forked by Ronak Malde',
   url='https://github.com/rmalde/rocket-learn-sim',
   packages=[package for package in find_packages() if package.startswith("rocket_learn")],
   long_description=long_description,
   install_requires=['cloudpickle==1.6.0', 'gym', 'torch', 'tqdm', 'trueskill',
                     'msgpack_numpy', 'wandb', 'pygame', 'keyboard', 'tabulate',
                     'rich', 'redis', 'rocketsim', 'numba', 'plotly', 'python-dotenv'],
   # will need to separately pip install git+https://github.com/AechPro/rocket-league-gym-sim@main
)
