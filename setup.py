from setuptools import setup, find_packages
from pip.req import parse_requirements

# get requirements from requirements.txt
install_reqs = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]

# find packages
prefix = 'neurodynex'
packages = find_packages(where=prefix, exclude=[])
packages_pre = ["%s.%s" % (prefix, s) for s in packages]

setup(
  name='neurodynex',
  packages=find_packages(),
  package_data={
    'neurodynex': ['data/*'],
  },
  description='Python exercises accompanying the book Neuronal Dynamics by Wulfram Gerstner, Werner M. Kistler, Richard Naud and Liam Paninski.',
  author='LCN-EPFL',
  author_email='alex.seeholzer@epfl.ch',
  url='https://github.com/EPFL-LCN/neuronaldynamics-exercises',  # use the URL to the github repo
  keywords=['compneuro', 'science', 'teaching', 'neuroscience', 'brian'],  # arbitrary keywords
  classifiers=[
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.4',
    'Topic :: Scientific/Engineering :: Medical Science Apps.'
  ],
  use_scm_version=True,
  setup_requires=['setuptools_scm'],
  install_requires=reqs,
  license='GPLv2',
)
