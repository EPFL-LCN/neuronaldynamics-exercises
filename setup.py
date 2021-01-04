from setuptools import setup, find_packages

try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements
# get requirements from requirements.txt
install_reqs = parse_requirements('requirements.txt', session=False)
try:
    reqs = [str(ir.req) for ir in install_reqs]
except:
    reqs = [str(ir.requirement) for ir in install_reqs]
# find packages
prefix = 'neurodynex3'
packages = find_packages(where=prefix, exclude=[])
packages_pre = ["%s.%s" % (prefix, s) for s in packages]

setup(
  name='neurodynex3',
  version = '1.0.0',
  packages=find_packages(),
  package_data={
    'neurodynex3': ['data/*'],
  },
  description='Python exercises accompanying the book Neuronal Dynamics by Wulfram Gerstner, Werner M. Kistler, Richard Naud and Liam Paninski.',
  author='LCN-EPFL',
  author_email='martin.barry@epfl.ch',
  url='https://github.com/martinbarry59/neurodynex3',  # use the URL to the github repo
  download_url ='https://github.com/martinbarry59/neurodynex3/archive/1.0.0.tar.gz',
  keywords=['compneuro', 'science', 'teaching', 'neuroscience', 'brian'],  # arbitrary keywords
  classifiers=[
    'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
    'Topic :: Scientific/Engineering :: Medical Science Apps.',
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
  ],
  python_requires='>=3.6',
  # use_scm_version=True,
  setup_requires=['setuptools_scm'],
  install_requires=reqs,
  license='GPLv2',
)
