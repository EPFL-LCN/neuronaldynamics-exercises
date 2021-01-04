import os
import sys


def run_nose():
    """Runs nose tests, used mainly for anaconda deployment"""
    import nose
    nose_argv = sys.argv
    nose_argv += ["--detailed-errors", "--exe", "-v"]
    initial_dir = os.getcwd()
    package_file = os.path.abspath(__file__)
    package_dir = os.path.dirname(package_file)
    os.chdir(package_dir)
    try:
        nose.run(argv=nose_argv)
    finally:
        os.chdir(initial_dir)
