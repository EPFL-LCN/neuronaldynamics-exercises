all: pypi

pypi:
	rm -rf dist/*
	python setup.py bdist_wheel sdist
	twine upload dist/*

conda: conda-build conda-upload

conda-build:
	rm -rf conda_build/build
	conda build conda_build
	@read -p "Enter Path:" cpath; \
	conda convert --platform all $$cpath -o conda_build/build;

conda-upload:
	anaconda upload  --user flinz conda_build/build/win-32/*
	anaconda upload  --user flinz conda_build/build/win-64/*
	anaconda upload  --user flinz conda_build/build/linux-32/*
	anaconda upload  --user flinz conda_build/build/linux-64/*
	anaconda upload  --user flinz conda_build/build/osx-64/*

