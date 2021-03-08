deploy: test pypi
#deploy: test conda

# # Documentation building
#
# sphinx:
# 	sphinx-apidoc -o doc/modules neurodynex3 -f
# 	make -C doc html
# Pypi deployment and documetnation building

pypi:
	sphinx-apidoc -o doc/modules neurodynex3 -f
	make -C doc html
	
	rm -rf dist/*
	python3 setup.py bdist_wheel sdist
	twine upload dist/* --config-file .pypirc

# Anaconda deployment

#conda: conda-build conda-upload

#conda-build:
#	rm -rf conda_build/build
#	conda build conda_build
# 	@read -p "Enter the path to local file from above ['anaconda upload PATH']: " cpath; \
# 	conda convert --platform all $$cpath -o conda_build/build;
#
# conda-upload:
# 	anaconda login
# 	anaconda upload -u epfl-lcn conda_build/build/win-32/*;
# 	anaconda upload -u epfl-lcn conda_build/build/win-64/*;
# 	anaconda upload -u epfl-lcn conda_build/build/linux-32/*;
# 	anaconda upload -u epfl-lcn conda_build/build/linux-64/*;
# 	anaconda upload -u epfl-lcn conda_build/build/osx-64/*;

# test
test:
	pycodestyle neurodynex3 --max-line-length=120
	nosetests neurodynex3 --nocapture --verbosity=2