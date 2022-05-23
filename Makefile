
dist_setup:
	python3 -m pip install --upgrade pip
	python3 -m pip install --upgrade build
	python3 -m pip install --upgrade twine

dist:
	python3 -m build
	python3 -m twine upload -u __token__ -p pypi-AgENdGVzdC5weXBpLm9yZwIkNmQ0NjIxZTgtYWU2MC00MjI3LTkxOWEtZmFhNzY3MmM3OTZjAAIleyJwZXJtaXNzaW9ucyI6ICJ1c2VyIiwgInZlcnNpb24iOiAxfQAABiDoN3DoETZ3j1c2HDYwkQIfs-NxbQ9P0w3UB2Lvap-igw  --skip-existing --repository testpypi dist/* --verbose
	

pip_install_pypi:
	python3 -m pip install -i https://test.pypi.org/simple/ --no-deps rosemary

pip_install_editable:
	# creates platform-independent `egg-link` file in site-package/ linked to current directory
	python3 -m pip -v install -e /data/vision/polina/scratch/wpq/github/code_snippets #  --log log.txt
	# check link installed properly
	python3 -m pip list | grep rosemary

pip_install:
	python3 -m pip install .