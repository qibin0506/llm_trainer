rm -fr build
rm -fr dist
rm -fr *.egg-info

python3 setup.py sdist bdist_wheel