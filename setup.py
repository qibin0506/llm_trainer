from setuptools import setup, find_packages

# 1. python3 setup.py sdist bdist_wheel
# 2. pip3 install dist/project_llm_trainer-0.1.tar.gz
setup(
    name='project_llm_trainer',
    version='0.2.1',
    description='llm trainer',
    scripts=[
        'scripts/smart_train',
        'scripts/ds_train',
        'scripts/ddp_train',
        'scripts/py_train',
        'scripts/plot_lr',
        'scripts/plot_loss',
        'scripts/calc_intermediate_size'
    ],
    # package_data={'': ['*.pyc']},
    # exclude_package_data={'': ['*.py']},
    # include_package_data=True,
    author='qibin',
    author_email='qibin0506@gmail.com',
    packages=find_packages()
)
