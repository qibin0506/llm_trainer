from setuptools import setup, find_packages

setup(
    name='project_llm_trainer',
    version='0.10.2',
    description='LLM and VLM trainer',
    scripts=[
        'scripts/smart_train',
        'scripts/ds_train',
        'scripts/ddp_train',
        'scripts/py_train',
        'scripts/plot_lr',
        'scripts/plot_loss',
        'scripts/calc_intermediate_size'
    ],
    author='qibin',
    author_email='qibin0506@gmail.com',
    packages=find_packages()
)
