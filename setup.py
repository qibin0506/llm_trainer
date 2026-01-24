from setuptools import setup, find_packages

setup(
    name='project_llm_trainer',
    version='0.13.9',
    description='LLM and VLM trainer',
    scripts=[
        'scripts/smart_train',
        'scripts/ds_train',
        'scripts/ddp_train',
        'scripts/py_train',
        'scripts/vis_lr',
        'scripts/vis_log',
        'scripts/calc_intermediate_size'
    ],
    author='qibin',
    author_email='qibin0506@gmail.com',
    packages=find_packages()
)
