from setuptools import setup, find_packages

setup(
    name='project_llm_trainer',
    version='0.15.0',
    description='LLM and VLM trainer',
    scripts=[
        'scripts/smart_train',
        'scripts/ds_train',
        'scripts/py_train',
        'scripts/vis_lr',
        'scripts/vis_log',
        'scripts/calc_intermediate_size',
    ],
    author='qibin',
    author_email='qibin0506@gmail.com',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'packaging',
        'deepspeed',
        'numpy',
        'transformers',
    ],
)
