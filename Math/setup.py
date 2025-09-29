from setuptools import setup, find_packages

setup(
    name='ROVER',
    version='0.0.0',
    description='ROVER is a minimalist and highly effective RL method for LLM reasoning, achieving superior optimality and diversity by evaluating uniform-policy Q-values.',
    author='Haoran He',
    packages=find_packages(include=['deepscaler',]),
    install_requires=[
        'google-cloud-aiplatform',
        'latex2sympy2',
        'pylatexenc',
        'sentence_transformers',
        'tabulate',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache 2.0 License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)