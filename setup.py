import os
from setuptools import setup, find_packages


# read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


requirements = [
    # for scenarios with in- and decreasing num. UEs (--ue-arrival) use Ray 1.0:
    # https://github.com/ray-project/ray/issues/15297
    'ray[rllib]==1.4.0',
    'structlog>=20.2.0',
    'structlog-round>=1.0',
    'shapely==1.7.0',
    'matplotlib==3.2.1',
    'seaborn==0.10.1',
    'numpy<1.20',
    'gym[atari]>=0.17.1',
    'tensorflow>=2.2.0',
    'gputil==1.4.0',
    'pandas>=1.0.5',
    'tqdm==4.47.0',
    'joblib==0.16.0',
    # extra dependencies for Ray RLlib
    # installing directly via ray[rllib] doesn't work with setup.py: https://github.com/ray-project/ray/issues/11274
    'scipy==1.4.1',
    # 'lz4',
    'svgpath2mpl>=0.2.1',
]

eval_requirements = [
    'jupyter>=1.0.0'
]

setup(
    name='deepcomp',
    version='1.4.0',
    author='Stefan Schneider',
    description="DeepCoMP: Self-Learning Dynamic Multi-Cell Selection for Coordinated Multipoint (CoMP)",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/CN-UPB/DeepCoMP',
    packages=find_packages(),
    python_requires=">=3.8.*",
    install_requires=requirements + eval_requirements,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'deepcomp=deepcomp.main:main'
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
