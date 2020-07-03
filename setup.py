from setuptools import setup, find_packages


# FIXME: structlog 20.1.0 doesn't support deepcopy
requirements = [
    'structlog==20.1.0',
    'shapely==1.7.0',
    'matplotlib==3.2.1',
    'seaborn==0.10.1',
    'numpy==1.18.3',
    'gym==0.17.1',
    'tensorflow==2.2.0',
    'ray[rllib]==0.8.6',
    'gputil==1.4.0'
]

# TODO: update on final release
setup(
    name='deep-comp',
    version=0.6,
    description="DeepCoMP: Coordinated Multipoint Using Multi-Agent Deep Reinforcement Learning",
    url=None,
    packages=find_packages(),
    python_requires=">=3.8.*",
    install_requires=requirements,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'deepcomp=drl_mobile.main:main'
        ]
    }
)
