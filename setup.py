from setuptools import setup, find_packages


# TODO: use structlog 20.2.0 once available, which will support deepcopy
requirements = [
    'ray[rllib]==1.0',
    'structlog>=20.1.0',
    # 'git+https://github.com/stefanbschneider/structlog.git@dev',
    'structlog-round>=1.0',
    'shapely==1.7.0',
    'matplotlib==3.2.1',
    'seaborn==0.10.1',
    'numpy==1.19.1',
    'gym>=0.17.1',
    'tensorflow==2.2.0',
    'gputil==1.4.0',
    'pandas==1.0.5',
    'tqdm==4.47.0',
    'joblib==0.16.0',
]

eval_requirements = [
    'jupyter>=1.0.0'
]

# TODO: update on final release
setup(
    name='deepcomp',
    version=0.10,
    description="DeepCoMP: Coordinated Multipoint Using Multi-Agent Deep Reinforcement Learning",
    url=None,
    packages=find_packages(),
    python_requires=">=3.8.*",
    install_requires=requirements + eval_requirements,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'deepcomp=drl_mobile.main:main'
        ]
    }
)
