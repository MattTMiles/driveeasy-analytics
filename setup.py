from setuptools import setup
# from version import get_version
setup(
    name='pydea',
    version='0.1',
    description='DriveEasy Analytics Python package.',
    packages=['pydea'],
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
    ],
)