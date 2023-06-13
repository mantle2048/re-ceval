import pathlib
from distutils.core import setup
from setuptools import find_packages
import pkg_resources

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(
    name='reLLMs',
    version='0.1.0',
    packages=find_packages(),
    license='MIT License',
    long_description=open('README.md').read(),
    install_requires=install_requires,
)
