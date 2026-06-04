from setuptools import setup, find_packages

setup(
    name="GGNN",
    version="0.1",
    packages=find_packages(include=['GGNN', 'GGNN.*']),
    #install_requires=[
    #    "requests",
    #],
    #entry_points={
    #    "console_scripts": [
    #        "my-command=GGNN:main",
    #    ],
    #},
    python_requires='>3.10',
)
