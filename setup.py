from bnn import __version__
from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='bnn',
    version=__version__,
    author='Mirko Nava',
    author_email='nava.mirko@gmail.com',
    description='Bayesian Deep Learning Extension Library for PyTorch',
    keywords=['pytorch', 'bayesian-deep-learning', 'bayesian-neural-networks'],
    long_description_content_type="text/markdown",
    long_description=README,
    url='https://haha',
    download_url='https://pypi.org/project/haha/',
    python_requires='>=3.6',
    install_requires=['torch'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    packages=find_packages(),
)

if __name__ == '__main__':
    setup(**setup_args)
