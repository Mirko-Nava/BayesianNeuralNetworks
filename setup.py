from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

with open('HISTORY.md') as history_file:
    HISTORY = history_file.read()

setup_args = dict(
    name='bnn',
    version='0.0.1',
    description='Bayesian Neural Networks in PyTorch',
    long_description_content_type="text/markdown",
    long_description=README,
    license='MIT',
    packages=find_packages(),
    author='Mirko Nava',
    author_email='nava.mirko@gmail.com',
    keywords=['Bayesian', 'BayesianNeuralNetwork', 'BNN', 'TorchBayesian'],
    url='https://haha',
    download_url='https://pypi.org/project/haha/'
)

install_requires = [
    'elasticsearch>=6.0.0,<7.0.0',
    'jinja2'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
