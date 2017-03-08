from setuptools import setup, find_packages
# from setuptools.command.test import test as TestCommand
import io

from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('nelpy/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.rst')

# class PyTest(TestCommand):
#     def finalize_options(self):
#         TestCommand.finalize_options(self)
#         self.test_args = []
#         self.test_suite = True

#     def run_tests(self):
#         import pytest
#         errcode = pytest.main(self.test_args)
#         sys.exit(errcode)

setup(
    name='nelpy',
    version=main_ns['__version__'],
    url='https://github.com/eackermann/nelpy/',
    download_url = 'https://github.com/eackermann/nelpy/tarball/' + main_ns['__version__'],
    license='MIT License',
    author='Etienne Ackermann, Emily Irvine',
    install_requires=['numpy>=1.9.0',
                    'scipy>=0.16.0',
                    'matplotlib>=1.5.0', # 1.4.3 doesn't support the step kwarg in rasterc yet
                    ],
    author_email='era3@rice.edu',
    description='Neuroelectrophysiology object model and data analysis in Python.',
    long_description=long_description,
    packages=find_packages(),
    keywords = "electrophysiology neuroscience data analysis",
    include_package_data=True,
    platforms='any'
)