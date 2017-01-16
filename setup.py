from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import io

import nelpy

# here = os.path.abspath(os.path.dirname(__file__))

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

# long_description = read('README.md', 'CHANGES.txt')
# long_description = read('README.md')
long_description = 'Neuroelectrophysiology object model and data analysis in Python.'

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

setup(
    name='nelpy',
    version=nelpy.__version__,
    url='https://github.com/eackermann/nelpy/',
    download_url = 'https://github.com/eackermann/nelpy/tarball/' + nelpy.__version__,
    license='MIT License',
    author='Etienne Ackermann',
    tests_require=['pytest'],
    install_requires=['numpy>=1.9.0',
                    'scipy>=0.16.0',
                    'matplotlib>=1.4.0',
                    ],
    cmdclass={'test': PyTest},
    author_email='era3@rice.edu',
    description='Neuroelectrophysiology object model and data analysis in Python.',
    long_description=long_description,
    packages=['nelpy'],
    keywords = ['electrophysiology', 'cow', 'moo', 'neural data analysis'],
    include_package_data=True,
    platforms='any',
    test_suite='nelpy.tests.test_nelpy',
    extras_require={
        'testing': ['pytest'],
        'docs': ['sphinx', 'numpydoc', 'mock']
    }
)