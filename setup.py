from setuptools import setup, find_packages
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

setup(
    name='nelpy',
    version=main_ns['__version__'],
    url='https://github.com/nelpy/nelpy/',
    download_url = 'https://github.com/nelpy/nelpy/tarball/' + main_ns['__version__'],
    license='MIT License',
    author='Etienne Ackermann',
    install_requires=['numpy>=1.11.0', # 1.11 introduced axis keyword in np.gradient
                    'scipy>=0.17.0', # 0.17.0 introduced functionality we use for interp1d
                    'matplotlib>=1.5.0', # 1.4.3 doesn't support the step kwarg in rasterc yet
                    # 'shapely>=1.6'
                    ],
    author_email='era3@rice.edu',
    description='Neuroelectrophysiology object model and data analysis in Python.',
    long_description=long_description,
    packages=find_packages(),
    keywords = "electrophysiology neuroscience data analysis",
    include_package_data=True,
    platforms='any'
)

# @misc{Ackermann2017,
#         author = {Etienne Ackermann},
#         title = {nelpy},
#         year = {2017},
#         publisher = {GitHub},
#         journal = {GitHub repository},
#         howpublished = {\url{https://github.com/nelpy/nelpy}},
#         commit = {enter commit that you used}
# }