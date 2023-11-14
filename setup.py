from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requires = [x for x in f.readlines()]

setup(
  name = 'sparse-interpretable-audio',
  version = '1.6.3',
  license='MIT',
  description = 'Models for extracting sparse, interpretable representations of audio',
  long_description_content_type = 'text/markdown',
  author = 'John Vinyard',
  author_email = 'john.vinyard@gmail.com',
  url = 'https://github.com/JohnVinyard/sparse-interpretable-audio',
  keywords = [
    'machine learning',
    'sparse',
    'representation learning'
  ],
  install_requires=requires,
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)