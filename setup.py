#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools

with open("README.rst", "r") as fh:
    README = fh.read()

import tsclust
VERSION = tsclust.__version__

NAME = "tsclust"
DESCRIPTION = "Streaming Time-series Clustering"
URL = "https://github.com/imartinezl/tsclust"
AUTHOR = "Iñigo Martinez"
AUTHOR_EMAIL = "inigomlap@gmail.com"
CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Natural Language :: English',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
]
ENTRY_POINTS = {
    'console_scripts': [
        #'tsclust=tsclust:main',
    ],
}
PROJECT_URLS = {
    'Bug Reports': URL + '/issues',
    'Documentation': 'https://tslearn.readthedocs.io',
    'Source Code': URL,
}
REQUIRES_PYTHON = '>=3.5, <4'
EXTRAS_REQUIRE = {}
KEYWORDS = ["time series", "clustering", "streaming", "average"]
LICENSE = "MIT license"
TEST_SUITE = "tests"
REQUIREMENTS = ['numpy', 'numba', 'matplotlib', 'tabulate']
SETUP_REQUIREMENTS = []
TEST_REQUIREMENTS = ['pytest', 'pytest-cov']


setuptools.setup(
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    classifiers = CLASSIFIERS,
    description = DESCRIPTION,
    entry_points = ENTRY_POINTS,
    extras_require = EXTRAS_REQUIRE,
    include_package_data = False,
    install_requires = REQUIREMENTS,
    keywords = KEYWORDS,
    license = LICENSE,
    long_description = README,
    name = NAME,
    package_data={},
    packages = setuptools.find_packages(),
    project_urls = PROJECT_URLS,
    python_requires = REQUIRES_PYTHON,
    setup_requires = SETUP_REQUIREMENTS,
    test_suite = TEST_SUITE,
    tests_require = TEST_REQUIREMENTS,
    url = URL,
    version = VERSION,
    zip_safe = False,
)
