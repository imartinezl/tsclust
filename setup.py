#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = []

setup_requirements = [ ]

test_requirements = [ ]

setuptools.setup(
    author="Iñigo Martinez",
    author_email="inigomlap@gmail.com",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Streaming Time-series Clustering",
    entry_points={
        'console_scripts': [
            'tsclust=tsclust:main',
        ],
    },
    extras_require={},
    include_package_data=True,
    install_requires=requirements,
    keywords='python_boilerplate',
    license="MIT license",
    long_description=long_description,
    long_description_content_type="text/markdown",
    name="tsclust", 
    package_data={},
    packages=setuptools.find_packages(include=['tsclust', 'tsclust.*']),
    project_urls={
        #'Bug Reports': 'https://github.com/pypa/sampleproject/issues',
        #'Funding': 'https://donate.pypi.org',
        #'Say Thanks!': 'http://saythanks.io/to/example',
        #'Source': 'https://github.com/pypa/sampleproject/',
    },
    python_requires='>=3.5, <4',
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url="https://github.com/imartinezl/tsclust",
    version="0.0.1",
    zip_safe=False,

)
