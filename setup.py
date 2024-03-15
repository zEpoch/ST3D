#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/28 16:09
# @Author  : zhoutao
# @File    : setup.py.py
# @Software: VScode
# @Email   : zhotoa@foxmail.com

import setuptools
from wheel.bdist_wheel import bdist_wheel

__version__ = "1.0.0"


class BDistWheel(bdist_wheel):
    def get_tag(self):
        return (self.python_tag, "none", "any")


cmdclass = {
    "bdist_wheel": BDistWheel,
}

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


requirements = open("requirements.txt").readline()

setuptools.setup(
    name = "ST3D",
    version = "1.0.0",
    description = "A visualization tool for 3D spatial transcriptomics data.",
    author = "zhoutao",
    author_email = "zhotoa@foxmail.com",
    url = "https://github.com/zEpoch/ST3D.git",
    python_requires=">=3.8",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    cmdclass=cmdclass,
)