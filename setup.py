#!/usr/bin/env python3
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vision_utils",
    author="glhr",
    author_email="grenblot@gmail.com",
    description="Utilities for image processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/glhr/img-utils",
    packages=setuptools.find_packages(),
    python_requires='>=3.6'
)
