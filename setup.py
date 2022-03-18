import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sequence-models",
    version="1.0.0",
    author="Kevin Yang",
    author_email="yang.kevin@microsoft.com",
    description="Machine learning for sequences.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/protein-sequence-models",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    python_requires='>=3.6',
)