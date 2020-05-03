import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sequence_models", # Replace with your own username
    version="0.0.1",
    author="Kevin Yang",
    author_email="yang.kevin@microsoft.com",
    description="Machine learning models for sequences.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://msrne-ml.visualstudio.com/sequence_models",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)