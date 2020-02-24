import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="MarkEasy-Oleg-Zero",
    version="0.1.0",
    author="Oleg Zero",
    author_email="oleg@zerowithdot.com",
    description="Easy implementation of Hidden Markov Model",
    license="MIT",
    long_description=long_description,
    url="https://github.com/OlegZero13/markeasy/archive/0.1.0.tar.gz",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
