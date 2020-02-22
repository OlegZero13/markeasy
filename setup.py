import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="MarkEasy-Oleg-Żero",
    version="0.1.0",
    author="Oleg Żero",
    author_email="oleg@zerowithdot.com",
    descrtiption="Easy implementation of Hidden Markov Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=None,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requries='>=3.6',
)
