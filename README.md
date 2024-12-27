<h1 align="center">Scratches</h1>

<p align="center">
Scratches is a project that is inspired by the «Deep Learning from Scratch»
book by Seth Weidman, which provides a comprehensive guide to creating deep
learning models from scratch using Python, NumPy and SciPy. The project aims to provide
pure Python and NumPy implementations of classic machine learning algorithms
such as k-nearest neighbors, linear and multiple regressions, and elementary and
convolutional neural networks.
</p>

## Requirements

The only system requirement for this application is that you use Conda or
Miniconda to manage your Python packages.

## Installation and usage

Use the Git command-line interface (CLI) to clone this repository into your
working directory using the following command:
```shell
git clone https://github.com/mkashirin/scratches
```
To create a virtual environment, please follow the lines below:
```shell
uv venv .venv
source .venv/bin/activate
uv pip install .
uv sync examples
```
Although NumPy and SciPy are crucial dependencies for the functioning of the algorithms,
Jupiter, Matplotlib, and Pandas are also present in the environment in order to
provide a seamless experience.

After that You can just run the Jupyter sever to access the notebooks from the
**examples** directory by executing the following command:
```shell
jupyter lab
```
Or open the project in VSCode:
```shell
code .
```
And that's it. You are all set!

## Suggestions

The only specific suggestion is to not use it outside the educational context.

If you are still unsure, do not worry. The documentation in the source code can
be considered sufficient. The code has been written in a clear and concise
manner, focusing on readability rather than efficiency.

So, feel free to experiment with machine learning models! Combine various
structures to create your own neural networks. Explore the code to gain a deeper
understanding of fundamental ML and AI principles.

## Licencing

This project is distributed under the MIT open source licence.
