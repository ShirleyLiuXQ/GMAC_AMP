# AMP for Gaussian Multiple Access (GMAC) with random access and coding

This repository contains code that reproduces the graphs in\
"Many-User Multiple Access with Random User Activity: Achievability Bounds and Efficient Schemes" 2024 by Xiaoqi Liu, Pablo Pascual Cobo, and Ramji Venkataramanan;\
"Coded Many-User Multiple Access via Approximate Message Passing" 2024 by Xiaoqi Liu, Kuan Hsieh, and Ramji Venkataramanan.

## Code overview

Source code is located in the `src` directory.
Scripts for running experiments are called `main_xxx`.

Each script and function contains detailed comments and docstring explaining their functionality.

## Setup

Use the following commands to set up and run the code:

### Conda environment

To set up and activate the conda environment, run:

```bash
conda env create --file environment.yml
conda activate gmac_amp
```

To add more dependencies later, add them to the `environment.yml` file
then run the following command with your environment activated:

```bash
conda env update --file environment.yml --prune
```

Optionally, install the following pre-commit hooks
to ensure code quality.

```bash
pre-commit install
```
