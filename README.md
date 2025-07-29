# AMP for Gaussian Multiple Access (GMAC) with Random Access and Coding

"This repository implements a CDMA-type coding scheme based on spatially coupled design matrices and approximate message passing (AMP) decoding, as proposed in the following papers:

[1] X. Liu, P. Pascual Cobo, and R. Venkataramanan, "*Many-User Multiple Access with Random User Activity: Achievability Bounds and Efficient Schemes*," in submission, 2025.\
[2] X. Liu, P. Pascual Cobo, and R. Venkataramanan, "*Many-User Multiple Access with Random User Activity*," in Proc. IEEE Int. Symp. Inf. Theory, 2024.\
[3] X. Liu, K. Hsieh, and R. Venkataramanan, "*Coded Many-User Multiple Access via Approximate Message Passing*," Information Theory, Probability and Statistical Learning: A Festschrift in Honor of Andrew Barron, 2025.\
[4] X. Liu, K. Hsieh, and R. Venkataramanan, "*Coded Many-User Multiple Access via Approximate Message Passing*," in Proc. IEEE Int. Symp. Inf. Theory, 2024.

## Code overview
The repository is organized as follows:

- All source code is in the `src` directory containing core implementations and utilities
- LDPC code implementation in `ldpc_jossy/` is adapted from code written by Jossy Sayir
- Experiment scripts are prefixed with `main_` and can be found in the root directory

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
