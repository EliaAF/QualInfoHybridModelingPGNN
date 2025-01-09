# Qualitative Information in Hybrid Modeling – PGNN

**(C) Elia Arnese-Feffin – December 19, 2024**

Version: 1.0.0

Date: 2024/12/19

Author: Elia Arnese-Feffin (elia249@mit.edu)

## Contents

This repository contains code to reproduce the case studies presented in the Journal Paper mentioned below. The code offers a prototype implementation of Physics-Guided Neural Networks (PGNNs), based on a custom implementation of Artificial Neural Networks (ANNs).
* `QIP_env_setup.pdf` details how to install a Python environment to reproduce the case studies.
* `PGNN_prototype.py` defines functions of ANN and PGNN modeling.
* `pH_data.xlsx` is the dataset for the pH neutralization case study.
* `pH_case_study.py` contains code to reproduce the pH neutralization case study.
* `deactivation_data.xlsx` is the dataset for the catalyst deactivation case study.
* `CA_case_study.py` contains code to reproduce the catalyst deactivation case study.

## Python environment setup

The code has been developed in a virtual environment based on ``Python 3.10.14``. The environment can be set up using any ``Python`` package manager. The Anancoda distribution (version 2.5.2) was used here, and the environment was set up using ``conda 24.1.2`` The required packages are listed in the the table below.

| Package       | Version   |
| ------------- | --------- |
| Python        | 3.10.14   |
| findiff       | 0.10.0    |
| Matplotlib    | 3.8.4     |
| NumPy         | 1.26.4    |
| OpenPyXL      | 3.1.2     |
| pandas        | 2.2.2     |
| Scikit-Learn  | 1.4.2     |
| SciPy         | 1.13.0    |

All packages were obtained from the ``conda-forge`` channel. The environment can be created using the command:
```
conda create --name <env_name> -c conda-forge python=3.10.14 findiff=0.10.0 matplotlib=3.8.4 numpy=1.26.4 openpyxl=3.1.2 pandas=2.2.2 scikit-learn=1.4.2 scipy=1.13.0
```
where ``<env_name>`` must be replace with the desired name for the virtual environment.

These instructions were tested on December 19, 2024 on an M1 MacBook Pro running MacOS 15.2.0.

## Attribution

To attribute credit to the author of the software, please refer to the companion Journal Paper.

REFERENCE<!--E. Arnese-Feffin, <OTHER_AUTHORS>, R. D. Braatz (2025): TITLE. *Journal*, **00**, 000–000. DOI: [text_to_display](link)-->

## License agreement

All the files provided are covered by the GNU General Public License version 3 (GPL-3.0); a copy of the license can be found in this folder (GPL-3.0.txt), or online at https://www.gnu.org/licenses/gpl-3.0.html. This license protects the code as open-source. Key points of GPL-3.0 are as follows.
* Attribution of credit to the author is required if the software is used.
* Free use of the software is allowed, even for commercial purposes.
* Redistribution of the software for commercial purposes is prevented, as any redistribution must be released under the GPL-3.0 license, therefore as a free and open-source software.