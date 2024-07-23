# Overview

This repository contains all the code used for generating data of the paper titled "Characterizing the Accuracy - Efficiency Trade-off of Low-rank Decomposition in Language Models" at IISWC 2024.

# Dependencies

## Software
* Miniconda
* Libriaries in `environment.yml` file
* `Llama-2-7b-chat-hf` model in the `decomposed-lm-evaluation-harness/Llama-2-7B` directory

## Hardware
* NVIDIA GPU with CUDA Support
* At least 16 GB of RAM and disk space

# How to run
1. Install all the software dependencies using conda
`conda create -f environment.yml`
2. Clone this repository
3. From the terminal, run:   
`cd scripts`   
`./validate_data_for_fig_9_and_10.sh`   
Output Directory:    
`decomposed-lm-evailuation-harness/output_dir`

# License

MIT License; please refer to [`LICENSE`](https://github.com/UCI-ISA-Lab/Low-Rank-Decomposition-Characterization-IISWC_2024-AE/blob/main/LICENSE).
[`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) is also under MIT license.
