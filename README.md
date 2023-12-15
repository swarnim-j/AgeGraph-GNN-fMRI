# AgeGraph: Age Classification from fMRI Data using Graph Convolutional Network (GCN)

## Introduction

AgeGraph is a Python repository that implements age classification from fMRI (functional Magnetic Resonance Imaging) data using Graph Convolutional Networks (GCN). The code is structured to facilitate easy understanding, modification, and usage. This documentation provides a guide on how to use and extend the functionalities provided by AgeGraph.

## Repository Structure

```
AgeGraph/
├── AgeGraph/
│ └── datasets.py
│ └── preprocess.py
│ └── utils.py
├── results/
│ └── result.png
│
├── .gitignore
├── main.py
├── model.py
└── run_baseline.sh
```


- **results:** This directory is intended for storing uploaded files containing results.

- **.gitignore:** This file specifies patterns that should be ignored by Git. Commonly ignored files and directories are listed here.

- **main.py:** The main Python script containing the primary logic and execution flow of the age classification from fMRI data using GCN. Users can run this script to perform the age classification.

- **model.py:** This file contains the implementation of the Graph Convolutional Network (GCN) used for age classification. Users interested in the model architecture and details can refer to this file.

- **run_baseline.sh:** This shell script provides a baseline command for running the age classification on fMRI data. Users can customize this script based on their data and requirements.

## Getting Started

1. Clone the repository:

    ```
    git clone https://github.com/swarnim-j/AgeGraph.git
    cd AgeGraph
    ```

    

2. Install dependencies:
    
    ```
    # Assuming you have Python and pip installed
    pip install -r requirements.txt
    ```

3. Run the baseline:

    ```
    ./run_baseline.sh
    ```

## Usage
- **main.py:** This script is the entry point for age classification. Users can modify this file to customize the training and evaluation procedures.

- **model.py:** If users want to modify the GCN architecture or explore details of the implemented model, they should refer to this file.
