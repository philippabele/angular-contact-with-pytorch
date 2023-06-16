# Shaft Bearing Lifetime Prediction Models

Welcome to the Shaft Bearing Lifetime Prediction codebase. This repository contains code for analyzing and categorizing the lifetime of shaft bearings based on their frequency of rotation and load.

## Table of Contents
- [Shaft Bearing Lifetime Prediction Models](#shaft-bearing-lifetime-prediction-models)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Exploratory Data Analysis](#exploratory-data-analysis)
    - [Generating Dataset](#generating-dataset)
    - [Splitting Dataset](#splitting-dataset)
    - [Training Models](#training-models)
    - [Testing Models](#testing-models)
    - [Predicting Targets](#predicting-targets)
    - [Performing Grid Search](#performing-grid-search)
  - [Contributing](#contributing)

## Getting Started

To use this codebase, ensure you have Python installed on your machine. 

## Installation

1. Clone this repository to your local machine.
2. Install the required libraries by running `pip install -r requirements.txt`. It's recommended to use a virtual environment to avoid conflicts with other Python packages you may have installed.

## Usage

Execute scripts from the root of the project using Python.

The codebase contains two models: `categorize_lt` for categorizing Lifetime based on Fr and n, and `categorize_n` for categorizing rotation frequency based on Lifetime and radial load. The `categorize_lt` is the primary model and supports all features, including Grid Search. Note that `categorize_n` does not support Grid Search at the moment.

### Exploratory Data Analysis

To perform Exploratory Data Analysis to understand the dataset and identify important parameters, run:

```shell
python src/shaftbearing/shaftbearing_eda.py
```

### Generating Dataset

Set the dataset range and interval parameters in the `generate_dataset.py` script, then generate a dataset based on the equation for Lifetime by running:

```shell
python src/shaftbearing/models/categorize_lt/generate_dataset.py
```

This example uses the `categorize_lt` model. You can replace `categorize_lt` with `categorize_n` for the other model.

### Splitting Dataset

Split the generated dataset into training, testing, and validation sets by running:

```shell
python src/shaftbearing/models/categorize_lt/split_dataset.py
```

### Training Models

Train the model on the dataset by running:

```shell
python src/shaftbearing/models/categorize_lt/train_model.py
```

### Testing Models

Test the performance of the trained model with the test dataset by running:

```shell
python src/shaftbearing/models/categorize_lt/test_model.py
```

### Predicting Targets

Input features manually to get an estimated target as output by running:

```shell
python src/shaftbearing/models/categorize_lt/predict_target.py
```

### Performing Grid Search

Perform Grid Search to evaluate the impact of hyperparameters. This script combines features from the other scripts for dataset generation, splitting, model training and testing.

```shell
python src/shaftbearing/models/categorize_lt/param_grid_search.py
```

## Contributing

Contributions are warmly welcomed. For enhancements, bug fixes, or feature requests, please submit a pull request or create an issue. Adhere to the existing code style and structure. For detailed information on

 how to contribute, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file.
