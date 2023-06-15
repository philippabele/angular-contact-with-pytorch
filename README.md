# Shaft Bearing Lifetime Prediction Models

Welcome to the Shaft Bearing Lifetime Prediction codebase. This repository contains code for analyzing and predicting the lifetime of shaft bearings based on their frequency of rotation and load.

## Table of Contents
- [Shaft Bearing Lifetime Prediction Models](#shaft-bearing-lifetime-prediction-models)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
  - [Usage](#usage)
    - [Exploratory Data Analysis](#exploratory-data-analysis)
    - [Generating Dataset](#generating-dataset)
    - [Splitting Dataset](#splitting-dataset)
    - [Training Models](#training-models)
    - [Testing Models](#testing-models)
    - [Predicting Targets](#predicting-targets)
  - [Contributing](#contributing)

## Getting Started

To use this codebase, you will need Python installed on your machine. This tutorial assumes you are familiar with basic Python programming.

1. Clone this repository to your local machine.
2. Ensure you have all the required libraries installed. (You might want to use a virtual environment)

## Usage

Execute scripts from the root of the project using Python.

The sample scripts in this readme use the categorize_lt model for categorizing Lifetime based on Fr and n. If you want to work with the model for categorizing n based on Fr and Lifetime, replace categorize_lt with categorize_n.

### Exploratory Data Analysis

To perform Exploratory Data Analysis to understand the dataset and find important parameters, run:

```shell
python src/shaftbearing/shaftbearing_eda.py
```

### Generating Dataset

Set the dataset range and interval parameters in the `split_dataset.py` script, then generate a dataset based on the equation for Lifetime by running:

```shell
python src/shaftbearing/models/categorize_lt/generate_dataset.py
```

This example uses the `categorize_lt` model. You can replace `categorize_lt` with `categorize_n` if you want to work with the other model.

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

Test how well the trained model performs with the test dataset by running:

```shell
python src/shaftbearing/models/categorize_lt/train_model.py
```

### Predicting Targets

Input features manually and get an estimated target as output by running:

```shell
python src/shaftbearing/models/categorize_lt/predict_target.py
```

## Contributing

Contributions are warmly welcomed. For enhancements, bug fixes, or feature requests, please submit a pull request or create an issue. Adhere to the existing code style and structure. For detailed information on how to contribute, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file.
