# XAI Exercise

This is a repository for the XAI (Explainable AI) exercise. 

## Description

The exercise consists of training a machine learning model to classify handwritten digits from the MNIST dataset, and then creating a web application to visualize and explain the model's predictions using SHAP (SHapley Additive exPlanations).

## Files

The repository contains the following files:

- `cfel.py`: Python class containing the code to compute the counterfactuals of the model.
- `LSM.py`: Python script containing the code for the LocalSurrogateModels.
- `process.py`: sample to encode various types of data.
- `vizualization.py`: script that provide the shap values.
- `main.py`: script that contains a sample of using entire flow.
- `data.csv`: generated data for testing purposes.


## Requirements

To run the code, you need to have the following libraries installed:

- pandas
- numpy
- matplotlib
- shap
- scikit-learn


## Usage

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the required packages
pip install -r Requirements.txt

# Run the code
python main.py

# Deactivate the virtual environment
deactivate
