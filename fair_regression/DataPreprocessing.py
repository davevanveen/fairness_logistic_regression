# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 19:35:05 2018

@author: ma56473
"""

import numpy as np
import pandas as pd


# Data import routine
# 'train_file' and 'test_file' are filenames relative to the notebook
# 'feature_list' matches .csv files for the Adult dataset. See usage for exact list.
def import_data(train_file, test_file, feature_list):
    # Read files
    train_data = pd.read_csv(train_file,
                             names=feature_list,
                             sep=r'\s*,\s*',
                             engine='python',
                             na_values="?")
    test_data = pd.read_csv(test_file,
                            names=feature_list,
                            sep=r'\s*,\s*',
                            engine='python',
                            na_values="?")

    # Return raw dataframes
    return train_data, test_data


# Data pre-processing routine
# 'raw_data' is the entire data (all features + classes)
# 'S_id' should be a valid string with the name of the protected feature
# Examples of valid 'S_id': False, 'Sex_Female', 'Race_Non-White'
# TODO: generalize list of fully dropped features
# TODO: generalize list of compressed features
# TODO: generalize list of post-encoding dropped features
def split_protected_variable(raw_data, S_id):
    # Drop completely irrelevant features
    # "fnlwgt" is a control feature related to how data was sampled
    # "Education" is the discrete version of "Education-Num"
    raw_data = raw_data.drop(["fnlwgt", "Education"], axis=1)

    # Compress country as US / non-US
    raw_data['Country'][raw_data['Country'].str.contains('United-States') is False] = 'Non-US'
    # Compress workclass as Private / non-Private
    raw_data['Workclass'][raw_data['Workclass'].str.contains('Private') is False] = 'Non-Private'
    # Compress race as White / non-White
    raw_data['Race'][raw_data['Race'].str.contains('White') is False] = 'Non-White'
    # Drop relationship
    raw_data = raw_data.drop(['Relationship'], axis=1)

    # Encode categorical features with dummy variables
    encoded_data = pd.get_dummies(raw_data,
                                  columns=["Workclass", "Race", "Marital Status",
                                           "Occupation", "Sex",
                                           "Country", "Target"],
                                  prefix=["Workclass", "Race", "Marital_status",
                                          "Occupation", "Sex",
                                          "Country", "Target"])

    # Drop the dummy feature with the most frequent value for colinearity reasons
    # This takes care of '?'/NaN by implictly replacing them with the most common entry
    encoded_data = encoded_data.drop(["Workclass_Private", "Race_White",
                                      "Marital_status_Married-civ-spouse", "Occupation_Prof-specialty",
                                      "Sex_Male", "Country_United-States", "Target_<=50K"],
                                     axis=1)

    # Centering and normalizing continuous columns entries
    numerical_features = ["Age", "Education-Num", "Capital Gain", "Capital Loss", "Hours per week"]
    for featureName in numerical_features:
        encoded_data[featureName] = (encoded_data[featureName] - np.mean(encoded_data[featureName], axis=0)) / np.std(encoded_data[featureName], axis=0)  # noqa

    # Return triple (X, S, y). Return only (X, y) is 'S_id' is void
    if not S_id:
        X = encoded_data.drop(['Target_>50K'], axis=1)
        S = False
        y = encoded_data['Target_>50K']
    else:
        X = encoded_data.drop(['Target_>50K'], axis=1)
        S = encoded_data[S_id]
        y = encoded_data['Target_>50K']

    return X, S, y


# Example usage
# Read from .csv files
train_file = '../data/adult_train.csv'
test_file = '../data/adult_test.csv'
feature_list = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
                "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                "Hours per week", "Country", "Target"]

# Read data
train_data, test_data = import_data(train_file, test_file, feature_list)


# Extract X, y, S
S_id = 'Sex_Female'  # False, 'Sex_Female' or 'Race_Non-White' are valid options
X_train, S_train, y_train = split_protected_variable(train_data, S_id)
X_test, S_test, y_test = split_protected_variable(test_data, S_id)
