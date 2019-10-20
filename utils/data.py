# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 14:58:05 2019

@author: liam.bui

The file contains functions to handle dataset and evaluation metrics on the dataset
"""

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix
from utils import config

def read_data(data_path, col_smiles='smiles', col_target='HIV_active'):
    """Split original data into train data and test data.
    :param data_path: str, path to the a CSV data file
    :param col_smiles: str, name of smiles column
    :param col_target: str, name of target column
    :param test_ratio: float, proportion of the original data for testset, must be from 0 to 1
    :param seed: int, randomization seed for reproducibility
    :return (X, y)
    """
    
    # read data
    df = pd.read_csv(data_path, sep=',')
    df_no_na = df[[col_smiles, col_target]].dropna()

    X = df_no_na[col_smiles]
    y = df_no_na[col_target]
    
    return X, y

                
def get_prediction_score(y_label, y_predict):
    """Evaluate predictions using different evaluation metrics.
    :param y_label: list, contains true label
    :param y_predict: list, contains predicted label
    :return scores: dict, evaluation metrics on the prediction
    """
    scores = {}
    scores[config.METRIC_ACCURACY] = accuracy_score(y_label, y_predict)
    scores[config.METRIC_F1_SCORE] = f1_score(y_label, y_predict, labels=None, average='macro', sample_weight=None)
    scores[config.METRIC_COHEN_KAPPA] = cohen_kappa_score(y_label, y_predict)
    scores[config.METRIC_CONFUSION_MATRIX] = confusion_matrix(y_label, y_predict)
    
    return scores