# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:10:54 2019

@author: liam.bui

The file contains configuration and shared variables

"""

##########################
## FOLDER STURCTURE ######
##########################
WORK_DIRECTORY = 'C:/Users/liam.bui/Desktop/drug-efficacy/'
DATA_FILE = 'HIV.csv'

##########################
## EVALUATION METRICS ####
##########################
METRIC_ACCURACY = 'accuracy'
METRIC_F1_SCORE = 'f1-score'
METRIC_COHEN_KAPPA = 'Cohen kappa'
METRIC_CONFUSION_MATRIX = 'Confusion Matrix'

###############
## MODEL ######
###############
CLASSES = ['benign', 'malignant']
TEST_RATIO = 0.2
SEED = 0