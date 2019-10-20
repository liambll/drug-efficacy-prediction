# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:29:37 2019

@author: liam.bui

This file contains functions to train and evaluate classical machine learning models

"""

import os
import sys
sys.path.insert(0, os.getcwd()) # add current working directory to pythonpath

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from utils import config
from utils.data import read_data, get_prediction_score
from models import feature_extraction
import pickle
import warnings
import argparse
    

def train_model(model, X_train, y_train, parameters, n_splits=3):
    """Train model with Grid-search cross validation to find the best hyperparameter
    :param model: Scikit-learn estimator
    :param X_train: trainset features
    :param y_train: trainset label
    :param parameters: dict, key is hyper parameter name and value is a list of hyper parameter values
    :return best_estimator: Scikit-learn estimator with the best hyper parameter
    :return best_score: best accuracy score
    :return best_param: dict, best hyper parameter
    """
    
    splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0).split(X=X_train, y=y_train)
    
    clf = GridSearchCV(model, parameters, cv=splits, scoring=make_scorer(accuracy_score))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # disable the warning on default optimizer
        clf.fit(X_train, y_train)

    return clf.best_estimator_, clf.best_score_, clf.best_params_


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model on testset
    :param model: Scikit-learn estimator
    :param X_train: trainset features
    :param y_train: trainset label
    :param X_test: testset features
    :param y_test: testset label
    :param parameters: dict, key is hyper parameter name and value is a list of hyper parameter values
    :return model: Scikit-learn estimator, fitted on the whole trainset
    :return y_predict: prediction on test set
    :return scores: dict, evaluation metrics on test set
    """
    
    # Refit the model on the whole train set
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # disable the warning on default optimizer
        model.fit(X_train, y_train)
        
    # Evaluate on test set
    y_predict = model.predict(X_test)
    scores = None
    if y_test is not None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # disable the warning on f1-score with not all labels
            scores = get_prediction_score(y_test, y_predict)

    return model, y_predict, scores


def build_base_models(X_train, y_train):
    """Train and evaluate different base models. "Base" means the model is not a stacking model. 
    :param X_train: trainset features
    :param y_train: trainset label
    :return fitted_models: list, contains fitted Scikit-learn estimators
    :return model_names: list, names of fitted Scikit-learn estimators
    :return model_scores: list, contains scores on test set for fitted Scikit-learn estimators.
                    Each score is a dict of evaluation metrics
    """
    
    ########################
    # DEFINE BASE MODELS ###
    ########################
    models = []
    model_params = []
    model_names = []
    
    # Random forest model
    for n_estimators in [500, 1000, 2000]:
        for max_depth in [3, 5, 7]:
            models.append(RandomForestClassifier(max_features='sqrt', class_weight='balanced', random_state=0))
            model_params.append({'n_estimators':[n_estimators], 'max_depth':[max_depth]})
            model_names.append('Random Forest')   
    
    # Boosted Tree
    for n_estimators in [500, 1000, 2000]:
        for max_depth in [3, 5, 7]:
            for learning_rate in [0.01, 0.1]:
                models.append(GradientBoostingClassifier(subsample=0.7, max_features='sqrt', random_state=0))
                model_params.append({'n_estimators':[n_estimators], 'max_depth':[max_depth], 'learning_rate':[learning_rate]})
                model_names.append('Gradient Boosting Machine')
    
    # SVM
#    for kernel in ['linear', 'rbf']:
#        for C in [1.0, 10.0, 100.0, 1000.0]:
#            models.append(SVC(probability=True, gamma='auto', tol=0.001, cache_size=200, class_weight='balanced',
#                              random_state=0,
#                              decision_function_shape='ovr'))
#            model_params.append({'kernel':[kernel], 'C':[C]})
#            model_names.append('Support Vector Machine')
    
    # Logistic regression model
    for penalty in ['l1', 'l2']:
        for C in [1.0, 10.0, 100.0, 1000.0]:
            models.append(linear_model.LogisticRegression(max_iter=500, solver='liblinear', multi_class='ovr',
                                                          class_weight='balanced', random_state=0))
            model_params.append({'penalty':[penalty], 'C':[C]})
            model_names.append('Logistic Regression')
        
    # KNN
#    for n_neighbors in [5, 10, 15]:
#        for weights in ['uniform', 'distance']:
#            models.append(KNeighborsClassifier())
#            model_params.append({'n_neighbors':[n_neighbors], 'weights':[weights]})
#            model_names.append('K Nearest Neighbour')
            
    ##################################
    # TRAIN AND EVALUATE BASE MODELS #
    ##################################
    fitted_models = []
    model_scores = []
    for i in range(len(models)):
        print('Evaluating model {} of {}: {}'.format((i+1), len(models), model_names[i]))
        model = models[i]
        fitted_cv, _, _ = train_model(model=model, X_train=X_train, y_train=y_train, parameters=model_params[i])
        fitted_whole_set, _, score = evaluate_model(model=fitted_cv, X_train=X_train, y_train=y_train,
                                                    X_test=X_test, y_test=y_test)
        fitted_models.append(fitted_whole_set)
        model_scores.append(score)
        print(model_names[i], score)
        
    return fitted_models, model_names, model_scores


def build_stack_models(base_models, X_train, y_train):
    """Train and evaluate different stack models
    :param base_models: list, contains fitted base models, which are Scikit-learn estimators
    :param X_train: trainset features
    :param y_train: trainset label
    :return stack_fitted_models: list, contains fitted Scikit-learn estimators
    :return stack_model_names: list, names of fitted Scikit-learn estimators
    :return stack_model_scores: list, contains scores on test set for fitted Scikit-learn estimators.
                    Each score is a dict of evaluation metrics
    """
    
    ###############################
    ### PREPARE DATA FOR STACKING #
    ###############################
    print('Preparing data for model stacking')
    from sklearn.preprocessing import OneHotEncoder
    label_encoder = OneHotEncoder(categories='auto', sparse=False)
    label_encoder.fit(np.unique(y_train).reshape(-1, 1))
    nb_classes = len(label_encoder.categories_[0])
    
    # Get base models' prediction for test set: simply use the trained models to predict on test set
    X_test_stack = np.zeros([X_test.shape[0], len(base_models)*nb_classes])
    for i in range(len(base_models)):
        model = base_models[i]
        X_test_stack[:, i*nb_classes:(i+1)*nb_classes] = label_encoder.transform(model.predict(X_test).reshape(-1, 1))
            
    # Get base models' prediction for train set: use 3-fold split, train model on 2 parts and predict on 3rd part
    splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0).split(X=X_train, y=y_train)
    X_train_stack = np.zeros([X_train.shape[0], len(base_models)*nb_classes])
    for train_index, val_index in splits:
        # train and validation set
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, _ = y_train[train_index], y_train[val_index]

        # Fit model
        for i in range(len(base_models)):
            model = base_models[i]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # disable the warning on default optimizer
                model.fit(X_tr, y_tr)
            X_train_stack[val_index, i*nb_classes:(i+1)*nb_classes] = \
                label_encoder.transform(model.predict(X_val).reshape(-1, 1))

    # Add base models' predictions into the feature space
    X_train_stack = np.concatenate([X_train, X_train_stack], axis=-1)
    X_test_stack = np.concatenate([X_test, X_test_stack], axis=-1)
          
    ########################
    # DEFINE STACK MODELS ##
    ########################
    stack_models = []
    stack_model_names = []
    stack_model_params = []
    
    stack_models.append(linear_model.LogisticRegression(max_iter=500, solver='liblinear', multi_class='ovr',
                                                        class_weight='balanced', random_state=0))
    stack_model_names.append('Stack Logistic Regression')
    stack_model_params.append({'penalty':['l1', 'l2'], 'C':[1.0, 10.0, 100.0, 1000.0]})
    
    stack_models.append(RandomForestClassifier(class_weight='balanced', random_state=0))
    stack_model_names.append('Stack Random Forest')
    stack_model_params.append({'n_estimators':[500, 1000, 2000], 'max_depth':[3, 5, 7]})
    
    stack_models.append(GradientBoostingClassifier(subsample=0.7, max_features='sqrt', learning_rate=0.01,
                                                   random_state=0))
    stack_model_names.append('Stack Gradient Boosting Machine')
    stack_model_params.append({'n_estimators':[500, 1000, 2000], 'max_depth':[3, 5, 7]})
    
    stack_models.append(SVC(probability=True, gamma='auto', tol=0.001, cache_size=200, random_state=0,
                             decision_function_shape='ovr', class_weight='balanced'))
    stack_model_names.append('Stack Support Vector Machine')
    stack_model_params.append({'kernel':['linear', 'rbf'], 'C':[1.0, 10.0, 100.0, 1000.0]})
    
    stack_models.append(KNeighborsClassifier())
    stack_model_names.append('Stack K Nearest Neighbour')
    stack_model_params.append({'n_neighbors':[5, 10, 15], 'weights':['uniform', 'distance']})          

    #########################
    # EVALUATE STACK MODELS #
    #########################
    stack_fitted_models = []
    stack_model_scores = []
    for i in range(len(stack_models)):
        print('Evaluating model {} of {}: {}'.format((i+1), len(stack_models), stack_model_names[i]))
        model = stack_models[i]
        fitted_cv, _, _ = train_model(model=model, X_train=X_train_stack, y_train=y_train,
                                      parameters=stack_model_params[i])
        fitted_whole_set, _, score = evaluate_model(model=fitted_cv, X_train=X_train_stack, y_train=y_train,
                                                    X_test=X_test_stack, y_test=y_test)
        stack_fitted_models.append(fitted_whole_set)
        stack_model_scores.append(score)
        print(stack_model_names[i], score)
        
    return stack_fitted_models, stack_model_names, stack_model_scores
        

if __name__ == '__main__':
    data_path = os.path.join(config.WORK_DIRECTORY, config.DATA_FILE)
    n_splits = 3
    save_path = config.WORK_DIRECTORY

    # parse parameters
    parser = argparse.ArgumentParser(description='Build base and stacking models')
    parser.add_argument('--data_path', help='A path to csv data file')
    parser.add_argument('--n_splits', type=int, help='Number of fold for Cross Valiation')
    parser.add_argument('--save_path', help='A path to save fitted models')
     
    args = parser.parse_args()
    if args.data_path:
        data_path = args.data_path
    if args.n_splits:
        n_splits = args.n_splits
    if args.save_path:
        save_path = args.save_path

    # Read data
    X, y = read_data(data_path, col_smiles='smiles', col_target='HIV_active')
    
    # Get train and test set
    smiles_train, smiles_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_RATIO, shuffle=True, stratify=y,
                                                      random_state=config.SEED)
    
    # Extract features
    feature_train = feature_extraction.extract_features(smiles_train, method=['morgan', 'mqn'],
                                                        from_smiles=True)
    feature_test = feature_extraction.extract_features(smiles_test, method=['morgan', 'mqn'],
                                                        from_smiles=True)
    
    # Filter out irrelevant features
    feature_train_filtered, pval_significant = feature_extraction.filter_feature_Chi2(feature_train, y_train)
    feature_test_filtered = feature_test[:, pval_significant]

    # Normalize features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(feature_train_filtered)
    X_test = scaler.transform(feature_test_filtered)

    # Build base models
    base_models, base_model_names, base_model_scores = build_base_models(X_train, y_train)
    if save_path is not None:
        # Save base models    
        os.makedirs(os.path.join(save_path, 'base_models'), exist_ok=True)
        for i in range(len(base_models)):
            with open(os.path.join(save_path, 'base_models', 'base_model_' + str(i+1) + '.pkl'), 'wb') as f:
                pickle.dump(base_models[i], f)
            
    # Build level 1 stack models
    stack_fitted_models, stack_model_names, stack_model_scores = build_stack_models(base_models.copy(),
                                                                                    X_train, y_train)
    if save_path is not None:
        # Save base models    
        os.makedirs(os.path.join(save_path, 'stacking_models'), exist_ok=True)
        for i in range(len(stack_fitted_models)):
            with open(os.path.join(save_path, 'stacking_models', stack_model_names[i] + '.pkl'), 'wb') as f:
                pickle.dump(stack_fitted_models[i], f)
                
    # Summarize model performance
    model_df = pd.DataFrame({'model': base_model_names + stack_model_names,
                             config.METRIC_ACCURACY: [score[config.METRIC_ACCURACY] for score in base_model_scores]\
                                 + [score[config.METRIC_ACCURACY] for score in stack_model_scores],
                            config.METRIC_F1_SCORE: [score[config.METRIC_F1_SCORE] for score in base_model_scores]\
                                 + [score[config.METRIC_F1_SCORE] for score in stack_model_scores],
                            config.METRIC_COHEN_KAPPA: [score[config.METRIC_COHEN_KAPPA] for score in\
                                                        base_model_scores]\
                                 + [score[config.METRIC_COHEN_KAPPA] for score in stack_model_scores],
                            config.METRIC_CONFUSION_MATRIX: [score[config.METRIC_CONFUSION_MATRIX] for score in\
                                                             base_model_scores]\
                                 + [score[config.METRIC_CONFUSION_MATRIX] for score in stack_model_scores]                            
                             })
    model_df = model_df[['model', config.METRIC_ACCURACY, config.METRIC_F1_SCORE, config.METRIC_COHEN_KAPPA,
                         config.METRIC_CONFUSION_MATRIX]]
    model_df.to_csv(os.path.join(config.WORK_DIRECTORY, 'summary_stacking_model.csv'), index=False)
    model_df.sort_values(by=[config.METRIC_ACCURACY, config.METRIC_F1_SCORE, config.METRIC_COHEN_KAPPA],
                         ascending=False, inplace=True)
    print('Best model:\n' + str(model_df.iloc[0]))
