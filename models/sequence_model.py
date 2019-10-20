# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:29:37 2019

@author: liam.bui

This file contains functions to train and evaluate convolutional neural networks
"""

import os
import sys
sys.path.insert(0, os.getcwd()) # add current working directory to pythonpath

import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"]="0" # Use only the 1st GPU
tf_config = tf.ConfigProto()
set_session(tf.Session(config=tf_config))
from keras import callbacks
from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten
from keras.layers import Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D, Conv1D, MaxPooling1D, GRU, Bidirectional
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import warnings
import argparse
import gc
from utils.data import read_data, get_prediction_score
from utils import config


def generate_tokens(smiles, len_percentile=100):
    """
    Generate character tokens from smiles
    :param smiles: Pandas series, containing smiles
    :param len_percentile: percentile of smiles length to set as max length
    :return tokens
    :return num_words
    :return max_phrase_len
    """ 
    
    # Get max length of smiles
    smiles_len = smiles.apply(lambda p: len(p))
    max_phrase_len = int(np.percentile(smiles_len, len_percentile))
    print('True max length is ' + str(np.max(smiles_len)) + ', ' + str(max_phrase_len) + ' is set the length cutoff.')
        
    # Get unique words
    unique_words = np.unique(np.concatenate(smiles.apply(lambda p: np.array(list(p))).values, axis=0))
    num_words = len(unique_words)
    print('Vocab size is ' + str(num_words))
    
    tokenizer = Tokenizer(
        num_words = num_words,
        filters = '$',
        char_level = True,
        oov_token = '_'
    )
    
    tokenizer.fit_on_texts(smiles)
    sequences = tokenizer.texts_to_sequences(smiles)
    tokens = pad_sequences(sequences, maxlen = max_phrase_len, padding='post', truncating='post')
    
    return tokens, num_words, max_phrase_len
    
    
def create_model(model_type, num_words, input_length, output_dim=1, dropout_rate=0.0):
    """Build different sequence model
    :param model_type: str, can be 'cnn-gru', 'cnn', 'gru', 'lstm'
    :param num_words: int
    :param input_length: int
    :param output_dim: int
    :return model: Keras model
    """ 
    
    model = Sequential()
    if model_type == 'lstm': # LSTM - LSTM
        model.add(Embedding(num_words+1, 50, input_length=input_length))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim, activation='sigmoid'))
    elif model_type == 'gru': # GRU - GRU
        model.add(Embedding(num_words+1, 50, input_length=input_length))
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(Bidirectional(GRU(128)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim, activation='sigmoid'))
    elif model_type == 'cnn-gru': # 1D CNN - GRU
        model.add(Embedding(num_words+1, 50, input_length=input_length))
        model.add(Conv1D(192,3,activation='relu'))
        model.add(Bidirectional(GRU(224, return_sequences=True)))
        model.add(Bidirectional(GRU(384)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim, activation='sigmoid'))
    elif model_type == 'cnn': # 1D CNN
        model.add(Embedding(num_words+1, 50, input_length=input_length))
        model.add(Conv1D(192, 10, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(192, 3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim, activation='sigmoid'))
    else:
        raise ValueError(model_type + ' is not supported.')
 
    model.summary()    
    return model


def build_sequence_model(trainset, testset, model_type, num_words, input_length, output_dim=1, dropout_rate=0.0,
                     batch_size=32, nb_epochs=100, lr=0.001,
                     save_path=None):
    """Train and evaluate CNN model
    :param trainset: (X_train, y_train)
    :param testset: (X_test, y_test)
    :param model_type: str, can be 'cnn-gru', 'cnn', 'gru', 'lstm'
    :param num_words: int
    :param input_length: int
    :param output_dim: int
    :param batch_size: int, batch size for model training
    :param nb_epochs: int, number of training epoches
    :param lr: float, learning rate
    :param save_path: path to save model
    :return model: fitted Keras model
    :return scores: dict, scores on test set for the fitted Keras model
    """
    
    # Create model
    model = create_model(model_type=model_type, num_words, input_length, output_dim=output_dim,
                         dropout_rate=dropout_rate)
    
    # Callback list
    callback_list = []
    # monitor val_loss and terminate training if no improvement
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, \
                patience=20, verbose=2, mode='auto', restore_best_weights=True)
    callback_list.append(early_stop)
    
    if save_path is not None:
        # save best model based on val_acc during training
        checkpoint = callbacks.ModelCheckpoint(os.path.join(save_path, backbone + '.h5'), monitor='val_acc', \
                    verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        callback_list.append(checkpoint)
        
    # Get train and test set
    (X_train, y_train) = trainset
    (X_test, y_test) = testset
    
    # Compute class weights
    weight_list = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    weight_dict = {}
    for i in range(len(np.unique(y_train))):
        weight_dict[np.unique(y_train)[i]] = weight_list[i]
    
    # Train only classification head
    optimizer = Adam(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=nb_epochs, \
                        class_weight=weight_dict, callbacks=callback_list, verbose=2)
    
    # Evaluate model    
    prediction = model.predict(X_test)
    y_val_predict = (prediction > 0.5).astype('uint8')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # disable the warning on f1-score with not all labels
        scores = get_prediction_score(y_val, y_val_predict)
        
    return model, scores

    
if __name__ == '__main__':
    data_path = os.path.join(config.WORK_DIRECTORY, config.DATA_FILE)
    model_list = ['cnn', 'cnn-gru', 'gru', 'lstm']
    batch_size = 16
    nb_epochs = 100
    lr = 0.001
    save_path = config.WORK_DIRECTORY

    # parse parameters
    parser = argparse.ArgumentParser(description='Build CNN models')
    parser.add_argument('--data_path', help='A path to csv data file')
    parser.add_argument('--batch_size', type=int, help='Batch size for model training')
    parser.add_argument('--nb_epochs', type=int, help='Number of training epoches')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--save_path', help='A path to save fitted models')
    
    args = parser.parse_args()
    if args.data_path:
        data_path = args.data_path
    if args.batch_size:
        batch_size = args.batch_size
    if args.nb_epochs:
        nb_epochs = args.nb_epochs
    if args.lr:
        lr = args.lr
    if args.save_path:
        save_path = args.save_path
      
    # Make save_path
    if save_path is not None:
        os.makedirs(os.path.join(save_path, 'sequence_models'), exist_ok=True)
        
    # Read data
    smiles, y = read_data(data_path, col_smiles='smiles', col_target='HIV_active')
    tokens, num_words, max_phrase_len = generate_tokens(smiles, len_percentile=100)
    
    # Get train and test set
    X_train, X_test, y_train, y_test = train_test_split(tokens, y, test_size=config.TEST_RATIO, shuffle=True, stratify=y,
                                                      random_state=config.SEED)
    
    # Build en evaluate graph models
    model_scores = []
    for model_type in model_list:
        model, scores = build_sequence_model((X_train, y_train), (X_test, y_test), model_type, num_words, input_length,
                                             output_dim=1, dropout_rate=0.0,
                                             batch_size=batch_size, nb_epochs=nb_epochs, lr=lr,
                             save_path=os.path.join(save_path, 'sequence_models', model_type + '.h5'))
        model_scores.append(scores)
            
        # force release memory
        K.clear_session()
        del model
        gc.collect()
        
    # Summarize model performance
    model_df = pd.DataFrame({'model': model_list,
                             config.METRIC_ACCURACY: [score[config.METRIC_ACCURACY] for score in model_scores],
                            config.METRIC_F1_SCORE: [score[config.METRIC_F1_SCORE] for score in model_scores],
                            config.METRIC_COHEN_KAPPA: [score[config.METRIC_COHEN_KAPPA] for score in\
                                                        model_scores],
                            config.METRIC_CONFUSION_MATRIX: [score[config.METRIC_CONFUSION_MATRIX] for score in\
                                                             model_scores]                            
                             })
    model_df = model_df[['model', config.METRIC_ACCURACY, config.METRIC_F1_SCORE, config.METRIC_COHEN_KAPPA,
                         config.METRIC_CONFUSION_MATRIX]]
    model_df.to_csv(os.path.join(config.WORK_DIRECTORY, 'summary_sequence_model.csv'), index=False)
    model_df.sort_values(by=[config.METRIC_ACCURACY, config.METRIC_F1_SCORE, config.METRIC_COHEN_KAPPA],
                         ascending=False, inplace=True)
    print('Best model:\n' + str(model_df.iloc[0]))
