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
from keras.layers import Input, Dense
from keras.models import Model
from keras import callbacks
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight

from spektral.layers import GraphConv, GlobalAvgPool
from spektral.layers import EdgeConditionedConv, GlobalAttentionPool
from spektral.layers.ops import sp_matrix_to_sp_tensor_value
from spektral.utils import Batch, batch_iterator
from spektral.utils import label_to_one_hot
import networkx as nx
from spektral.utils import nx_to_numpy
from rdkit import Chem

import warnings
import argparse
import gc
from utils.data import read_data, get_prediction_score
from utils import config


def convert_mol_to_graph(mol):
    """
    Convert RDKit mol into a graph representation atoms are nodes, and bonds are vertices store as networkx graph
    :param mol: RDKit mol
    :return G: networkx graph
    """
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum())
    for bond in mol.GetBonds():
      G.add_edge(bond.GetBeginAtomIdx(),
               bond.GetEndAtomIdx(),
               bond_type=bond.GetBondType())      
    return G


def generate_graph_matrices(graphs, auto_pad=False):
    """
    Generate A, X, E matrix from smiles
    :param graphs: list of networkx graphs
    :param auto_pad: bool. whether to pad the node matrix to have the same length
    :return A, X, E
    """ 
    
    A, X, E = nx_to_numpy(graphs, nf_keys=['atomic_num'],
                               ef_keys=['bond_type'], auto_pad=auto_pad, self_loops=True)
    
    uniq_X = np.unique([v for x in X for v in np.unique(x)])
    X = [label_to_one_hot(x, uniq_X) for x in X]
    uniq_E = np.unique([v for x in E for v in np.unique(x)])
    E = [label_to_one_hot(x, uniq_E) for x in E]
    
    return A, X, E
    
    
def build_gcn_model(trainset, testset, nb_node_features, nb_classes=1, batch_size=32, nb_epochs=100, lr=0.001,
                     save_path=None):
    """Build a Graph Convolutional Network model
    :param trainset: [A_train, X_train, y_train]
    :param testset: [A_test, X_test, y_test]
    :param nb_node_features: int, number of node features
    :param nb_classes: int, number of output classes
    :param batch_size: int, batch size for model training
    :param nb_epochs: int, number of training epoches
    :param lr: float, learning rate
    :param save_path: path to save model
    :return model: fitted Keras model
    :return scores: dict, scores on test set for the fitted Keras model
    """ 
    
    # Create model architecture
    X_in = Input(batch_shape=(None, nb_node_features))
    A_in = Input(batch_shape=(None, None), sparse=True)
    I_in = Input(batch_shape=(None, ), dtype='int64')
    target = Input(tensor=tf.placeholder(tf.float32, shape=(None, nb_classes), name='target'))
    
    gc1 = GraphConv(64, activation='relu')([X_in, A_in])
    gc2 = GraphConv(128, activation='relu')([gc1, A_in])
    pool = GlobalAvgPool()([gc2, I_in])
    dense1 = Dense(128, activation='relu')(pool)
    output = Dense(nb_classes, activation='sigmoid')(dense1)
    
    model = Model(inputs=[X_in, A_in, I_in], outputs=output)
    
    # Compile model
    #optimizer = Adam(lr=lr)    
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', target_tensors=target, metrics=['accuracy'])
    model.summary()
    loss = model.total_loss
    train_step = opt.minimize(loss)
    
    # Initialize all variables
    sess = K.get_session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    # Get train and test data
    [A_train, X_train, y_train] = trainset
    [A_test, X_test, y_test] = testset
    
    SW_KEY = 'dense_2_sample_weights:0' # Keras automatically creates a placeholder for sample weights, which must be fed
    best_accuracy = 0
    for i in range(nb_epochs):
        # Train
        # TODO: compute class weight and use it in loss function
        batches_train = batch_iterator([A_train, X_train, y_train], batch_size=batch_size)
        model_loss = 0
        prediction = []
        for b in batches_train:
            batch = Batch(b[0], b[1])
            X_, A_, I_ = batch.get('XAI')
            y_ = b[2]
            tr_feed_dict = {X_in: X_,
                            A_in: sp_matrix_to_sp_tensor_value(A_),
                            I_in: I_,
                            target: y_,
                            SW_KEY: np.ones((1,))}
            outs = sess.run([train_step, loss, output], feed_dict=tr_feed_dict)
            model_loss += outs[1]
            prediction.append(list(outs[2].flatten()))    
        y_train_predict = (np.concatenate(prediction)[:len(y_train)] > 0.5).astype('uint8')
        train_accuracy = accuracy_score(y_train, y_train_predict)
        train_loss = model_loss / (np.ceil(len(y_train) / batch_size))
        
        # Validation
        batches_val = batch_iterator([A_test, X_test, y_test], batch_size=batch_size)
        model_loss = 0
        prediction = []
        
        for b in batches_val:
            batch = Batch(b[0], b[1])
            X_, A_, I_ = batch.get('XAI')
            y_ = b[2]
            tr_feed_dict = {X_in: X_,
                            A_in: sp_matrix_to_sp_tensor_value(A_),
                            I_in: I_,
                            target: y_,
                            SW_KEY: np.ones((1,))}
            loss_, output_ = sess.run([loss, output], feed_dict=tr_feed_dict)
            model_loss += loss_
            prediction.append(list(output_.flatten()))
        
        y_val_predict = (np.concatenate(prediction)[:len(y_test)] > 0.5).astype('uint8')
        val_accuracy = accuracy_score(y_test, y_val_predict)
        val_loss = model_loss / (np.ceil(len(y_test) / batch_size))
        print('---------------------------------------------')
        print('Epoch {}: train_loss: {}, train_acc: {}, val_loss: {}, val_acc: {}'.format(i+1, train_loss, train_accuracy,
              val_loss, val_accuracy))
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model.save(save_path)
        
    # Evaluate the model
    model = load_model(save_path)
    batches_val = batch_iterator([A_test, X_test, y_test], batch_size=batch_size)
    prediction = []
    for b in batches_val:
        batch = Batch(b[0], b[1])
        X_, A_, I_ = batch.get('XAI')
        y_ = b[2]
        tr_feed_dict = {X_in: X_,
                        A_in: sp_matrix_to_sp_tensor_value(A_),
                        I_in: I_,
                        target: y_,
                        SW_KEY: np.ones((1,))}
        output_ = sess.run([output], feed_dict=tr_feed_dict)
        prediction.append(list(output_.flatten()))
    
    y_val_predict = (np.concatenate(prediction)[:len(y_test)] > 0.5).astype('uint8')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # disable the warning on f1-score with not all labels
        scores = get_prediction_score(y_val, y_val_predict)
        
    return model, scores


def build_ecn_model(trainset, testset, nb_nodes, nb_node_features, nb_edge_features, nb_classes=1,
                    batch_size=32, nb_epochs=100, lr=0.001, save_path=None):
    """Build an Edge Convolutional Network model
    :param trainset: [X_train, A_train, E_train, y_train]
    :param testset: [X_test, A_test, E_test, y_test]
    :param nb_nodes: int, number of nodes in a graph (after auto-pad)
    :param nb_node_features: int, number of node features
    :param nb_edge_features: int, number of edge features
    :param nb_classes: int, number of output classes
    :param batch_size: int, batch size for model training
    :param nb_epochs: int, number of training epoches
    :param lr: float, learning rate
    :param save_path: path to save model
    :return model: fitted Keras model
    :return scores: dict, scores on test set for the fitted Keras model
    """ 
    
    # Create model architecture
    X_in = Input(shape=(nb_nodes, nb_node_features))
    A_in = Input(shape=(nb_nodes, nb_nodes))
    E_in = Input(shape=(nb_nodes, nb_nodes, nb_edge_features))
    gc1 = EdgeConditionedConv(32, activation='relu')([X_in, A_in, E_in])
    gc2 = EdgeConditionedConv(64, activation='relu')([gc1, A_in, E_in])
    pool = GlobalAttentionPool(128)(gc2)
    dense1 = Dense(128, activation='relu')(pool)
    output = Dense(nb_classes, activation='sigmoid')(dense1)
    
    # Build model
    model = Model(inputs=[X_in, A_in, E_in], outputs=output)
    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # Callback list
    callback_list = []
    # monitor val_loss and terminate training if no improvement
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, \
                patience=20, verbose=2, mode='auto', restore_best_weights=True)
    callback_list.append(early_stop)
    
    if save_path is not None:
        # save best model based on val_acc during training
        checkpoint = callbacks.ModelCheckpoint(save_path, monitor='val_acc', \
                    verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        callback_list.append(checkpoint)
        
    # Get train and test data
    [X_train, A_train, E_train, y_train] = trainset
    [X_test, A_test, E_test, y_test] = testset
    
    # Compute class weights
    weight_list = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    weight_dict = {}
    for i in range(len(np.unique(y_train))):
        weight_dict[np.unique(y_train)[i]] = weight_list[i]
    
    # Train model
    model.fit([X_train, A_train, E_train],
              y_train,
              batch_size=batch_size,
              validation_data = ([X_val, A_val, E_val], y_val),
              epochs=nb_epochs,
              verbose=2,
              class_weight=weight_dict,
              callbacks=callback_list)
    
    # Evaluate model    
    prediction = model.predict([X_val, A_val, E_val])
    y_val_predict = (prediction > 0.5).astype('uint8')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # disable the warning on f1-score with not all labels
        scores = get_prediction_score(y_val, y_val_predict)
        
    return model, scores

    
if __name__ == '__main__':
    data_path = os.path.join(config.WORK_DIRECTORY, config.DATA_FILE)
    model_list = ['gcn', 'ecn']
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
        os.makedirs(os.path.join(save_path, 'graph_models'), exist_ok=True)
        
    # Read data
    smiles, y = read_data(data_path, col_smiles='smiles', col_target='HIV_active')
    graphs = smiles.apply(lambda x: convert_mol_to_graph(Chem.MolFromSmiles(x))).values
    
    # Build en evaluate graph models
    model_scores = []
    if 'gcn' in model_list: # Graph Convolution Network
        # Extract features
        A, X, E = generate_graph_matrices(graphs, auto_pad=False)
        nb_node_features = X[0].shape[-1]    # Dimension of node features
    
        # Get train and test set
        A_train, A_val, \
        X_train, X_val, \
        E_train, E_val, \
        y_train, y_val = train_test_split(A, X, E, y, test_size=config.TEST_RATIO, shuffle=True, stratify=y,
                                          random_state=config.SEED)    
        
        # Build GCN model
        model, scores = build_gcn_model([A_train, X_train, y_train], [A_val, X_val, y_val],
                                        nb_node_features, nb_classes=1, batch_size=32, nb_epochs=100, lr=0.001,
                         save_path=os.path.join(save_path, 'graph_models', 'gcn.h5'))
        model_scores.append(scores)
            
        # force release memory
        K.clear_session()
        del model
        gc.collect()
        
    if 'ecn' in model_list: # Edge Convolution Network
        # Extract features
        A, X, E = generate_graph_matrices(graphs, auto_pad=True)
        nb_nodes = X.shape[0] # Number of nodes in the graphs
        nb_node_features = X.shape[-1] # Node features dimensionality
        nb_edge_features = E.shape[-1] # Edge features dimensionality
    
        # Get train and test set
        A_train, A_val, \
        X_train, X_val, \
        E_train, E_val, \
        y_train, y_val = train_test_split(A, X, E, y, test_size=config.TEST_RATIO, shuffle=True, stratify=y,
                                          random_state=config.SEED)    
        
        # Build GCN model
        nb_node_features = X[0].shape[-1]    # Dimension of node features
        model, scores = build_gcn_model([X_train, A_train, E_train, y_train], [X_val, A_val, E_val, y_val],
                                        nb_nodes, nb_node_features, nb_edge_features, nb_classes=1,
                                        batch_size=32, nb_epochs=100, lr=0.001,
                                        save_path=os.path.join(save_path, 'graph_models', 'ecn.h5'))
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
    model_df.to_csv(os.path.join(config.WORK_DIRECTORY, 'summary_graph_model.csv'), index=False)
    model_df.sort_values(by=[config.METRIC_ACCURACY, config.METRIC_F1_SCORE, config.METRIC_COHEN_KAPPA],
                         ascending=False, inplace=True)
    print('Best model:\n' + str(model_df.iloc[0]))
