# DRUG EFFICACY PREDICTION
## Overview:
The project aims to build a model to predict drug efficacy of molecules.

A pre-processed HIV dataset with 3 classess (CA - Confirmed active, CM - Confirmed moderately active, CI - Confirmed inactive and benign) is available at: http://moleculenet.ai/datasets-1. The raw data is available here: https://wiki.nci.nih.gov/display/NCIDTPdata/AIDS+Antiviral+Screen+Data<br/>
We perform stratified random splitting on the dataset: 80% of the images are in train set and 20% of the images are in test set. 

## Model Architecture
**1. Classical Machine learning with 1D, 2D and 3D molecular descriptors**
<img src="assets/stacking_model.jpg" alt="" width="80%"><br/>

* We perform feature extraction by contructing a graph structure representing the molecules and calculating 1D descriptors (logP, Ring Count, Aromaticity, etc), 2D descriptors(MQNs, Morgan Fingerprints, 2D Pharmacophore Fingerprints, etc) and 3D descriptors (MORSE, WHIM, etc) for the molecules.
* Using these molecular descriptors, we train base models with different hyper parameters
* The predictions made by these base models are added to the feature space. We then train Level 1 Model Stacking using Cross-Validation on trainset

**2. Graph Convolution Network to learn neural fingerprints **
<img src="assets/gcn_model.jpg" alt="" width="80%"><br/>

* Instead of molecular descriptors, we can train a graph convolutional network on the molecules' graph structures to learn vector representation of molecules. https://arxiv.org/abs/1609.02907

**3. Deep learning and Transfer learning to learn image features (representation learning with Convolutional Neural Network)**
<img src="assets/smiles2vec_model.jpg" alt="" width="80%"><br/>

* Instead of constructing graph representation of molecules, we can use a recurrent neural network on the molecules' SMILES string to learn vector representation of molecules. https://arxiv.org/pdf/1712.02034.pdf

## Model Performance
**1. Evaluation Metrics**

We will look at several evaluation metrics:
* **Accuracy**: Accuracy simply measures proportion of predictions that are correct. . It ranges between 0 (worst) and 1 (best)
* **F1-Score**: F1-Score is the harmonic mean of precision and recall. It ranges between 0 (worst) and 1 (best)
* **Cohen’s kappa**: Cohen’s kappa statistics measures how agreeable the prediction and the true label are. It ranges between -1 (completely disagree) and 1 (completely agree).

**2. Performance Report**

The models' performance on test set is reported below. The best model is a convolutional neural network with InceptionV3 backbone.

| Model  | Accuracy | F1-Score | Cohen’s kappa |
| ------ | -------- | -------- | ------------- |

<br/>

Confusion Matrix of the best model's prediction on test set:<br/>

## Instruction
