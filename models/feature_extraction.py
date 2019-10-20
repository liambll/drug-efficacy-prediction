# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 17:12:29 2019

@author: liam.bui

This file contains functions to extract different "hand-crafted" features for images
"""

import numpy as np
from sklearn.feature_selection import chi2
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem, ChemicalFeatures, Descriptors, Crippen, Lipinski
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
fdefName = 'models/MinimalFeatures.fdef'
featFactory = ChemicalFeatures.BuildFeatureFactory(fdefName)
from rdkit.Chem.Pharm2D.SigFactory import SigFactory


####################
## 1D descriptors ##
####################
def extract_properties(column, include_3D=False):
    """Extract various 1D descriptors
    https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors
    
    :param column: Pandas Series, containing smiles or RDKit mol object
    :param from_smiles: bool, indicate whether column contains smiles string
    :return: feature_properties: Pandas Series, containing 1D descriptors
    """

    def extract(x, from_smiles):
        if from_smiles:
            mol = Chem.MolFromSmiles(x)
        else:
            mol = x
        
        if (mol is None) or (len(mol.GetAtoms()) == 0):
            if include_3D:
                return [0] * 29
            else:
                return [0] * 24
        else:
            logP = Crippen.MolLogP(mol)
            refractivity = Crippen.MolMR(mol)
            
            weight = Descriptors.MolWt(mol)
            exact_weight = Descriptors.ExactMolWt(mol)
            heavy_weight = Descriptors.HeavyAtomMolWt(mol)
            heavy_count = Lipinski.HeavyAtomCount(mol)
            nhoh_count = Lipinski.NHOHCount(mol)
            no_count = Lipinski.NOCount(mol)
            hacceptor_count = Lipinski.NumHAcceptors(mol)
            hdonor_count = Lipinski.NumHDonors(mol)
            hetero_count = Lipinski.NumHeteroatoms(mol)
            rotatable_bond_count = Lipinski.NumRotatableBonds(mol)
            valance_electron_count = Descriptors.NumValenceElectrons(mol)
            amide_bond_count = rdMolDescriptors.CalcNumAmideBonds(mol)
            aliphatic_ring_count = Lipinski.NumAliphaticRings(mol)
            aromatic_ring_count = Lipinski.NumAromaticRings(mol)
            saturated_ring_count = Lipinski.NumSaturatedRings(mol)
            aliphatic_cycle_count = Lipinski.NumAliphaticCarbocycles(mol)
            aliphaticHetero_cycle_count = Lipinski.NumAliphaticHeterocycles(mol)
            aromatic_cycle_count = Lipinski.NumAromaticCarbocycles(mol)
            aromaticHetero_cycle_count = Lipinski.NumAromaticHeterocycles(mol)
            saturated_cycle_count = Lipinski.NumSaturatedCarbocycles(mol)
            saturatedHetero_cycle_count = Lipinski.NumSaturatedHeterocycles(mol)
            
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            
            if include_3D:
                mol_3D=Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol_3D)
                AllChem.MMFFOptimizeMolecule(mol_3D)
                eccentricity = rdMolDescriptors.CalcEccentricity(mol_3D)
                asphericity = rdMolDescriptors.CalcAsphericity(mol_3D)
                spherocity = rdMolDescriptors.CalcSpherocityIndex(mol_3D)
                inertial = rdMolDescriptors.CalcInertialShapeFactor(mol_3D)
                gyration = rdMolDescriptors.CalcRadiusOfGyration(mol_3D)
            
                return [logP, refractivity, weight, exact_weight, heavy_weight, heavy_count, nhoh_count, no_count,
                        hacceptor_count, hdonor_count, hetero_count, rotatable_bond_count, valance_electron_count,
                        amide_bond_count, aliphatic_ring_count, aromatic_ring_count, saturated_ring_count,
                        aliphatic_cycle_count, aliphaticHetero_cycle_count, aromatic_cycle_count,
                        aromaticHetero_cycle_count, saturated_cycle_count, saturatedHetero_cycle_count, tpsa,
                        eccentricity, asphericity, spherocity, inertial, gyration]
            else:
                return [logP, refractivity, weight, exact_weight, heavy_weight, heavy_count, nhoh_count, no_count,
                        hacceptor_count, hdonor_count, hetero_count, rotatable_bond_count, valance_electron_count,
                        amide_bond_count, aliphatic_ring_count, aromatic_ring_count, saturated_ring_count,
                        aliphatic_cycle_count, aliphaticHetero_cycle_count, aromatic_cycle_count,
                        aromaticHetero_cycle_count, saturated_cycle_count, saturatedHetero_cycle_count, tpsa]
                
    feature_properties = column.apply(lambda x: extract(x))

    return np.array(list(feature_properties))


####################
## 2D descriptors ##
####################
def extract_MQNs(column, from_smiles=True):
    """Extract MQN features from smiles
    :param column: Pandas Series, containing smiles or RDKit mol object
    :param from_smiles: bool, indicate whether column contains smiles string
    :return feature_MQN: Pandas Series, containing 42 MQN feature
    """
    
    def get_MQNs(x, from_smiles):
        if from_smiles:
            mol = Chem.MolFromSmiles(x)
        else:
            mol = x
        if (mol is None) or (len(mol.GetAtoms()) == 0):
            return [0]*42
        else:
           return rdMolDescriptors.MQNs_(mol) 
       
    feature_MQN = column.apply(lambda x: get_MQNs(x, from_smiles))
    return np.array(list(feature_MQN))


def extract_Morganfp(column, radius=2, nBits=2048, useFeatures=False, from_smiles=True):
    """Extract Morganfingerprint
    :param column: Pandas Series, containing smiles or RDKit mol object
    :param radius: int, indicates the radius in the Morgan fingerprint calculation.
    :param nBits: int, the number of bits in the resulting bit vector.
    :param useFeatures: bool, whether atoms' specific features are used
    :param from_smiles: bool, indicate whether column contains smiles string
    :return: feature_morgan: Pandas Series, containing Morganfingerprint features
    """
    
    def get_Morganfp(x, from_smiles):
        if from_smiles:
            mol = Chem.MolFromSmiles(x)
        else:
            mol = x
        if (mol is None) or (len(mol.GetAtoms()) == 0):
            return [0]*nBits
        else:
           return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useFeatures=useFeatures) 
        
    feature_morgan = column.apply(lambda x: get_Morganfp(x, from_smiles))
    return np.array(list(feature_morgan))


def extract_Pharm2D(column, minPointCount=2, maxPointCount=3, bins=[(0,2),(2,5),(5,8)], from_smiles=True):
    """Extract Pharm2D fingerprint
    :param column: Pandas Series, containing smiles or RDKit mol object
    :param minPointCount: int
    :param maxPointCount: int
    :param bins: lits of tuples
    :param from_smiles: bool, indicate whether column contains smiles string
    :return: feature_Pharm2D: Pandas Series, containing Pharm2D features
    """
    sigFactory = SigFactory(featFactory,
                            minPointCount=minPointCount,
                            maxPointCount=maxPointCount,
                            trianglePruneBins=False)
    sigFactory.SetBins(bins)
    sigFactory.Init()
    
    def get_Pharm2D(x):
        mol = Chem.MolFromSmiles(x)
        if (mol is None) or (len(mol.GetAtoms()) == 0):
            return [0]*sigFactory.GetSigSize()
        else:
           return Generate.Gen2DFingerprint(mol, sigFactory)     

    fp = column.apply(lambda x: get_Pharm2D(x))
    return np.array(list(fp))


def extract_Gobbi_Pharm2D(column, from_smiles=True):
    """Extract Gobbi Pharm2D fingerprint
    :param column: Pandas Series, containing smiles or RDKit mol object
    :param from_smiles: bool, indicate whether column contains smiles string
    :return: feature_Gobbi Pharm2D: Pandas Series, containing Gobbi Pharm2D  features
    """
    
    def get_Gobbi_Pharm2D(x, from_smiles):
        if from_smiles:
            mol = Chem.MolFromSmiles(x)
        else:
            mol = x
        return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
        
    feature_Gobbi = column.apply(lambda x: get_Gobbi_Pharm2D(x, from_smiles))
    return np.array(list(feature_Gobbi))


####################
## 3D descriptors ##
####################
def extract_RDF(column, from_smiles=True):
    """Extract RDF descriptor
    :param column: Pandas Series, containing smiles or RDKit mol object
    :param from_smiles: bool, indicate whether column contains smiles string
    :return: feature_RDF: Pandas Series, containing 210 RDF features
    """
    def get_RDF(x, from_smiles):
        if from_smiles:
            mol = Chem.MolFromSmiles(x)
        else:
            mol = x
        if (mol is None) or (len(mol.GetAtoms()) == 0):
            return [0]*210
        else:
            mol_3D=Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_3D)
            AllChem.MMFFOptimizeMolecule(mol_3D)
            return rdMolDescriptors.CalcRDF(mol_3D) 
        
    feature_RDF = column.apply(lambda x: get_RDF(x, from_smiles))
    return np.array(list(feature_RDF))


def extract_AUTOCORR3D(column, from_smiles=True):
    """Extract AUTOCORR3D descriptor
    :param column: Pandas Series, containing smiles or RDKit mol object
    :param from_smiles: bool, indicate whether column contains smiles string
    :return: feature_AUTOCORR3D: Pandas Series, containing 80 AUTOCORR3D features
    """
    def get_AUTOCORR3D(x, from_smiles):
        if from_smiles:
            mol = Chem.MolFromSmiles(x)
        else:
            mol = x
        if (mol is None) or (len(mol.GetAtoms()) == 0):
            return [0]*80
        else:
            mol_3D=Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_3D)
            AllChem.MMFFOptimizeMolecule(mol_3D)
            return rdMolDescriptors.CalcAUTOCORR3D(mol_3D) 
        
    feature_AUTOCORR3D = column.apply(lambda x: get_AUTOCORR3D(x, from_smiles))
    return np.array(list(feature_AUTOCORR3D)), np.arange(80)


def extract_MORSE(column, from_smiles=True):
    """Extract MORSE descriptor
    :param column: Pandas Series, containing smiles or RDKit mol object
    :param from_smiles: bool, indicate whether column contains smiles string
    :return: feature_MORSE: Pandas Series, containing 224 MORSE features
    """
    def get_MORSE(x, from_smiles):
        if from_smiles:
            mol = Chem.MolFromSmiles(x)
        else:
            mol = x
        if (mol is None) or (len(mol.GetAtoms()) == 0):
            return [0]*224
        else:
            mol_3D=Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_3D)
            AllChem.MMFFOptimizeMolecule(mol_3D)
            return rdMolDescriptors.CalcMORSE(mol_3D) 
        
    feature = column.apply(lambda x: get_MORSE(x, from_smiles))
    return np.array(list(feature)), np.arange(224)


def extract_WHIM(column, from_smiles=True):
    """Extract WHIM descriptor
    :param column: Pandas Series, containing smiles or RDKit mol object
    :param from_smiles: bool, indicate whether column contains smiles string
    :return: feature_WHIM: Pandas Series, containing 114 WHIM features
    """
    def get_WHIM(x, from_smiles):
        if from_smiles:
            mol = Chem.MolFromSmiles(x)
        else:
            mol = x
        if (mol is None) or (len(mol.GetAtoms()) == 0):
            return [0]*114
        else:
            mol_3D=Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_3D)
            AllChem.MMFFOptimizeMolecule(mol_3D)
            return rdMolDescriptors.CalcWHIM(mol_3D) 
        
    feature = column.apply(lambda x: get_WHIM(x, from_smiles))
    return np.array(list(feature)), np.arange(114)


def extract_GETAWAY(column, from_smiles=True):
    """Extract GETAWAY descriptor. GETAWAT descriptors have NaN values sometimes.
    :param column: Pandas Series, containing smiles or RDKit mol object
    :param from_smiles: bool, indicate whether column contains smiles string
    :return: feature_GETAWAY: Pandas Series, containing 273 GETAWAY features
    """
    def get_GETAWAY(x):
        mol = Chem.MolFromSmiles(x)
        if (mol is None) or (len(mol.GetAtoms()) == 0):
            return [0]*273
        else:
            mol_3D=Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_3D)
            AllChem.MMFFOptimizeMolecule(mol_3D)
            return rdMolDescriptors.CalcGETAWAY(mol_3D) 
        
    feature = column.apply(lambda x: get_GETAWAY(x))
    return np.array(list(feature)), np.arange(273)


def extract_features(column, method=['morgan'], from_smiles=True):
    """Extract 1D, 2D and 3D descriptors
    :param column: Pandas Series, containing smiles or RDKit mol object
    :param method: list, containing names of descriptors to extract
    :param from_smiles: bool, indicate whether column contains smiles string
    :return features: list of features extracted with method list
    """

    feature_list = []
    if 'morgan' in method:
        feature_list.append(extract_Morganfp(column))
    if 'mqn' in method:
        feature_list.append(extract_MQNs(column))
    if 'pharm2D' in method:
        feature_list.append(extract_Pharm2D(column))
    if 'gobbi' in method:
        feature_list.append(extract_Gobbi_Pharm2D(column))
    if 'physical' in method:
        feature_list.append(extract_properties(column, include_3D=False))
    if 'physical3D' in method:
        feature_list.append(extract_properties(column, include_3D=True))
    if 'autocorr3D' in method:
        feature_list.append(extract_AUTOCORR3D(column))
    if 'rdf' in method:
        feature_list.append(extract_RDF(column))
    if 'morse' in method:
        feature_list.append(extract_MORSE(column))
    if 'whim' in method:
        feature_list.append(extract_WHIM(column))
    if 'getaway' in method:
        feature_list.append(extract_GETAWAY(column))

    return np.concatenate(feature_list, axis=1)
        
     
def filter_feature_Chi2(feature_column, target_column, threshold=None):
    """Filter feature using Chi2 test
    :param: feature_column: numpy array containing feature
    :param: target_column: numpy array containing target variable
    :param: threshold: threshold to filter using Chi2.
                        threshold=None means all features with non Nan Chi2 pval will be returned.
    :return: feature_selected: features that are signficant
    :return: pval_significant: list of bool to indicate which features are significant
    """

    # Perform Chi2 test
    chi2_stats, pval = chi2(feature_column, target_column)

    # select only significant pvals
    pval_result = pval.copy()
    if threshold:
        pval_result[np.isnan(pval_result)] = 100  # replace Nan with any large value
        pval_significant = pval_result <= threshold
    else:
        pval_significant = np.logical_not(np.isnan(pval))

    # select features with significant pvals
    feature_selected = feature_column[:, pval_significant]

    return feature_selected, pval_significant