# Load libraries
import sys
import os.path as osp

import torch
import pyro
import pyro.distributions as dist
from pyro.ops.indexing import Vindex

import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
#from sklearn.feature_extraction.text import TfidfTransformer

def simulate_lda_datum(N, theta, phi):
    """
    Simulate a single datum to test LDA

    :param N: Cell size in terms of nUMIs or nCounts
    :param theta: cell-topic distribution
    :param phi: matrix of topic-region probabilities
    """
    d = np.empty([N])
    d_dict = {}
    d_str = []
    
    # Sample topics and corresponding regions
    z = dist.Categorical(theta).sample([N])
    phi_z = Vindex(phi)[z, :]
    w = dist.Categorical(phi_z).sample().detach().tolist()
    # populate information
    for n in range(N):
        if w[n] not in d_dict:
            d_dict[w[n]] = 0
        d_dict[w[n]] = d_dict[w[n]] + 1
        d_str.append('w{}'.format(w[n]))
        d[n] = w[n]
        
    obj = dict()
    obj['d'] = d
    obj['d_str'] = d_str
    obj['d_dict'] = d_dict
    return obj


def simulate_lda_dataset(nTopics, nCells, nRegions, N, a, b):
    """
    Simulate a dataset to test LDA

    :param nTopics: Number of topics
    :param nCells: Number of cells
    :param nRegions: Number of genomic regions
    :param N: Cell sizes in terms of nUMIs or nCounts (list of length nCells) 
    :param a: Dirichlet prior (list of length nTopics) on cell-topics parameter theta
    :param b: Dirichlet prior (list of length nRegions) on topics-regions parameter phi
    """
    
    # Cell topics distribution
    theta = dist.Dirichlet(a).sample([nCells])
    # Topics region distribution
    phi = dist.Dirichlet(b).sample([nTopics])
    
    D = np.empty([nCells, N[1]], dtype = np.int64)
    D_str = []
    D_dict = []
    # Simulate each cell
    for c in range(nCells):
        obj = simulate_lda_datum(N = N[c], theta = theta[c], phi = phi)
        D[c, :] = obj['d']
        D_dict.append(obj['d_dict'])
        D_str.append(' '.join(obj['d_str']))
        
    # make a nice matrix
    # D_freq is a matrix of peak counts (rows are cells, columns are regions, elements are count values)
    D_freq = lil_matrix((nCells, nRegions), dtype = np.int64)
    for i, d in enumerate(D_dict):
        counts = sorted(list(d.items()), key = lambda tup: tup[0])
        for tup in counts:
            D_freq[i, tup[0]] = tup[1]

    # D_tfidf is a matrix of tf-idf (rows are cells, columns are regions, elements are tf-idf values)
    #D_tfidf = TfidfTransformer().fit_transform(D_freq)

    # add vocabulary (region <-> word dataframe)
    region_idx   = list(range(nRegions))
    region_words = ['w'+str(idx) for idx in region_idx]
    vocab = pd.DataFrame(zip(region_words,region_idx),columns=['region_id','region_index'])
    
    obj = dict()
    obj['D'] = D
    obj['D_str'] = D_str
    obj['D_freq'] = D_freq
    #obj['D_tfidf'] = D_tfidf
    obj['theta_true'] = theta
    obj['phi_true'] = phi
    obj['vocab'] = vocab
    
    return obj


def qc_lda_D_entry(D_ent, qc_peak_idx, target_len):
    """
    Function to perform amend (according to quality control) a single
    entry of obj["D"], where obj is a simulated data object.
    Inputs:
    1. D_ent: single row of obj["D"] matrix
    2. qc_peak_idx: array of indices of peak regions passing QC
    3. target_len: length of original obj["D"] matrix rows (nCounts) 
    """
    tmp_ent = [a*b for a,b in zip(D_ent,qc_peak_idx.reshape(-1,1).tolist())]
    tmp_ent = [item for sublist in tmp_ent for item in sublist]
    tmp_ent = np.array(tmp_ent)
    
    # if length < target length
    # have to pad with random regions (chosen frm QC'd regions)
    tmp_ent_len = len(tmp_ent)
    if tmp_ent_len < target_len:
        tmp_ent = np.append(tmp_ent,np.random.choice(qc_peak_idx,
                                        size=target_len-tmp_ent_len, replace=True))
        
    tmp_ent = list(np.sort(tmp_ent))
    
    # create string (for WordCloud visualizations)
    tmp_ent_str = ' '.join(['w'+str(item) for item in tmp_ent])

    return tmp_ent, tmp_ent_str


def qc_lda_simulation_object(obj, peak_detection_qc_thr):
    """
    Function to perform quality control on simulated dataset object.
    Inputs: 
    1. obj: created using simulate_lda_dataset function
    2. peak_detection_qc_thr: QC threshold = proportion of cells in which 
       peak signal should be **detected** in order for it to pass QC.
    """
    qc_peak_idx = np.where(np.array((obj['D_freq']>0).mean(axis=0))[0] > peak_detection_qc_thr)[0]
    print('Number of peak regions passing QC threshold:', len(qc_peak_idx))
    
    obj_qc = dict()
    
    nCells, Nsize = obj['D'].shape
    ## amend D_freq matrix after QC ##
    obj_qc['D_freq'] = obj['D_freq'][:,qc_peak_idx]
    
    ## amend D and D_str matrices after QC-ing D_freq matrix ##
    obj_qc['D'] = []
    obj_qc['D_str'] = []

    for k in range(nCells):

        #tmp_D, tmp_str = qc_D_entry(obj_qc['D_freq'][k].toarray()[0], qc_peak_idx, N[k]) # N='cell size'
        tmp_D, tmp_str = qc_lda_D_entry(obj_qc['D_freq'][k].toarray()[0], qc_peak_idx, Nsize) # Nsize='cell size'
        obj_qc['D'].append(tmp_D)
        obj_qc['D_str'].append(tmp_str)

    obj_qc['D'] = np.array(obj_qc['D'])
    # now need to re-index D matrix so as to form consistent/valid input to pyro routines
    # first, build dictionary of old region ids to new indices
    original_qc_dict = {}
    tmp_original_entries = np.sort(list(set(obj_qc['D'].flatten())))
    for k, entry in enumerate(tmp_original_entries):
        original_qc_dict[entry] = k

    # second, adjust entries of D matrix according to above dictionary
    # for this trick using features of python dictionaries, 
    # see: https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
    obj_qc['D'] = np.vectorize(original_qc_dict.__getitem__)(obj_qc['D']) 

    # finally adjust vocab dataframe
    obj_qc_vocab = obj['vocab'].iloc[qc_peak_idx].copy(deep=True)
    obj_qc_vocab.reset_index(drop=True, inplace=True)
    obj_qc_vocab['region_index'] = obj_qc_vocab.index.tolist()

    print('shape  of D after QC:            ',obj_qc['D'].shape)
    print('length of D_freq after QC:       ',len(obj_qc['D_str']))

    ## amend theta_true (unchanged) and phi_true matrices after QC ##
    obj_qc['theta_true'] = obj['theta_true']
    obj_qc['phi_true']   = obj['phi_true'][:,qc_peak_idx]
    obj_qc['phi_true_norm']   = (obj_qc['phi_true'].T / obj_qc['phi_true'].sum(axis=1)).T
    obj_qc['vocab'] = obj_qc_vocab
    print('shape  of theta_true after QC:   ', obj_qc['theta_true'].shape)
    print('shape  of phi_true after QC:     ', obj_qc['phi_true'].shape)
    print('shape  of phi_true_norm after QC:',  obj_qc['phi_true_norm'].shape)
    
    return obj_qc
    
    
def scale_zero_one(x):
    """
    General function to scale a 1D tensor object to (0, 1)

    :param x: Input 1D tensor
    """
    x_min = torch.min(x)#, dim=1, keepdim=True)
    x_max = torch.max(x)#, dim=1, keepdim=True)
    x = (x - x_min) / (x_max - x_min) # Broadcasting rules apply
    return x