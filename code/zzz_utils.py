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

import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
    
    
def scale_zero_one(x):
    """
    General function to scale a 1D tensor object to (0, 1)

    :param x: Input 1D tensor
    """
    x_min = torch.min(x)#, dim=1, keepdim=True)
    x_max = torch.max(x)#, dim=1, keepdim=True)
    x = (x - x_min) / (x_max - x_min) # Broadcasting rules apply
    return x

#########################
### plotting routines ###
#########################

def make_theta_comparison_plots(theta_true,theta_infer):
    """
    Quick plot to compare sets of theta values (presumably true vs inferred)
    Inputs:
    theta_true: array of theta values (true)
    theta_infer: array of theta values (inferred)
    """
    
    ntopics_true  = theta_true.shape[1]
    ntopics_infer = theta_infer.shape[1] 
    fig, axs = plt.subplots(ntopics_true,ntopics_infer,
                            figsize=(ntopics_true*1.5,ntopics_true*1.5))
    
    for i in range(ntopics_true):
        for j in range(ntopics_infer):
            axs[i,j].plot(theta_true[:,i],theta_infer[:,j],'o',markersize=2,alpha=0.5)
            axs[i,j].set_xlim(0,1)
            axs[i,j].set_ylim(0,1)
            axs[i,j].grid(True)
            
    
    plt.tight_layout()
    return fig


# def plot_word_cloud(phi, vocab, max_ids, ax, title):
#     """
#     Word cloud visualisation helpful for interpreting topic-word distributions
# 
#     :param phi: Vector of word probabilities for specific topic
#     :param vocab: Vocabulary array with columns ('index', 'id')
#     :param max_ids: Maximum number of word ids to plot.
#     :param ax: Axis object
#     :param title: Plot title
#     """
#     sorted_, indices = torch.sort(phi, descending=True)
#     df = pd.DataFrame(indices[:max_ids].numpy(), columns=['index'])
#     words = pd.merge(df, vocab[['index', 'id']],
#                      how='left', on='index')['id'].values.tolist()
#     sizes = (sorted_[:100] * 10).int().numpy().tolist()
#     freqs = {words[i]: sizes[i] for i in range(len(words))}
#     wc = WordCloud(background_color="white", width=800, height=500)
#     wc = wc.generate_from_frequencies(freqs)
#     ax.set_title(title)
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis("off")



def plot_word_cloud(phi, vocab, max_ids, ax, title):
    """
    Word cloud visualisation helpful for interpreting topic-word distributions

    :param phi: Vector of word probabilities for specific topic
    :param vocab: Vocabulary array with columns ('id', 'index')
    :param max_ids: Maximum number of word ids to plot.
    :param ax: Axis object
    :param title: Plot title
    """
    sorted_, indices = torch.sort(phi, descending=True)
    df = pd.DataFrame(indices[:max_ids].numpy(), columns=['region_index'])
    words = pd.merge(df, vocab[['region_index', 'region_id']],
                     how='left', on='region_index')['region_id'].values.tolist()
    sizes = (sorted_[:100] * 10).int().numpy().tolist()
    freqs = {words[i]: sizes[i] for i in range(len(words))}
    wc = WordCloud(background_color="white", width=800, height=500)
    wc = wc.generate_from_frequencies(freqs)
    ax.set_title(title)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    

def make_word_cloud_figure(phi, vocab, max_ids):
    """
    Make grid of wordclouds to visualize topic-word distributions

    :param phi: Vector of word probabilities for specific topic
    :param vocab: Vocabulary array with columns ('id', 'index')
    :param max_ids: Maximum number of word ids to plot.
    """
    
    ntopics = phi.shape[0]
    ncols = 4
    nrows = int(np.floor(ntopics/4)) + 1
    nremain = (ncols * nrows) - ntopics
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, nrows * 3.))
    axs = axs.ravel()
    for n in range(ntopics):
        plot_word_cloud(scale_zero_one(phi[n]), vocab, 
                        max_ids, axs[n], 'Topic %d' % (n + 1))
        
    for j in range(nremain):
        axs[-j-1].axis('off')

    plt.tight_layout()
    return fig
