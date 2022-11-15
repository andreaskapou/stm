# Load libraries
import sys
import os.path as osp

import torch
import pandas as pd
import numpy as np

from scipy.stats import dirichlet, multinomial
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
    # for each word
    for n in range(N):
        # sample the possible topics
        z_n = multinomial.rvs(1, theta)
        # get the identity of the topic; the one with the highest probability
        z = np.argmax(z_n)
        # sample the possible regions from the topic
        w_n = multinomial.rvs(1, phi[z, :])
        # get the identity of the region; the one with the highest probability
        w = np.argmax(w_n)

        if w not in d_dict:
            d_dict[w] = 0
        d_dict[w] = d_dict[w] + 1
        d_str.append('w{}'.format(w))
        d[n] = w
        
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
    
    # Documents-topics distribution
    theta = np.array([dirichlet.rvs(a)[0] for _ in range(nCells)])
    # Terms-topics distribution
    phi = np.array([dirichlet.rvs(b)[0] for _ in range(nTopics)])
    
    
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
    
    obj = dict()
    obj['D'] = D
    obj['D_str'] = D_str
    obj['D_freq'] = D_freq
    #obj['D_tfidf'] = D_tfidf
    obj['theta_true'] = theta
    obj['phi_true'] = phi
    
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

def plot_word_cloud(phi, vocab, ax, title):
    """
    Word cloud visualisation helpful for interpreting topic-word distributions

    :param phi: Vector of word probabilities for specific topic
    :param vocab: Vocabulary array with columns ('index', 'id')
    :param ax: Axis object
    """
    sorted_, indices = torch.sort(phi, descending=True)
    df = pd.DataFrame(indices[:100].numpy(), columns=['index'])
    words = pd.merge(df, vocab[['index', 'id']],
                     how='left', on='index')['id'].values.tolist()
    sizes = (sorted_[:100] * 10).int().numpy().tolist()
    freqs = {words[i]: sizes[i] for i in range(len(words))}
    wc = WordCloud(background_color="white", width=800, height=500)
    wc = wc.generate_from_frequencies(freqs)
    ax.set_title(title)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")