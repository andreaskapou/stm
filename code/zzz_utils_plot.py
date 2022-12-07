# Load libraries
import sys
import os.path as osp

import torch

import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix

import matplotlib.pyplot as plt
from wordcloud import WordCloud


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
