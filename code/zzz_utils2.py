import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def qc_D_entry(D_ent, qc_peak_idx, target_len):
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


def qc_simulation_object(obj, peak_detection_qc_thr):
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
        tmp_D, tmp_str = qc_D_entry(obj_qc['D_freq'][k].toarray()[0], qc_peak_idx, Nsize) # Nsize='cell size'
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


#########################
### plotting routines ###
#########################

# def make_theta_comparison_plots(theta_true,theta_infer):
#     """
#     Quick plot to compare sets of theta values (presumably true vs inferred)
#     Inputs:
#     theta_true: array of theta values (true)
#     theta_infer: array of theta values (inferred)
#     """
#     
#     ntopics_true  = theta_true.shape[1]
#     ntopics_infer = theta_infer.shape[1] 
#     fig, axs = plt.subplots(ntopics_true,ntopics_infer,
#                             figsize=(ntopics_true*1.5,ntopics_true*1.5))
#     
#     for i in range(ntopics_true):
#         for j in range(ntopics_infer):
#             axs[i,j].plot(theta_true[:,i],theta_infer[:,j],'o',markersize=2,alpha=0.5)
#             axs[i,j].set_xlim(0,1)
#             axs[i,j].set_ylim(0,1)
#             axs[i,j].grid(True)
#             
#     
#     plt.tight_layout()
#     return fig
# 
# 
# def plot_word_cloud(phi, vocab, max_ids, ax, title):
#     """
#     Word cloud visualisation helpful for interpreting topic-word distributions
# 
#     :param phi: Vector of word probabilities for specific topic
#     :param vocab: Vocabulary array with columns ('id', 'index')
#     :param max_ids: Maximum number of word ids to plot.
#     :param ax: Axis object
#     :param title: Plot title
#     """
#     sorted_, indices = torch.sort(phi, descending=True)
#     df = pd.DataFrame(indices[:max_ids].numpy(), columns=['region_index'])
#     words = pd.merge(df, vocab[['region_index', 'region_id']],
#                      how='left', on='region_index')['region_id'].values.tolist()
#     sizes = (sorted_[:100] * 10).int().numpy().tolist()
#     freqs = {words[i]: sizes[i] for i in range(len(words))}
#     wc = WordCloud(background_color="white", width=800, height=500)
#     wc = wc.generate_from_frequencies(freqs)
#     ax.set_title(title)
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis("off")
#     
# def make_word_cloud_figure(phi, vocab, max_ids):
#     """
#     Make grid of wordclouds to visualize topic-word distributions
# 
#     :param phi: Vector of word probabilities for specific topic
#     :param vocab: Vocabulary array with columns ('id', 'index')
#     :param max_ids: Maximum number of word ids to plot.
#     """
#     
#     ntopics = phi.shape[0]
#     ncols = 4
#     nrows = int(np.floor(ntopics/4)) + 1
#     nremain = (ncols * nrows) - ntopics
#     
#     fig, axs = plt.subplots(nrows, ncols, figsize=(12, nrows * 3.))
#     for n in range(ntopics):
#         i, j = divmod(n, ncols)
#         plot_word_cloud(scale_zero_one(phi[n]), vocab, 
#                         max_ids, axs[i, j], 'Topic %d' % (n + 1))
#         
#     for j in range(nremain):
#         axs[-1,-j-1].axis('off')
# 
#     plt.tight_layout()
#     return fig
