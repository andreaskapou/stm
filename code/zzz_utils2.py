import numpy as np

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
    print('shape  of D after QC:            ',obj_qc['D'].shape)
    print('length of D_freq after QC:       ',len(obj_qc['D_str']))

    ## amend theta_true (unchanged) and phi_true matrices after QC ##
    obj_qc['theta_true'] = obj['theta_true']
    obj_qc['phi_true']   = obj['phi_true'][:,qc_peak_idx]
    obj_qc['phi_true_norm']   = (obj_qc['phi_true'].T / obj_qc['phi_true'].sum(axis=1)).T
    print('shape  of theta_true after QC:   ', obj_qc['theta_true'].shape)
    print('shape  of phi_true after QC:     ', obj_qc['phi_true'].shape)
    print('shape  of phi_true_norm after QC:',  obj_qc['phi_true_norm'].shape)
    
    return obj_qc
