import numpy as np
import pandas as pd
from ete3 import Tree, TreeStyle
import re


def build_dictionary(tree_sequence, tree_confidence, filename='one_tree.csv'):

    depths = np.array(tree_sequence, dtype=int)[:,2]
    transitions = np.array(tree_sequence,dtype=int)[:,:2]
    counts = np.max(np.bincount(np.array(depths)))
        
    dct = {}
    for depth in list(set(depths)):
        dct[depth] = []
    for depth in list(set(depths)):
        dct['{} confidents'.format(depth)] = []
    for idx, depth in enumerate(depths):
        dct[depth].append(transitions[idx])
        dct['{} confidents'.format(depth)].append(tree_confidence[idx])
    for depth in list(set(depths)):
        dct[depth] += [''] * (counts-len(dct[depth]))
        dct['{} confidents'.format(depth)] += [''] * (counts-len(dct['{} confidents'.format(depth)]))
        
    df = pd.DataFrame.from_dict(dct)
    df.to_csv(filename)
    
    return dct

#one_tree = seqs[8]
#confidences = tree_conf[8]


def build_tree(one_tree, confidences):
    # one_tree - egy lista, amely a szekvenciasorozatokat tartalmazza, amiket lépésekben generál
    # confidences - a generálási lépéshez tartozó konfidencia
    
    confidences = np.insert(confidences, 0, 1.0)
    chain_confidences = []
    tree = Tree()
    nodes = {'': [1.0, tree]}
    for idx, seq in enumerate(one_tree):
        parent = ''
        for i in range(len(seq)): 
            if not str(seq[:i+1]) in tree:
                
                nodes[str(seq[:i+1])] = []
                nodes[str(seq[:i+1])].append(confidences[idx] * nodes[parent][0]) 
                chain_confidences.append(confidences[idx] * nodes[parent][0])
                nodes[str(seq[:i+1])].append(nodes[parent][1].add_child(name=str(seq), dist=nodes[str(seq[:i+1])][0]))

            else:
                parent = str(seq[:i+1])
    
    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.show_branch_length = True
    ts.show_branch_support = True
    print(tree)
    
    return tree, chain_confidences[1:]


def rolling_window(a, size):
    shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.strides + (a. strides[-1],)
    
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def find_frequency(one_tree, decoder_input_data):

    occurances = []
    for generated_sequence in one_tree:
        
        occurance = 0
        for decoder_data in decoder_input_data:
            if np.sum(np.all(rolling_window(decoder_data,len(generated_sequence)) == generated_sequence, axis=1)):
                occurance += 1
        occurances.append(occurance / len(decoder_input_data))
        
    return occurances


def get_conf(one_seq, decoder_input_data):
    if len(one_seq) == 1:

        return 1

    else:
        supp = float(np.sum(np.all(rolling_window(decoder_input_data,len(one_seq))[:,0] == one_seq, axis=1)))
        supp_before = float(np.sum(np.all(rolling_window(decoder_input_data,len(one_seq[:-1]))[:,0] == one_seq[:-1], axis=1)))
    
        return get_conf(one_seq[:-1], decoder_input_data) * supp/supp_before
    
    

#tree_states = states[8]
#sequence_states = tree_states
"""
def plot_states(sequence_states):
    
    
    
    tree_states = np.array(tree_states[8])
    tree_states = tree_states.reshape((len(tree_states),2,256))
    h=tree_states[:,0,:]
    c=tree_states[:,1,:]
    
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    pca = PCA(n_components=2)
    h_pca = pca.fit_transform(h)
    c_pca = pca.fit_transform(c)
    
    plt.scatter(h_pca[:,0], h_pca[:,1])
    plt.scatter(c_pca[:,0], c_pca[:,1])
    plt.plot([h_pca[:,0], c_pca[:,0]],[h_pca[:,1], c_pca[:,1]],'k-', lw=0.1)
    pos = (h_pca + c_pca) / 2.
    for i, trans in enumerate(transitions):
        plt.annotate(trans, (pos[i,0],pos[i,1]))
    plt.show()
"""

def pca_activations(seqs, tree_conf, tree_seq, decoder_input_data, states):
    
    sequence_occurances = []
    for one_tree in seqs:
        sequence_occurances.append(find_frequency(one_tree))
    
    tree_objects = []
    chain_conf = []
    for one_tree, one_tree_conf in zip(seqs, tree_conf):
        tmp_tree, tmp_chain_conf = build_tree(one_tree, one_tree_conf)
        chain_conf.append(tmp_chain_conf)
        tree_objects.append(tmp_tree)
    chain_conf = np.array(chain_conf)
    occurance_array, confidence_array = [], []
    for occur, transit in zip(sequence_occurances, chain_conf):
        occurance_array += occur[1:]
        confidence_array += transit
    arr = np.array([occurance_array, confidence_array])
    import seaborn as sns
    sns.regplot(arr[0], arr[1])
    sns.regplot(arr[1], arr[0])
    print(np.corrcoef(arr[0], arr[1]))
    transitions = []
    for idx, one_tree in enumerate(tree_seq):
        try:
            transitions.append(np.array(one_tree,dtype=int)[:,:2])
        except:
            print('Tree doesn\'t contain enough node: {}.'.format(idx))
    states = [has_states for has_states in states if has_states]

    states_tmp = np.array(states[0])
    states_tmp = states_tmp.reshape((len(states_tmp),2,256))
    states_tmp = states_tmp[:,0,:]
    for state in states[1:]:
        state = np.array(state)
        state = state.reshape((len(state),2,256))
        states_tmp = np.vstack((states_tmp,state[:,0,:]))
    transitions = np.array(tree_seq[0],dtype=int)[:,:2]
    for idx, one_tree in enumerate(tree_seq[1:]):
        try:
            transitions = np.vstack((transitions,np.array(one_tree,dtype=int)[:,:2]))
        except:
            print('Tree doesn\'t contain enough node: {}.'.format(idx))
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    pca = PCA(n_components=2)
    h_pca = pca.fit_transform(states)
    
    plt.scatter(h_pca[:,0], h_pca[:,1])
    for i, trans in enumerate(transitions):
        plt.annotate(trans, (h_pca[i,0],h_pca[i,1]))
    plt.show()
    states = states_tmp
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    pca = PCA(n_components=2)
    h_pca = pca.fit_transform(states)


    transitions_list = []
    for tr in transitions:
        transitions_list.append(str(tr.tolist()))
    len(set(transitions_list))
    col_transitions = list(set(transitions_list))
    col_transitions.sort()

    col_vals = np.arange(0,1,1./len(col_transitions))
    cmap = plt.cm.gist_ncar
    cmaplist = [cmap(i) for i in col_vals]
    
    cmap_dict = dict(zip(col_transitions, cmaplist))
    for i, trans in enumerate(transitions):
        plt.scatter(h_pca[i,0],h_pca[i,1],c=cmap_dict[str(trans.tolist())])


def param_df(param_dict):
    
    results_array_1, results_array_2, labels = [], [], []
    for param, res in param_dict.items():
        results_array_1.append(res[0][0].tolist())
        results_array_2.append(res[1][0].tolist())
        labels.append(param)
    
    df_res_labs = [11 * [item] for item in labels] 
    df_res_labs = [item for sublist in df_res_labs for item in sublist]
    
    results_array_1 = [item for sublist in results_array_1 for item in sublist]
    results_array_2 = [item for sublist in results_array_2 for item in sublist]
    
    df_res = pd.DataFrame()
    df_res['vals'] = results_array_1 + results_array_2
    df_res['labels'] = df_res_labs * 2
    df_res['hue'] = ['val_1'] * len(df_res_labs) + ['val_2'] * len(df_res_labs)
    
    emb_tags = []
    s = str(df_res_labs)
    pattern = re.compile(r'embed:(\d*)')
    for m in re.finditer(pattern, s):
        emb_tags.append(m.group(1))    
    lstm_tags = []
    s = str(df_res_labs)
    pattern = re.compile(r'lstm:(\d*)')
    for m in re.finditer(pattern, s):
        lstm_tags.append(m.group(1))
        
    df_res['emb_size'] = emb_tags * 2
    df_res['lstm_size'] = lstm_tags * 2
        
    
    ovr_res_array, ovr_res_labels = [], []
    for param, res in param_dict.items():
        ovr_res_array.append(res[2][0].tolist())
        ovr_res_labels.append(param)
    
    ovr_res_array = [res for res in ovr_res_array]
    ovr_res_array = [item for sublist in ovr_res_array for item in sublist]
    ovr_res_labels = [[lab] * 2 for lab in ovr_res_labels]
    ovr_res_labels = [item for sublist in ovr_res_labels for item in sublist]
    
    ovr_res_hue = [['res_1', 'res_2'] for label in labels]
    ovr_res_hue = [item for sublist in ovr_res_hue for item in sublist]
    df_res_ovr = pd.DataFrame()
    df_res_ovr['vals'] = ovr_res_array
    df_res_ovr['labels'] = ovr_res_labels
    df_res_ovr['hue'] = ovr_res_hue
        
    lstm_tags = []
    s = str(labels)
    pattern = re.compile(r'lstm:(\d*)')
    for m in re.finditer(pattern, s):
        lstm_tags.append(m.group(1))
    emb_tags = []
    s = str(labels)
    pattern = re.compile(r'embed:(\d*)')
    for m in re.finditer(pattern, s):
        emb_tags.append(m.group(1))
    emb_tags = [[emb_tag] * 2 for emb_tag in emb_tags]
    lstm_tags = [[lstm_tag] * 2 for lstm_tag in lstm_tags]
    df_res_ovr['emb_size'] = [item for sublist in emb_tags for item in sublist]
    df_res_ovr['lstm_size'] = [item for sublist in lstm_tags for item in sublist]
    
    return df_res, df_res_ovr


def plot_params(df, lstm=None, embedding=None, figsize=None):
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    if figsize:
        fig.set_size_inches(figsize)
    if lstm or embedding:
        if lstm:
            b = sns.boxplot(x='labels', y='vals', hue='hue', data=df[df['lstm_size'] == str(lstm)])
        elif embedding:
            b = sns.boxplot(x='labels', y='vals', hue='hue', data=df[df['emb_size'] == str(embedding)])
        else:
            print('Conditioned select!!')
    else:
        b = sns.boxplot(x='labels', y='vals', hue='hue', data=df) 
    plt.xticks(rotation=90)
    plt.show()
    
    

print('finished..')