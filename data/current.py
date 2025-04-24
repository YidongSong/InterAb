import pandas as pd
import pickle


example_specificity = pd.read_csv('/mnt/ai4x_ceph/fandiwu/buddy1/yidongsong/Ab_Ag_affinity/github/data/example_affinity.csv')
example_affinity = pd.read_csv('/mnt/ai4x_ceph/fandiwu/buddy1/yidongsong/Ab_Ag_affinity/github/data/example_specificity.csv')
# affinity_atom_inputs = pickle.load(open('/mnt/ai4x_ceph/fandiwu/buddy1/yidongsong/Ab_Ag_affinity/github/data/atom/affinity_atom_inputs.pkl', 'rb'))
# spe_atom_inputs = pickle.load(open('/mnt/ai4x_ceph/fandiwu/buddy1/yidongsong/Ab_Ag_affinity/github/data/atom/atom_inputs_native_chai.pkl', 'rb'))
# affinity_geo_data = pickle.load(open('/mnt/ai4x_ceph/fandiwu/buddy1/yidongsong/Ab_Ag_affinity/github/data/Geo_data/affinity_geo_data.pkl', 'rb'))
affinity_geo_data = pickle.load(open('/mnt/ai4x_ceph/fandiwu/buddy1/yidongsong/Ab_Ag_affinity/github/data/Geo_data/specificity_geo_data.pkl', 'rb'))
# exa_specificity_atom_inputs = {}
exa_affinity_geo_data = {}
exa_affinity_geo_data['ab_h_X'] = {}
exa_affinity_geo_data['ab_l_X'] = {}
exa_affinity_geo_data['ag_X'] = {}
exa_affinity_geo_data['ab_h_node_feat'] = {}
exa_affinity_geo_data['ab_l_node_feat'] = {}
exa_affinity_geo_data['ag_node_feat'] = {}
exa_affinity_geo_data['ab_h_edge_index'] = {}
exa_affinity_geo_data['ab_l_edge_index'] = {}
exa_affinity_geo_data['ag_edge_index'] = {}
exa_affinity_geo_data['ab_h_seq'] = {}
exa_affinity_geo_data['ab_l_seq'] = {}
exa_affinity_geo_data['ag_seq'] = {}

for i in range(len(example_affinity)):
    exa_affinity_geo_data['ab_h_X'][example_affinity['pdb'][i]] = affinity_geo_data['ab_h_X'][example_affinity['pdb'][i]]
    exa_affinity_geo_data['ab_l_X'][example_affinity['pdb'][i]] = affinity_geo_data['ab_l_X'][example_affinity['pdb'][i]]
    exa_affinity_geo_data['ag_X'][example_affinity['pdb'][i]] = affinity_geo_data['ag_X'][example_affinity['pdb'][i]]
    exa_affinity_geo_data['ab_h_node_feat'][example_affinity['pdb'][i]] = affinity_geo_data['ab_h_node_feat'][example_affinity['pdb'][i]]
    exa_affinity_geo_data['ab_l_node_feat'][example_affinity['pdb'][i]] = affinity_geo_data['ab_l_node_feat'][example_affinity['pdb'][i]]
    exa_affinity_geo_data['ag_node_feat'][example_affinity['pdb'][i]] = affinity_geo_data['ag_node_feat'][example_affinity['pdb'][i]]
    exa_affinity_geo_data['ab_h_edge_index'][example_affinity['pdb'][i]] = affinity_geo_data['ab_h_edge_index'][example_affinity['pdb'][i]]
    exa_affinity_geo_data['ab_l_edge_index'][example_affinity['pdb'][i]] = affinity_geo_data['ab_l_edge_index'][example_affinity['pdb'][i]]
    exa_affinity_geo_data['ag_edge_index'][example_affinity['pdb'][i]] = affinity_geo_data['ag_edge_index'][example_affinity['pdb'][i]]
    exa_affinity_geo_data['ab_h_seq'][example_affinity['pdb'][i]] = affinity_geo_data['ab_h_seq'][example_affinity['pdb'][i]]
    exa_affinity_geo_data['ab_l_seq'][example_affinity['pdb'][i]] = affinity_geo_data['ab_l_seq'][example_affinity['pdb'][i]]
    exa_affinity_geo_data['ag_seq'][example_affinity['pdb'][i]] = affinity_geo_data['ag_seq'][example_affinity['pdb'][i]]

# pickle.dump(exa_affinity_geo_data, open('/mnt/ai4x_ceph/fandiwu/buddy1/yidongsong/Ab_Ag_affinity/github/data/Geo_data/exa_specificity_geo_data.pkl', 'wb'))