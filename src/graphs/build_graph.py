import sys
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as scsp
import seaborn as sns
from collections import Counter
from fa2 import ForceAtlas2
from gensim.matutils import corpus2csc
from ast import literal_eval
import networkx.algorithms.community as nx_comm


path2data = '/data/source/NSF_df_35topics.xlsx'
NSF_df_all = pd.read_excel(path2data)
corpus_size = len(NSF_df_all)
print(f"Original dataset contains {corpus_size} documents")
NSF_df_all.head()

print(set(NSF_df_all['main_topic_35']))

# Remove garbage topics
NSF_df_all = NSF_df_all[NSF_df_all['main_topic_35']!='Garbage topic']
corpus_size = len(NSF_df_all)
print(f"Clean dataset contains {corpus_size} documents")

# Take a sample of documents
n_docs = 1000
print(f"{n_docs} documents")

# Take a sample of documents
sample_factor = n_docs / corpus_size
NSF_df = NSF_df_all.sample(n_docs, random_state=0)
print(f"Dataset reduced to {n_docs} documents")
print("Sample documents:")
NSF_df.head()


X = [literal_eval(el) for el in NSF_df['LDA_35'].values.tolist()]
X = corpus2csc(X).T
n_topics = X.shape[1]
print(f"Number of topics: {n_topics}")
print(f"X: sparse matrix with {X.nnz} nonzero values out of {n_docs *n_topics}")
print(X.shape)

# Normalization:
X = scsp.csr_matrix(X / np.sum(X, axis=1))
print(f"Average row sum: {np.mean(X.sum(axis=1).T)}")

S = np.sqrt(X) * np.sqrt(X.T)
print("Number of non-zero components: ", S.nnz)
nnz_prop = S.nnz / (S.shape[0] * S.shape[1])
print("Proportion of non-zero components: ", nnz_prop)
print("Proportion of zeros: ", 1 - nnz_prop)

# Matrix is symmetric. Keep only upper triangular part
S = scsp.triu(S, k=1)

print('Number of non-zero components in S:', S.nnz)

n_nodes = S.shape[0]
n_edges = S.nnz
n_edges_per_node = n_edges / n_nodes

print(f"Number of nodes: {n_nodes}")
print(f"Number of edges: {n_edges}")
print(f"Number of edges per node: {n_edges_per_node}")

# Set average number of edges per node
n_edges_per_node = 10

# Compute threshold to get the target number of edges
# <SOL>
n_edges = n_nodes * n_edges_per_node
sorted_similarity_values = np.sort(S.data)[::-1]
thr = sorted_similarity_values[n_edges]
# </SOL>

# Apply the threshold to similarity matrix
# <SOL>
S.data[S.data < thr] = 0
S.eliminate_zeros()
# </SOL>

print(f"Threshold: {thr}")
print(f"Number of edges: {n_edges}")
print('Estimated number of links in full corpus:', len(S.data)/2/sample_factor**2)


# Transform graph to networkx format
G = nx.from_scipy_sparse_array(S)

# largest connected component  (LCC) from the graph
nodes = list(nx.connected_components(G))
lcc = max(nodes, key=len)
G_lcc = G.subgraph(lcc)
seed_value = 0
positions_lcc = nx.spring_layout(G_lcc, seed=seed_value)

plt.figure(figsize=(3, 3))
nx.draw(G_lcc, pos=positions_lcc, node_size=1, width=0.02)
file_path = '/data/source/draw_graph_lcc.png'
plt.savefig(file_path)


# Compute positions using layout algorithm
gravity = 50

# Create layout object
forceatlas2 = ForceAtlas2(
    gravity=gravity
)

# This is a hack to avoid an execution error depending on the version of networkx. You can
# simply ignore but not remove it
if not hasattr(nx, "to_scipy_sparse_matrix"):
    def to_scipy_sparse_matrix(G, dtype='f', format='lil'):
        return nx.to_scipy_sparse_array(G)
nx.to_scipy_sparse_matrix = to_scipy_sparse_matrix


positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=200)
print(positions)
print(len(positions))
print(positions.keys())
print(list(positions))

G = G.subgraph(list(G.nodes))

valid_positions = {k: positions[k] for k in list(positions)}
#valid_positions = {k: positions[k] for k in positions.keys()}


nx.draw_networkx_nodes(G, valid_positions, node_size=20, node_color="blue", alpha=0.4)
nx.draw_networkx_edges(G, valid_positions, edge_color="green", alpha=0.05)
plt.axis('off')
plt.show()

C = nx_comm.louvain_communities(G_lcc, seed=0)

# Modularity of the partition
modularity = nx_comm.modularity(G_lcc, C)

nc = len(C)
print(f"Number of communities: {nc}")
print(f"Modularity: {modularity}")

# Get a palette of rgb colors as large as the list of unique attributes
palette = sns.color_palette(palette="Paired", n_colors=nc)

# Mapping communities in nx format into an ordered list of communities, one per node.
node2comm = {n: 0 for n in G_lcc}
for i, com in enumerate(C):
    for node in list(com):
        node2comm[node] = i
        
# Map node attribute to rgb colors
node_colors = [palette[node2comm[n]] for n in G_lcc]

# Get list of degrees
degrees = [val / 3 for (node, val) in G_lcc.degree()]

#  Draw graph
plt.figure(figsize=(5, 5))
nx.draw(G_lcc, positions, node_size=degrees, node_color=node_colors, width=0.1)

file_path = '/data/source/draw_graph.png'

# Save the figure as an image
plt.savefig(file_path)

print(f"Code saved to '{file_path}' successfully.")