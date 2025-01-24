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
import argparse
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import adjusted_rand_score
import json

# This is a hack to avoid an execution error depending on the version of networkx. You can simply ignore but not remove it
if not hasattr(nx, "to_scipy_sparse_matrix"):
    def to_scipy_sparse_matrix(G, dtype='f', format='lil'):
        return nx.to_scipy_sparse_array(G)
nx.to_scipy_sparse_matrix = to_scipy_sparse_matrix

def save_graph_data(G, output_path, filename_prefix):
    # Save nodes
    nodes_filename = f"{output_path}/{filename_prefix}_nodes.csv"
    pd.DataFrame(list(G.nodes()), columns=['Node']).to_csv(nodes_filename, index=False)
    print(f"Nodes saved to {nodes_filename}")

    # Save edges with weights
    edges_filename = f"{output_path}/{filename_prefix}_edges.csv"
    edges_data = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
    pd.DataFrame(edges_data, columns=['Source', 'Target', 'Weight']).to_csv(edges_filename, index=False)
    print(f"Edges saved to {edges_filename}")


def visualize_graph(G, output_path, filename, topic_labels, positions, dpi=300):
    # Detect communities using the Louvain method
    communities = nx_comm.louvain_communities(G, seed=42)
    modularity = nx_comm.modularity(G, communities)

    # Number of communities
    nc = len(communities)
    print(f"Number of communities: {nc}")
    print(f"Modularity: {modularity}")

    # Get a palette of colors for communities
    palette = sns.color_palette("hsv", n_colors=nc)
    node2comm = {node: i for i, com in enumerate(communities) for node in com}

    # Create a list of node colors
    node_colors = [palette[node2comm[node]] for node in G.nodes()]
    degrees = [G.degree(node) for node in G.nodes()]
    
    # Label each community with the most frequent topic
    community_labels = []
    for i, com in enumerate(communities):
        topic_counter = Counter([topic_labels[node] for node in com])
        most_common_topic = topic_counter.most_common(1)[0][0]
        community_labels.append(most_common_topic)

    plt.figure(figsize=(14, 14), dpi=dpi)
    nx.draw_networkx_nodes(G, positions, node_size=[v * 10 for v in degrees], node_color=node_colors, alpha=0.85)
    nx.draw_networkx_edges(G, positions, alpha=0.6, width=0.5)
    nx.draw_networkx_labels(
        G,
        positions,
        labels={node: label for node, label in enumerate(community_labels)},
        font_size=12,
        font_color='darkblue',
        font_weight='bold',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
    )
    plt.axis('off')
    plt.margins(0.15)
    plt.savefig(f"{output_path}/{filename}", dpi=dpi, bbox_inches='tight', pad_inches=0.5)
    plt.show()

def process_and_visualize(data_path, output_path, language, n_docs=200, n_edges_per_node=10, gravity=50, random_state=0, dpi=300, use_sample=True):
    # Load dataset
    df = pd.read_parquet(data_path)
    corpus_size = len(df)
    print(f"-- -- {language.capitalize()} dataset contains {corpus_size} documents")
    print(f"LABELS: {set(df['label'])}")

    if use_sample:
        # Take a sample of documents
        sample_factor = n_docs / corpus_size
        df_sample = df.sample(n_docs, random_state=random_state)
        print(f"Dataset reduced to {n_docs} documents")
    else:
        df_sample = df
        sample_factor = 1
        n_docs = corpus_size

    print("-- - Sample documents:")
    print(df_sample.head())

    X = [literal_eval(el) for el in df_sample['thetas'].values.tolist()]
    X = corpus2csc(X).T
    n_topics = X.shape[1]
    print(f"-- -- Number of topics: {n_topics}")
    print(f"X: sparse matrix with {X.nnz} nonzero values out of {n_docs * n_topics}")
    print(X.shape)

    # Normalization
    X = scsp.csr_matrix(X / np.sum(X, axis=1))
    print(f"-- -- Average row sum: {np.mean(X.sum(axis=1).T)}")

    S = np.sqrt(X) * np.sqrt(X.T)
    print("-- -- Number of non-zero components: ", S.nnz)
    nnz_prop = S.nnz / (S.shape[0] * S.shape[1])
    print("-- -- Proportion of non-zero components: ", nnz_prop)
    print("-- -- Proportion of zeros: ", 1 - nnz_prop)

    # Matrix is symmetric. Keep only upper triangular part
    S = scsp.triu(S, k=1)
    print('-- -- Number of non-zero components in S:', S.nnz)

    n_nodes = S.shape[0]
    n_edges = S.nnz
    n_edges_per_node = n_edges / n_nodes
    print(f"-- -- Number of nodes: {n_nodes}")
    print(f"-- -- Number of edges: {n_edges}")
    print(f"-- -- Number of edges per node: {n_edges_per_node}")

    # Set average number of edges per node
    n_edges_per_node = 10
    n_edges = n_nodes * n_edges_per_node
    sorted_similarity_values = np.sort(S.data)[::-1]
    thr = sorted_similarity_values[n_edges]

    # Apply the threshold to similarity matrix
    S.data[S.data < thr] = 0
    S.eliminate_zeros()
    print(f"-- -- Threshold: {thr}")
    print(f"-- -- Number of edges: {n_edges}")
    print('-- -- Estimated number of links in full corpus:', len(S.data) / 2 / sample_factor**2)

    # Transform graph to networkx format
    G = nx.from_scipy_sparse_array(S)

    # Filter edges by weight
    edges_to_remove = [(u, v) for u, v, w in G.edges(data=True) if w['weight'] < thr]
    G.remove_edges_from(edges_to_remove)
    
    # Save the graph data
    save_graph_data(G, output_path, f"graph_{language}")

    # Largest connected component (LCC) from the graph
    nodes = list(nx.connected_components(G))
    lcc = max(nodes, key=len)
    G_lcc = G.subgraph(lcc)
    positions_lcc = nx.spring_layout(G_lcc, seed=random_state)

    plt.figure(figsize=(10, 10), dpi=dpi)
    nx.draw(G_lcc, pos=positions_lcc, node_size=50, width=0.1)
    plt.savefig(f"{output_path}/draw_graph_lcc_{language}.png", dpi=dpi)

    # Compute positions using layout algorithm
    forceatlas2 = ForceAtlas2(gravity=gravity)
    positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=200)
    G = G.subgraph(list(G.nodes()))
    valid_positions = {k: positions[k] for k in list(positions)}

    plt.figure(figsize=(10, 10), dpi=dpi)
    nx.draw_networkx_nodes(G, valid_positions, node_size=50, node_color="blue", alpha=0.7)
    nx.draw_networkx_edges(G, valid_positions, edge_color="green", alpha=0.1)
    plt.axis('off')
    plt.savefig(f"{output_path}/draw_graph_forceatlas2_{language}.png", dpi=dpi)

    topic_labels = df_sample['label'].tolist()
    visualize_graph(G_lcc, output_path, f"draw_graph_communities_{language}.png", topic_labels, positions_lcc, dpi)

    return df_sample, G, X, positions_lcc, topic_labels

def find_closest_documents(X1, X2):
    similarities = []
    for i, theta1 in enumerate(X1):
        similarities.append([])
        for j, theta2 in enumerate(X2):
            sim = 1 - jensenshannon(theta1.toarray().ravel(), theta2.toarray().ravel())
            similarities[-1].append((j, sim))
    return similarities

def save_closest_docs_info(df1, df2, closest_docs, filename):
    info = []
    for i, doc_similarities in enumerate(closest_docs):
        doc_info = {"document": i, "closest_documents": []}
        sorted_docs = sorted(doc_similarities, key=lambda x: -x[1])[:5]
        for j, sim in sorted_docs:
            doc_info["closest_documents"].append({"document": j, "similarity": sim})
        info.append(doc_info)
    with open(filename, 'w') as f:
        json.dump(info, f, indent=4)

def color_nodes_based_on_closest_documents(df1, G1, df2, G2, closest_docs, positions_G1, positions_G2, filename, dpi=300, info_filename=None):
    # Detect communities in the second graph (G2)
    communities2 = nx_comm.louvain_communities(G2, seed=42)
    node2comm2 = {node: i for i, com in enumerate(communities2) for node in com}
    n_communities2 = len(communities2)

    # Color nodes in the second graph (G2) based on its communities
    palette = sns.color_palette("hsv", n_colors=n_communities2)
    node_colors_G2 = [palette[node2comm2[node]] for node in G2.nodes()]
    
    # Color nodes in the first graph (G1) based on the closest documents in the second graph's communities
    closest_community_colors_G1 = []
    for i in range(len(df1)):
        closest_docs_i = sorted(closest_docs[i], key=lambda x: -x[1])[:5]
        closest_communities = [node2comm2[closest_doc[0]] for closest_doc in closest_docs_i if closest_doc[1] > 0]
        if closest_communities:
            common_community = Counter(closest_communities).most_common(1)[0][0]
            closest_community_colors_G1.append(palette[common_community])
        else:
            closest_community_colors_G1.append((0.5, 0.5, 0.5))  # Gray color for unmatched nodes

    if info_filename:
        save_closest_docs_info(df1, df2, closest_docs, info_filename)

    degrees_G2 = [G2.degree(node) for node in G2.nodes()]
    degrees_G1 = [G1.degree(node) for node in G1.nodes()]

    plt.figure(figsize=(20, 10), dpi=dpi)
    
    # Plot G2 with its communities
    plt.subplot(121)
    nx.draw_networkx_nodes(G2, positions_G2, node_size=[v * 10 for v in degrees_G2], node_color=node_colors_G2, alpha=0.85)
    nx.draw_networkx_edges(G2, positions_G2, alpha=0.6, width=0.5)
    plt.title("Graph 2 with its Communities")
    plt.axis('off')

    # Plot G1 colored by G2's communities
    plt.subplot(122)
    nx.draw_networkx_nodes(G1, positions_G1, node_size=[v * 10 for v in degrees_G1], node_color=closest_community_colors_G1, alpha=0.85)
    nx.draw_networkx_edges(G1, positions_G1, alpha=0.6, width=0.5)
    plt.title("Graph 1 Colored by Graph 2's Communities")
    plt.axis('off')

    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.show()
    
def compare_communities(G1, G2):
    comms1 = list(nx_comm.louvain_communities(G1, seed=42))
    comms2 = list(nx_comm.louvain_communities(G2, seed=42))
    node2comm1 = {node: i for i, comm in enumerate(comms1) for node in comm}
    node2comm2 = {node: i for i, comm in enumerate(comms2) for node in comm}
    labels1 = [node2comm1.get(node, -1) for node in G1.nodes()]
    labels2 = [node2comm2.get(node, -1) for node in G2.nodes()]
    return adjusted_rand_score(labels1, labels2)

def analyze_cluster_composition(G1, G2):
    comms1 = list(nx_comm.louvain_communities(G1, seed=42))
    comms2 = list(nx_comm.louvain_communities(G2, seed=42))

    # Count nodes in each community
    sizes1 = [len(comm) for comm in comms1]
    sizes2 = [len(comm) for comm in comms2]

    print(f"Cluster sizes in Graph 1: {sizes1}")
    print(f"Cluster sizes in Graph 2: {sizes2}")
    
    return sizes1, sizes2

def compute_jaccard_similarity(comms1, comms2):
    # Convert to sets
    comms1_sets = [set(comm) for comm in comms1]
    comms2_sets = [set(comm) for comm in comms2]

    # Compute Jaccard similarity
    similarities = []
    for comm1 in comms1_sets:
        for comm2 in comms2_sets:
            intersection = len(comm1 & comm2)
            union = len(comm1 | comm2)
            jaccard = intersection / union if union > 0 else 0
            similarities.append(jaccard)
    return np.mean(similarities)

def compare_clusters_jaccard(G1, G2):
    comms1 = list(nx_comm.louvain_communities(G1, seed=42))
    comms2 = list(nx_comm.louvain_communities(G2, seed=42))
    jaccard_score = compute_jaccard_similarity(comms1, comms2)
    return jaccard_score

def compute_topic_similarity(X1, X2, output_path):
    # Compute similarity matrix
    similarity_matrix = np.sqrt(X1.toarray()) @ np.sqrt(X2.toarray().T)

    # Save similarity matrix
    similarity_filename = f"{output_path}/topic_similarity_matrix.csv"
    pd.DataFrame(similarity_matrix).to_csv(similarity_filename, index=False)
    print(f"Topic similarity matrix saved to {similarity_filename}")

    # Visualize as heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap="coolwarm", annot=False)
    plt.title("Topic Similarity Matrix")
    plt.xlabel("Spanish Topics")
    plt.ylabel("English Topics")
    plt.savefig(f"{output_path}/topic_similarity_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Topic similarity heatmap saved.")

def main(
    data_path_spanish,
    data_path_english,
    output_path,
    n_docs=200,
    n_edges_per_node=10,
    gravity=50,
    random_state=0,
    dpi=300,
    use_sample=True
):
    df_spanish, G_spanish, X_spanish, positions_spanish, labels = process_and_visualize(data_path_spanish, output_path, "spanish", n_docs, n_edges_per_node, gravity, random_state, dpi, use_sample)
    df_english, G_english, X_english, positions_english, labels = process_and_visualize(data_path_english, output_path, "english", n_docs, n_edges_per_node, gravity, random_state, dpi, use_sample)

    # Find closest documents between English and Spanish
    closest_docs_en_to_es = find_closest_documents(X_english, X_spanish)
    closest_docs_es_to_en = find_closest_documents(X_spanish, X_english)

    # Visualize English communities on Spanish graph
    color_nodes_based_on_closest_documents(df_spanish, G_spanish, df_english, G_english, closest_docs_en_to_es, positions_spanish, positions_english, f"{output_path}/spanish_colored_by_english.png", dpi, f"{output_path}/closest_docs_en_to_es.json")

    # Visualize Spanish communities on English graph
    color_nodes_based_on_closest_documents(df_english, G_english, df_spanish, G_spanish, closest_docs_es_to_en, positions_english, positions_spanish, f"{output_path}/english_colored_by_spanish.png", dpi, f"{output_path}/closest_docs_es_to_en.json")

    # Compare communities
    community_similarity = compare_communities(G_spanish, G_english)
    print(f"Adjusted Rand Index for community comparison: {community_similarity}")
    
    # Compute Jaccard similarity between clusters
    print("Computing Jaccard similarity between clusters")
    jaccard_score = compare_clusters_jaccard(G_spanish, G_english)
    print(f"Jaccard similarity between clusters: {jaccard_score}")

    # Analyze cluster composition
    print("Analyzing cluster composition")
    analyze_cluster_composition(G_spanish, G_english)
    print("Computing heatmap of topic similarity")
    compute_topic_similarity(X_english, X_spanish, output_path)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process and visualize network data.")
    parser.add_argument(
        '--data_path_english',
        type=str,
        required=False, 
        help="Path to the data file",
        default="/data/source/poly_rosie_1_20/df_graph_en.parquet"
    )
    parser.add_argument(
        '--data_path_spanish',
        type=str,
        required=False, 
        help="Path to the data file",
        default="/data/source/poly_rosie_1_20/df_graph_es.parquet"
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=False,
        help="Path to save output files",
        default="/data/source/poly_rosie_1_20"
    )
    parser.add_argument(
        '--n_docs',
        type=int,
        default=200,
        help="Number of documents to sample")
    parser.add_argument(
        '--n_edges_per_node',
        type=int,
        default=10,
        help="Average number of edges per node"
    )
    parser.add_argument(
        '--gravity',
        type=int,
        default=50,
        help="Gravity parameter for ForceAtlas2"
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=0,
        help="Random state for reproducibility"
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help="DPI for saving figures"
    )

    args = parser.parse_args()
    main(args.data_path_spanish, args.data_path_english, args.output_path, args.n_docs, args.n_edges_per_node, args.gravity, args.random_state, args.dpi)