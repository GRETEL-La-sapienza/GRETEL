import sys
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import networkx as nx
from src.dataset.dataset import Dataset

from src.dataset.instances.graph import GraphInstance
from src.dataset.generators.base import Generator
 


class GithubStargazersGenerator(Generator):
    
    def init(self):
        print("Inizio inizializzazione di GithubStargazersGenerator.")

        base_path = self.local_config['parameters']['data_dir']
        self.adj_matrix_filename = join(base_path, self.local_config['parameters']['adj_matrix_filename'])
        self.graph_indicator_filename = join(base_path, self.local_config['parameters']['graph_indicator_filename'])
        self.graph_labels_filename = join(base_path, self.local_config['parameters']['graph_labels_filename'])
        self.num_adj_instances = self.local_config['parameters']['num_adj_instances']
        self.num_graphs = self.local_config['parameters']['num_graphs']
        #self.max_number_nodes = self.local_config['parameters']['max_number_nodes']
  

        self.generate_dataset()


    def check_configuration(self):
        super().check_configuration()
        local_config = self.local_config
        # set defaults
        local_config['parameters']['data_dir'] = local_config['parameters'].get('data_dir','github_stargazers')
        local_config['parameters']['adj_matrix_filename']= local_config['parameters'].get('adj_matrix_filename','github_stargazers_A.txt')
        local_config['parameters']['graph_indicator_filename'] = local_config['parameters'].get('graph_indicator_filename','github_stargazers_graph_indicator.txt')
        local_config['parameters']['graph_labels_filename'] = local_config['parameters'].get('graph_labels_filename','github_stargazers_graph_labels.txt')
        local_config['parameters']['num_adj_instances']= local_config['parameters'].get('num_adj_instances',5971562)
        local_config['parameters']['num_graphs']= local_config['parameters'].get('num_graphs',12725)
        #local_config['parameters']['max_number_nodes'] = local_config['parameters'].get('max_number_nodes', 10)

     
    def generate_dataset(self):
        if not len(self.dataset.instances):
            # Carica la lista degli archi (coppie di nodi) 
            total_graphs_added = 0  # Inizializza un contatore per i grafi aggiunti
            # Costruisci il percorso al file desiderato per adj_matrix_filename
            project_root_dir = '/Users/bruzzese/GRETEL/'
            adj_matrix_filename = os.path.join(project_root_dir, 'data', 'datasets', 'github_stargazers', 'github_stargazers_A.txt')           
            edges = np.loadtxt(adj_matrix_filename, delimiter=',',dtype=int)
            # Carica gli indicatori dei grafi
            # ogni nodo appartiene ad uno dei 12725 grafi
            graph_indicator_filename = os.path.join(project_root_dir, 'data', 'datasets', 'github_stargazers', 'github_stargazers_graph_indicator.txt')
            graph_indicator = np.loadtxt(graph_indicator_filename, dtype=int)
            # Carica le etichette dei grafi
            #sono 12725 righe. un etichetta per ciascun grafo
            graph_labels_filename = os.path.join(project_root_dir, 'data', 'datasets', 'github_stargazers', 'github_stargazers_graph_labels.txt')
            graph_labels = np.loadtxt(graph_labels_filename, dtype=int)
            for graph_id in range(1, self.num_graphs+1):
                # Filtra gli archi per il grafo corrente
                graph_nodes = np.where(graph_indicator == graph_id)[0] + 1 
               # if len(graph_nodes) >= self.max_number_nodes: 
                print(f"lunghezza nodi grafo {len(graph_nodes)}")
                #limita i grafi alla lunghezza massima di max_number_nodes: * attualmente commentato*
                #if len(graph_nodes) > self.max_number_nodes:
                #     continue # skiping the biggest graphs to optimize needed resources
                # Stampa la lista dei nodi per il grafo corrente
                print(f"Grafo {graph_id}: Lista nodi {list(graph_nodes)}")
                graph_edges = edges[np.isin(edges, graph_nodes).any(axis=1)]
                graph_edges += 1
                # Stampa gli archi per il grafo corrente
                print(f"Grafo {graph_id}: Archi {graph_edges}")
                # Crea il grafo NetworkX
                G = nx.Graph()
                G.add_edges_from(graph_edges)   
                features = generate_node_features(G)
                # Esempio di iterazione sui nodi e assegnazione delle features:
                for idx, node in enumerate(G.nodes()):
                    # Sottrai 1 da idx se i tuoi indici iniziano da 1
                    G.nodes[node]['degree_centrality'] = features[idx][0]
                    G.nodes[node]['betweenness_centrality'] = features[idx][1]
                    G.nodes[node]['clustering_index'] = features[idx][2]
                    G.nodes[node]['eigenvector_centrality'] = features[idx][3]
                    G.nodes[node]['closeness_centrality'] = features[idx][4]

                graph = nx.to_numpy_array(G)
                # Aggiungi il grafo all'elenco delle istanze
                label = graph_labels[graph_id - 1]
                # Aggiungi il grafo come istanza GraphInstance al dataset
                self.dataset.instances.append(GraphInstance(id=graph_id, data=graph, node_features=features, label=label)) 
                # Incrementa il contatore dei grafi aggiunti
                total_graphs_added += 1
                # Stampa l'ID del grafo e il numero di nodi
                print(f"Grafo ID: {graph_id}, Numero di nodi: {len(graph_nodes)}")
                # Controlla se sono stati aggiunti 32 grafi e interrompe il ciclo se vero: *attualmente commentato*
                #if total_graphs_added == 32:
                #  print("Raggiunto il limite di 32 grafi aggiunti.")
                #  break
                # Stampa il numero totale di grafi aggiunti alla fine
            print(f"Numero totale di grafi aggiunti: {total_graphs_added}")
            print("Fine generazione dataset in GithubStargazersGenerator.")


    def get_num_instances(self):
        # Restituisce il numero di istanze di grafi generate
        return len(self.dataset.instances)


def generate_node_features(graph: nx.Graph):
    # Calcola le metriche di centralità
    degree_centrality = nx.degree_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    clustering_index = nx.clustering(graph)
    eigenvector_centrality = nx.eigenvector_centrality(graph)
    closeness_centrality = nx.closeness_centrality(graph)
    num_nodes = graph.number_of_nodes()

    # Inizializza le feature con zero
    features = np.zeros((num_nodes, 5), dtype=np.float32)
  
    # Mappa gli indici dei nodi a un intervallo che inizia da 0
    node_idx_map = {node: idx  for idx, node in enumerate(graph.nodes())}


    for node, idx in node_idx_map.items():
        features[idx][0] = degree_centrality[node]
        features[idx][1] = betweenness_centrality[node]
        features[idx][2] = clustering_index[node]
        features[idx][3] = eigenvector_centrality[node]
        features[idx][4] = closeness_centrality[node]

    return features
