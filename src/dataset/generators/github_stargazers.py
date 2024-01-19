from os import listdir
from os.path import isfile, join
import numpy as np
import networkx as nx
from src.dataset.instances.graph import GraphInstance
from src.dataset.generators.base import Generator

class GithubStargazersGenerator(Generator):
    
    def init(self):
        base_path = self.local_config['parameters']['data_dir']
        self.adj_matrix_filename = join(base_path, self.local_config['parameters']['adj_matrix_filename'])
        self.graph_indicator_filename = join(base_path, self.local_config['parameters']['graph_indicator_filename'])
        self.graph_labels_filename = join(base_path, self.local_config['parameters']['graph_labels_filename'])
        self.num_adj_instances = self.local_config['parameters']['num_adj_instances']
        self.num_graphs = self.local_config['parameters']['num_graphs']
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

    def generate_dataset(self):
        if not len(self.dataset.instances):
            # Carica la lista degli archi (coppie di nodi)
            #sono collegamenti tra nodi 5971562 righe
            edges = np.loadtxt(self.adj_matrix_filename, delimiter=',',dtype=int)

            # Carica gli indicatori dei grafi
            # sono 1448038 righe , una per ogni nodo;
            # ogni nodo appartiene ad uno dei 12725 grafi
            graph_indicator = np.loadtxt(self.graph_indicator_filename, dtype=int)

            # Carica le etichette dei grafi
            #sono 12725 righe. un etichetta per ciascun grafo
            graph_labels = np.loadtxt(self.graph_labels_filename, dtype=int)

            for graph_id in range(1, self.num_graphs ):
                # Filtra gli archi per il grafo corrente
                graph_nodes = np.where(graph_indicator == graph_id)[0]
                graph_edges = edges[np.isin(edges, graph_nodes).any(axis=1)]

                # Crea il grafo NetworkX
                G = nx.Graph()
                G.add_edges_from(graph_edges - 1)  # Sottrai 1 se gli indici dei nodi partono da 1
                graph = nx.to_numpy_array(G)

                # Aggiungi il grafo all'elenco delle istanze
                label = graph_labels[graph_id - 1]
                self.dataset.instances.append(GraphInstance(id=graph_id, data=graph, label=label))    
    

            # Logging o altre operazioni


    def get_num_instances(self):
        # Restituisce il numero di istanze di grafi generate
        return len(self.dataset.instances)