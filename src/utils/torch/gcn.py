'''import torch.nn as nn
from torch_geometric.nn.aggr import MeanAggregation,SoftmaxAggregation
from torch_geometric.nn.conv import GCNConv
#import torch
from src.core.factory_base import build_w_params_string

class GCN(nn.Module):
   
    def __init__(self, node_features, num_conv_layers=2, conv_booster=1, pooling=MeanAggregation):
        super(GCN, self).__init__()

        #num_node_features = 5 #modificato aggiunto
        print(f"Parametro node_features passato: {node_features}")
        #self.in_channels = num_node_features
        self.in_channels = node_features
        self.out_channels = int(self.in_channels * conv_booster)
         # modifica aggiunto Stampa i valori calcolati di in_channels e out_channels
        #print(f"in_channels: {self.in_channels}, out_channels: {self.out_channels}")
        self.pooling =  build_w_params_string(pooling)
 

        #modificato aggiunto  Imposta self.num_conv_layers per tutti i casi
        #self.num_conv_layers = [(self.in_channels, self.out_channels)]
        #self.num_conv_layers += [(self.out_channels, self.out_channels) for _ in range(num_conv_layers - 1)]
        if num_conv_layers>1:
            self.num_conv_layers = [(self.in_channels, self.out_channels)] + [(self.out_channels, self.out_channels) * (num_conv_layers - 1)]
        else:
            self.num_conv_layers = [(self.in_channels, self.out_channels)]
        #if num_conv_layers>1:
        #    self.num_conv_layers = [(self.in_channels, self.out_channels)] + [(self.out_channels, self.out_channels) for _ in range(num_conv_layers - 1)]
        #else:
        #    self.num_conv_layers = [(self.in_channels, self.out_channels)]
        self.graph_convs = self.__init__conv_layers()
        


    def forward(self, node_features, edge_index, edge_weight, batch):
        # convolution operations
        for conv_layer in self.graph_convs[:-1]:
            node_features = node_features.float()
            #node_features = torch.tensor(node_features).float()

            node_features = conv_layer(node_features, edge_index, edge_weight)
            node_features = node_features.float() #aggiunto per il problema di double
            node_features = nn.functional.relu(node_features)

        #node_features = node_features.float()
        
        #self.graph_convs = self.graph_convs[:-1]  # Assumi che vogliamo solo i primi due layer per il ciclo
        #print(f"Layer convoluzionali attuali: {len(self.graph_convs)}")
        #print(f"Numero totale di layer convoluzionali: {len(self.graph_convs)}")
        #print(f"Numero di layer processati nel ciclo: {len(self.graph_convs[:2])}")
        #for conv_layer in self.graph_convs[:-1]:
        #for conv_layer in self.graph_convs:
        #    node_features = node_features.float()
            # Applica il layer convoluzionale
        #    print(f"node_features shape: {node_features.shape}")
        #    node_features = conv_layer(node_features, edge_index, edge_weight)
        #    node_features = node_features.float()
            # Applica la funzione di attivazione ReLU
        #    node_features = nn.functional.relu(node_features)

        # global pooling
        #if isinstance(self.graph_convs[-1],nn.Identity):
        #    return self.graph_convs[-1](node_features)
        #return self.graph_convs[-1](node_features, batch)
        # Gestione del layer finale
        #final_layer = self.graph_convs[-1]  # Assicurati che questo sia il layer corretto dopo le modifiche
        final_layer = self.graph_convs[-1]  # Questo ora è il secondo layer, dato che self.graph_convs è stato limitato a due layer
        # Stampa le informazioni sul final_layer per il debugging
        #print("Il final_layer è di tipo:", type(final_layer).__name__)
        #print("Dettagli del final_layer:", final_layer)
        # Controlla se il layer finale è di tipo nn.Identity
        print("Numero di layer in self.graph_convs:", len(self.graph_convs))
        for i, layer in enumerate(self.graph_convs):
            print(f"Layer {i}: {layer}")
        if isinstance(final_layer, nn.Identity):
            print("Sto prendendo il ramo con nn.Identity.")
            return final_layer(node_features)
        else:
            # Se non è di tipo nn.Identity, supponiamo che richieda i parametri batch
            print("Sto prendendo il ramo standard con argomenti node_features e batch.")
            return final_layer(node_features, batch)





    def __init__conv_layers(self):
        ############################################
        # initialize the convolutional layers interleaved with pooling layers
        graph_convs = []
         
        for i in range(len(self.num_conv_layers)):#add len
            graph_convs.append(GCNConv(in_channels=self.num_conv_layers[i][0],
                                      out_channels=self.num_conv_layers[i][1]).float())
        graph_convs.append(self.pooling)
        #modificato return nn.Sequential(*graph_convs).double()
        sequential_layers = nn.Sequential(*graph_convs).float()
        print("Complete Sequential Layer Configuration:")
        for idx, layer in enumerate(sequential_layers):
            print(f"Sequential Layer {idx}: {layer}")
        return sequential_layers
        #return nn.Sequential(*graph_convs).float()
'''
import torch.nn as nn
from torch_geometric.nn.aggr import MeanAggregation,SoftmaxAggregation
from torch_geometric.nn.conv import GCNConv

from src.core.factory_base import build_w_params_string

class GCN(nn.Module):
   
    def __init__(self, node_features, num_conv_layers=2, conv_booster=1, pooling=MeanAggregation):
        super(GCN, self).__init__()
        
        self.in_channels = node_features
        self.out_channels = int(self.in_channels * conv_booster)
          
        self.pooling =  build_w_params_string(pooling)
 
        
        if num_conv_layers>1:
            self.num_conv_layers = [(self.in_channels, self.out_channels)] + [(self.out_channels, self.out_channels) * (num_conv_layers - 1)]
        else:
            self.num_conv_layers = [(self.in_channels, self.out_channels)]
        self.graph_convs = self.__init__conv_layers()
        
    def forward(self, node_features, edge_index, edge_weight, batch):
        # convolution operations
        for conv_layer in self.graph_convs[:-1]:
        #    print(f"node_features shape: {node_features.shape}")
            node_features = node_features.float()
            node_features = conv_layer(node_features, edge_index, edge_weight)
            node_features = node_features.float()
            node_features = nn.functional.relu(node_features)

        #print("Numero di layer in self.graph_convs:", len(self.graph_convs))
        #for i, layer in enumerate(self.graph_convs):
        #    print(f"Layer {i}: {layer}")

        # global pooling
        if isinstance(self.graph_convs[-1],nn.Identity):
         #   print("Sto prendendo il ramo con nn.Identity.")
            return self.graph_convs[-1](node_features)
        #print("Sto prendendo il ramo senza nn.Identity.")
        #print("Contenuto di node_features:", node_features)
        return self.graph_convs[-1](node_features, batch)
    
    def __init__conv_layers(self):
        ############################################
        # initialize the convolutional layers interleaved with pooling layers
        graph_convs = []
        for i in range(len(self.num_conv_layers)):#add len
            graph_convs.append(GCNConv(in_channels=self.num_conv_layers[i][0],
                                      out_channels=self.num_conv_layers[i][1]).float())
        graph_convs.append(self.pooling)
        sequential_layers = nn.Sequential(*graph_convs).float()
        #print("Complete Sequential Layer Configuration:")
        for idx, layer in enumerate(sequential_layers):
          print(f"Sequential Layer {idx}: {layer}")
        return sequential_layers
        #return nn.Sequential(*graph_convs).float()
