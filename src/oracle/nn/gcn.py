import torch
import torch.nn as nn
from torch_geometric.nn.aggr import MeanAggregation

from src.utils.torch.gcn import GCN


class DownstreamGCN(GCN):
   
    def __init__(self, node_features,
                 n_classes=2,
                 num_conv_layers=2,
                 num_dense_layers=2,
                 conv_booster=2,
                 linear_decay=2,
                 pooling=MeanAggregation()):
        
        super().__init__(node_features, num_conv_layers, conv_booster, pooling)
        
        self.num_dense_layers = num_dense_layers
        self.linear_decay = linear_decay
        self.n_classes = n_classes
        
        self.downstream_layers = self.__init__downstream_layers()
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
      
    #def forward(self, node_features, edge_index, edge_weight, batch):
    #    node_features = super().forward(node_features, edge_index, edge_weight, batch)
    #    import torch

        # Inizializza un tensore vuoto con le dimensioni desiderate e lo sposta sul dispositivo di node_features
    #    new_node_features = torch.empty(32, 5, device=node_features.device)  # (dimensione_tensor, numero_posizioni_iniziali_da_riempire)

        # Riempi le prime 5 posizioni con il valore desiderato (in questo caso, 0)
    #    new_node_features.fill_(0)

        # Aggiungi le restanti dimensioni del tensore originale
    #    new_node_features = torch.cat([new_node_features, node_features[:, 5:]], dim=1)

        # Assegna il nuovo tensore a node_features
    #    node_features = new_node_features

    #    for idx, layer in enumerate(self.downstream_layers):
    #        if isinstance(layer, torch.nn.Linear):
    #            print(f"Layer {idx}: Linear with input size {layer.in_features} and output size {layer.out_features}")
    #        else:
     #           print(f"Layer {idx}: {layer.__class__.__name__}")



     #   return self.downstream_layers(node_features)

    def forward(self, node_features, edge_index, edge_weight, batch):
        #print("Dimensioni di node_features:", node_features.size())
        #print("Node features is empty:", node_features.nelement() == 0)
        node_features = super().forward(node_features, edge_index, edge_weight, batch)
        node_features = node_features.float()
        if node_features.nelement() == 0:
               # print("Attenzione: node_features è vuoto!")
                output = torch.zeros(32, self.n_classes, device=node_features.device, requires_grad=True)
                #print("Dimensioni di output per tensore vuoto:", output.size())
                return output
        #print("Contenuto di node_features in Downstream:", node_features)
        output = self.downstream_layers(node_features)
        #print("Dimensioni di output normale:", output.size())
        return output
        #return self.downstream_layers(node_features)
    #def forward(self, node_features, edge_index, edge_weight, batch):
    # Chiamata al metodo forward del super classe, se necessario
    #    if hasattr(super(), "forward"):
           # node_features = super().forward(node_features, edge_index, edge_weight, batch)
        
        # Applica le downstream layers direttamente ai node_features
    #    return self.downstream_layers(node_features)
# Calcola le features dei nodi con le layer upstream
    #   if hasattr(super(), "forward"):
    #        node_features = super().forward(node_features, edge_index, edge_weight, batch)

        # Applica le downstream layers ai node_features una alla volta
       #for layer in self.downstream_layers:
            # Calcola l'output della layer corrente
        #    layer_output = layer(node_features)
            
            # Aggiorna le node_features con l'output della layer corrente
        #    node_features = layer_output

        # Ritorna le features dei nodi elaborate
       #return node_features
       #return
    def __init__downstream_layers(self):
        ############################################
        # initialize the linear layers interleaved with activation functions
        #print("Entrata nel metodo init_downstream_layers.")
        downstream_layers = []
        in_linear = self.out_channels
        for _ in range(self.num_dense_layers-1):
            downstream_layers.append(nn.Linear(in_linear, int(in_linear // self.linear_decay)))
            downstream_layers.append(nn.ReLU())
            in_linear = int(in_linear // self.linear_decay)
        # add the output layer
        downstream_layers.append(nn.Linear(in_linear, self.n_classes))
        #downstream_layers.append(nn.Sigmoid())
        #downstream_layers.append(nn.Softmax())
        # put the linear layers in sequential
        #print("Uscita nel metodo init_downstream_layers.")
        return nn.Sequential(*downstream_layers).float()


    #def __init__downstream_layers(self):
        ############################################
        # initialize the linear layers interleaved with activation functions
    #    downstream_layers = []
    #    in_linear = self.out_channels
        #for _ in range(self.num_dense_layers-1):
        #    downstream_layers.append(nn.Linear(in_linear, int(in_linear // self.linear_decay)))
       #     downstream_layers.append(nn.ReLU())
        #    in_linear = int(in_linear // self.linear_decay)
        # add the output layer
        #downstream_layers.append(nn.Linear(in_linear, self.n_classes))
    #    for idx in range(self.num_dense_layers-1):
    #        linear_layer = nn.Linear(in_linear, int(in_linear // self.linear_decay))
    #        downstream_layers.append(linear_layer)
    #        downstream_layers.append(nn.ReLU())
    #        in_linear = int(in_linear // self.linear_decay)
    #        print(f"Layer {idx}: Linear with input size {linear_layer.in_features} and output size {linear_layer.out_features}")
        # add the output layer
    #    output_layer = nn.Linear(in_linear, self.n_classes)
    #    downstream_layers.append(output_layer)
    #    print(f"Output Layer: Linear with input size {output_layer.in_features} and output size {output_layer.out_features}")

        
        
        #downstream_layers.append(nn.Sigmoid())
        #downstream_layers.append(nn.Softmax())
        # put the linear layers in sequential
        #return nn.Sequential(*downstream_layers).double()
        #return nn.Sequential(*downstream_layers).float()


'''



import torch.nn as nn
from torch_geometric.nn.aggr import MeanAggregation

from src.utils.torch.gcn import GCN


class DownstreamGCN(GCN):
   
    def __init__(self, node_features,
                 n_classes=2,
                 num_conv_layers=2,
                 num_dense_layers=2,
                 conv_booster=2,
                 linear_decay=2,
                 pooling=MeanAggregation()):
        
        super().__init__(node_features, num_conv_layers, conv_booster, pooling)
        
        self.num_dense_layers = num_dense_layers
        self.linear_decay = linear_decay
        self.n_classes = n_classes
        
        self.downstream_layers = self.__init__downstream_layers()
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, node_features, edge_index, edge_weight, batch):
        node_features = super().forward(node_features, edge_index, edge_weight, batch)
        return self.downstream_layers(node_features)
    
    def __init__downstream_layers(self):
        ############################################
        # initialize the linear layers interleaved with activation functions
        downstream_layers = []
        in_linear = self.out_channels
        for _ in range(self.num_dense_layers-1):
            downstream_layers.append(nn.Linear(in_linear, int(in_linear // self.linear_decay)))
            downstream_layers.append(nn.ReLU())
            in_linear = int(in_linear // self.linear_decay)
        # add the output layer
        downstream_layers.append(nn.Linear(in_linear, self.n_classes))
        #downstream_layers.append(nn.Sigmoid())
        #downstream_layers.append(nn.Softmax())
        # put the linear layers in sequential
        return nn.Sequential(*downstream_layers).float()

'''