
'''import torch
import torch.nn as nn
from torch.nn import LeakyReLU

from src.core.factory_base import build_w_params_string
from src.utils.cfg_utils import default_cfg


class ResGenerator(nn.Module):
    
    def __init__(self, num_nodes,
                 conv_kernel=(3,3), conv_stride=(2,2),
                 deconv_kernel=(4,4), deconv_stride=(2,2),
                 activation=LeakyReLU(),
                 dropout_p=0.2,
                 residuals=True):
        
        super(ResGenerator, self).__init__()
        
        self.num_nodes = num_nodes
        print("Numero di nodi:", self.num_nodes)  # Aggiunta stampa del numero di nodi

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=conv_kernel, stride=conv_stride)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=conv_kernel, stride=conv_stride)
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=deconv_kernel, stride=deconv_stride, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=deconv_kernel, stride=deconv_stride, padding=1)
        
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(7,7), stride=(1,1), padding=self.__get_same_padding(7,1))
        
        self.flatten = nn.Flatten()
        self.act =  build_w_params_string(activation)
        self.dropout = nn.Dropout2d(p=dropout_p)

        self.flatten = nn.Flatten()

        self.fc_len = self.init_run()
        
        # modificato aggiunto Linear layers with an intermediate layer to reduce dimensions
        #intermediate_features = max(64, min(128, self.fc_len, self.num_nodes // 4))
        #self.fc = nn.Linear(in_features=64 * self.fc_len**2, out_features=intermediate_features)
        #self.fc2 = nn.Linear(in_features=intermediate_features, out_features=128 * (self.num_nodes // 4)**2)

        # modificato aggiunto Linear layers with an intermediate layer to reduce dimensions
        intermediate_features = max(64, min(128, self.fc_len, self.num_nodes // 4))
        self.fc = nn.Linear(in_features=64 * self.fc_len**2, out_features=intermediate_features).to(self.device).float()
        self.fc2 = nn.Linear(in_features=intermediate_features, out_features=128 * (self.num_nodes // 4)**2).to(self.device).float()


        #self.fc = nn.Linear(in_features=64 * self.fc_len**2,
        #                    out_features=128 * (self.num_nodes // 4)**2)
        
        self.residuals = residuals
        self.training = False
        
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
           # if isinstance(m, nn.Conv3d):
            if isinstance(m, nn.Conv2d):  # Modificato aggiunta da Conv3d a Conv2d se stai usando Conv2d nel tuo modello
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.data = m.weight.data.float()  # modificata aggiunta Converti i pesi in float32
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    m.bias.data = m.bias.data.float()  # modificata aggiunta Converti il bias in float32
            #elif isinstance(m, nn.BatchNorm3d):
            elif isinstance(m, nn.BatchNorm2d): #mdoficata aggiunta da batchnorm3d a batchnorm2d
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                m.weight.data = m.weight.data.float()  #modificata aggiunta  Converti i pesi in float32
                m.bias.data = m.bias.data.float()  # modificata aggiunta Converti il bias in float32
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                m.weight.data = m.weight.data.float()  # modificata aggiunta Converti i pesi in float32
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    m.bias.data = m.bias.data.float()  # moficiata aggiunta Converti il bias in float32
        
    def set_training(self, training):
        self.training = training

    def init_run(self):
        with torch.no_grad():
            #dummy_input = torch.randn((1, 1, self.num_nodes, self.num_nodes), device=self.device).to(torch.float32)
            #dummy_input = torch.randn((1, 1, self.num_nodes, self.num_nodes), device='cpu').to(torch.float32)
            dummy_input = torch.randn((1, 1, self.num_nodes, self.num_nodes)).to(torch.float32)
            x = self.conv2(self.conv1(dummy_input))
            print("Dimensione dell'ultimo asse dell'output:", x.shape[-1])  # Stampa la dimensione dell'ultimo asse dell'output
        return x.shape[-1]

    #def init_run(self):
    #    with torch.no_grad():
            # Crea dummy_input del tipo corretto e assicurati che sia su 'self.device'
    #        dummy_input = torch.randn((1, 1, self.num_nodes, self.num_nodes), dtype=torch.float).to(self.device) #modificato aggiunto 
    #        print("Tipo input:", dummy_input.dtype, "Tipo peso:", self.conv1.weight.dtype)

            #dummy_input = torch.randn((1,1, self.num_nodes, self.num_nodes)).to(self.device)
            #x = self.conv2(self.conv1(dummy_input))
    #        x = self.conv1(dummy_input) #modificato aggiunto
    #        print("Dopo conv1:", x.dtype, x.shape)
    #        x = self.conv2(x) #modificato aggiunto
    #        print("Dopo conv2:", x.dtype, x.shape)
            # Assicurati che l'output non sia vuoto o di dimensione inaspettata
     #       if x.nelement() == 0:
     #            raise RuntimeError("Dimensione dell'output convoluzionale è zero. Controlla la configurazione dei layer.")

      #  return x.shape[-1]

    def forward(self, graph):
        graph = graph.to(dtype=torch.float32, device=self.device)  # modificata aggiunta Assicurati che l'input sia in float32
        x = self.act(self.conv1(graph))
        x = self.dropout(x)
        x = self.act(self.conv2(x))
        x = self.dropout(x)
        x = self.flatten(x)
        #x = self.act(self.fc(x))
        x = self.act(self.fc2(x))
        print("Dimensioni di x prima del reshape:", x.shape)  # Aggiunta per debug
        expected_num_elements = 128 * (self.num_nodes // 4) * (self.num_nodes // 4) #aggiunto per debug
        actual_num_elements = x.numel()  # Numero effettivo di elementi in x #aggiunto per debug

        print("Elementi attesi:", expected_num_elements) #aggiunto per debug
        print("Elementi attuali:", actual_num_elements) #aggiunto per debug

        #if actual_num_elements != expected_num_elements: #aggiunto per debug
        #     raise ValueError("Il numero di elementi non corrisponde.") #aggiunto per debug
        print("Dimensioni di x prima del reshaping:", x.shape)
        print("Dimensioni di x prima del reshaping:", x.shape)
        print("Numero totale di elementi in x:", x.numel())
        x = x.view((-1, 128, self.num_nodes//4, self.num_nodes//4))
        print("Dimensioni di x dopo il reshaping:", x.shape)
        x = self.act(self.deconv1(x))
        x = self.act(self.deconv2(x))
        x = self.final_conv(x)
        x = torch.tanh(x) # Tanh values are in [-1, 1] so allow the residual to add or remove edges
        # building the counterfactual example from the union of the residuals and the original instance
        if self.residuals:
            x = torch.add(graph, x)
        # delete self loops
        for i in range(x.shape[0]):
            x[i][0].fill_diagonal_(0)
            
        #mask = ~torch.eye(self.num_nodes, dtype=bool)
        #mask = torch.stack([mask] * x.shape[0])[:,None,:,:]
        mask = ~torch.eye(self.num_nodes, dtype=bool).to(x.device) #MODIFICATO AGGIUNTO
        mask = torch.stack([mask] * x.shape[0], dim=0).unsqueeze(1) #modificato aggiunto
        x[mask] = torch.sigmoid(x[mask])

         # Binarizza l'output basato sulla soglia casuale
        x = (torch.rand_like(x) < x).float()  # modificata aggiunta Assicurati che l'output sia in float
        
        #x = (torch.rand_like(x) < x).to(torch.float)
                      
        return x


    def __get_same_padding(self, kernel_size, stride):
        return 0 if stride != 1 else (kernel_size - stride) // 2
    
    
    @default_cfg
    def grtl_default(kls, num_nodes):
        return {"class": kls, "parameters": { "num_nodes": num_nodes } }
    
        '''
import torch
import torch.nn as nn
from torch.nn import LeakyReLU
import torch.nn.functional as F
from src.core.factory_base import build_w_params_string
from src.utils.cfg_utils import default_cfg


class ResGenerator(nn.Module):
    
    def __init__(self, num_nodes,
                 conv_kernel=(3,3), conv_stride=(2,2),
                 deconv_kernel=(4,4), deconv_stride=(2,2),
                 activation=LeakyReLU(),
                 dropout_p=0.2,
                 residuals=True):
        
        super(ResGenerator, self).__init__()
        
        self.num_nodes = num_nodes
        
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=conv_kernel, stride=conv_stride)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=conv_kernel, stride=conv_stride)
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=deconv_kernel, stride=deconv_stride, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=deconv_kernel, stride=deconv_stride, padding=1)
        
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(7,7), stride=(1,1), padding=self.__get_same_padding(7,1))
        
        self.flatten = nn.Flatten()
        self.act =  build_w_params_string(activation)
        self.dropout = nn.Dropout2d(p=dropout_p)

        self.flatten = nn.Flatten()

        self.fc_len = self.init_run()
        
        self.fc = nn.Linear(in_features=64 * self.fc_len**2,
                            out_features=128 * (self.num_nodes // 4)**2)
        
        self.residuals = residuals
        self.training = False
        
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def set_training(self, training):
        self.training = training

    def init_run(self):
        with torch.no_grad():
            dummy_input = torch.randn((1,1, self.num_nodes, self.num_nodes)).to(self.device)
            print(dummy_input.size())
            self.conv1.to(self.device)
            self.conv2.to(self.device)

            x = self.conv1(dummy_input)
            print("Output after conv1:", x.shape)
            #x = self.conv2(self.conv1(dummy_input))
            x = self.conv2(x)
            print("Output after conv2:", x.shape)
        return x.shape[-1]

    def forward(self, graph):
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.to(self.device)
        graph = graph.to(self.device)
        #print("Dimensione di graph iniziale:", graph.shape)
        x = self.act(self.conv1(graph))
        #print("Dimensione di x dopo conv1 e attivazione:", x.shape)
        x = self.dropout(x)
        #print("Dimensione di x dopo dropout:", x.shape)
        x = self.act(self.conv2(x))
        #print("Dimensione di x dopo conv2 e attivazione:", x.shape)
        x = self.dropout(x)
        #print("Dimensione di x dopo dropout:", x.shape)
        x = self.flatten(x)
       # print("Dimensione di x dopo flatten:", x.shape)
        x = self.act(self.fc(x))
        #print("Dimensione di x dopo fully connected e attivazione:", x.shape)
        x = x.view((-1, 128, self.num_nodes//4, self.num_nodes//4))
        #print("Dimensione di x dopo reshaping:", x.shape)
        x = self.act(self.deconv1(x))
        #print("Dimensione di x dopo deconv1 e attivazione:", x.shape)
        x = self.act(self.deconv2(x))
        #print("Dimensione di x dopo deconv2 e attivazione:", x.shape)
        x = self.final_conv(x)
        #print("Dimensione di x dopo finalconv(conv2) :", x.shape)
        x = torch.tanh(x) # Tanh values are in [-1, 1] so allow the residual to add or remove edges
        #print("Dimensione di x dopo torch.tanh(x):", x.shape)
        # building the counterfactual example from the union of the residuals and the original instance
        
        #print("Dimensione di graph:", graph.shape)
        #print("Dimensione di x:", x.shape)
        target_size = graph.shape[2:]
        #x = F.interpolate(x, size=(10, 10), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        #print("Dimensione di x dopo interpolazione:", x.shape)

        if self.residuals:
            x = torch.add(graph, x)
        # delete self loops
        for i in range(x.shape[0]):
            x[i][0].fill_diagonal_(0)
            
        mask = ~torch.eye(self.num_nodes, dtype=bool)
        mask = torch.stack([mask] * x.shape[0])[:,None,:,:]
        x[mask] = torch.sigmoid(x[mask])
        
        x = (torch.rand_like(x) < x).to(torch.float)
                      
        return x


    def __get_same_padding(self, kernel_size, stride):
        return 0 if stride != 1 else (kernel_size - stride) // 2
    
    
    @default_cfg
    def grtl_default(kls, num_nodes):
        return {"class": kls, "parameters": { "num_nodes": num_nodes } }