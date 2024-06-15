import numpy as np
import torch
from src.core.trainable_base import Trainable
from src.utils.cfg_utils import init_dflts_to_of
from src.core.factory_base import get_instance_kvargs
from sklearn.metrics import accuracy_score
import torch.optim.lr_scheduler as lr_scheduler

class TorchBase(Trainable):
       
    def init(self):
        self.epochs = self.local_config['parameters']['epochs']
        self.batch_size = self.local_config['parameters']['batch_size']
        
        self.model = get_instance_kvargs(self.local_config['parameters']['model']['class'],
                                   self.local_config['parameters']['model']['parameters'])
        
        self.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})
        
        self.loss_fn = get_instance_kvargs(self.local_config['parameters']['loss_fn']['class'],
                                           self.local_config['parameters']['loss_fn']['parameters'])
        
        self.lr_scheduler =  lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=self.epochs)

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.float()  #aggiunto  Assicura che tutti i parametri del modello siano in float32

        self.model.to(self.device)                            
    
    def real_fit(self):

        if isinstance(self.dataset,list): #aggiunto 
            self.dataset = self.dataset[0] #aggiunto

        loader = self.dataset.get_torch_loader(fold_id=self.fold_id, batch_size=self.batch_size, usage='train')
        
        for epoch in range(self.epochs):
            losses = []
            preds = []
            labels_list = []
            for batch in loader:
                #modifca aggiunta
                #print("TorchBase")
                #print("Dimensioni di batch.x:", batch.x.shape)
                #print("Elementi in batch.x:", batch.x.nelement())
                  # Controlla che i tensori non siano vuoti
                #if batch.x.nelement() == 0 or batch.edge_index.nelement() == 0 or batch.edge_attr.nelement() == 0 or batch.y.nelement() == 0:
                #  raise ValueError("Uno o più tensori di input sono vuoti.")
                
                batch.batch = batch.batch.to(self.device)
                node_features = batch.x.float().to(self.device) #modificato 
               #node_features = batch.x.to(self.device)
                edge_index = batch.edge_index.to(self.device)
                edge_weights = batch.edge_attr.float().to(self.device) #modificato
                #edge_weights = batch.edge_attr.to(self.device)
                labels = batch.y.to(self.device).long()
                
                self.optimizer.zero_grad()
                
                node_features = node_features.to(dtype=torch.float32)
                edge_weights = edge_weights.to(dtype=torch.float32) # se utilizzati
                #node_features=5
                pred = self.model(node_features, edge_index, edge_weights, batch.batch)
                loss = self.loss_fn(pred, labels)
                losses.append(loss.to('cpu').detach().numpy())
                loss.backward()
                
                labels_list += list(labels.squeeze().long().detach().to('cpu').numpy())
                preds += list(pred.squeeze().detach().to('cpu').numpy())
               
                self.optimizer.step()

            accuracy = self.accuracy(labels_list, preds)
            self.context.logger.info(f'epoch = {epoch} ---> loss = {np.mean(losses):.4f}\t accuracy = {accuracy:.4f}')
            self.lr_scheduler.step()
        return accuracy #aggiunto ritorno del valore accuracy per ottimizzazione hyperparameters
            
    def check_configuration(self):
        super().check_configuration()
        local_config=self.local_config
        # set defaults
        local_config['parameters']['epochs'] = local_config['parameters'].get('epochs', 200)
        local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 4)
        # populate the optimizer
        init_dflts_to_of(local_config, 'optimizer', 'torch.optim.Adam',lr=0.001)
        init_dflts_to_of(local_config, 'loss_fn', 'torch.nn.BCELoss')
        
    def accuracy(self, testy, probs):
        acc = accuracy_score(testy, np.argmax(probs, axis=1))
        return acc

    def read(self):
        super().read()
        if isinstance(self.model, list):
            for mod in self.model:
                mod.to(self.device)
        else:
            self.model.to(self.device)
            
    def to(self, device):
        if isinstance(self.model, torch.nn.Module):
            self.model.to(device)
        elif isinstance(self.model, list):
            for model in self.model:
                if isinstance(model, torch.nn.Module):
                    model.to(self.device)