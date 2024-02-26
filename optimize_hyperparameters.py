import optuna
import torch
import numpy as np
import torch.nn as nn
from src.core.trainable_base import Trainable
from src.core.torch_base import TorchBase
from src.core.trainable_base import Trainable
from src.dataset.dataset_base import Dataset
from torch_geometric.loader import DataLoader
from src.utils.context import Context
from torch.optim import Adam
from src.dataset.instances.base import DataInstance
from src.evaluation.evaluator_manager_do_bis import EvaluatorManager
from src.oracle.nn.gcn import DownstreamGCN
from src.dataset.utils.dataset_torch import TorchGeometricDataset
 

def optuna_objective(trial,context,dataset):
   
    num_conv_layers = trial.suggest_int("num_conv_layers",1,5,log=True)
    num_dense_layers = trial.suggest_int("num_dense_layers",1,3,log=True)
    conv_booster = trial.suggest_int("conv_booster",1,2,log=True)
    linear_decay = trial.suggest_float("linear_decay",0.5,1,log=True)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    node_features = 7
    model = DownstreamGCN(num_conv_layers=num_conv_layers, num_dense_layers=num_dense_layers, conv_booster=conv_booster, linear_decay=linear_decay,node_features=node_features)
    n_epochs = 50
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(n_epochs):
            losses = list()
            for graph in dataset.instances:
                predicted = model(graph)
                loss = loss_fn(predicted, graph.label)
                losses.append(loss.to('cpu').detach().numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            mean_loss = np.mean(losses)
            trial.report(mean_loss, epoch+1)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return mean_loss


if __name__ == "__main__":
    context = Context.get_context(config_file="config/TGITG-12725-1.4M-GCN_OBSE0.jsonc")
    context.run_number =  -1
    dataset_manager = EvaluatorManager(context)
    dataset = dataset_manager.get_dataset()
    study = optuna.create_study(study_name="GCN optimization",direction='maximize')
    study.optimize(lambda trial: optuna_objective(trial, context,dataset), n_trials=20)
    num_conv_layers = study.best_params['num_conv_layers']
    num_dense_layers = study.best_params['num_dense_layers']
    conv_booster = study.best_params['conv_booster']
    linear_decay = study.best_params['linear_decay']
    learning_rate = study.best_params['learning_rate']    
    print(f"Best trial: {study.best_trial}")
