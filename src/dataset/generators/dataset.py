from os import listdir
from os.path import isfile, join

import numpy as np

from src.dataset.instances.graph import GraphInstance
from src.dataset.generators.base import Generator

class Dataset:
     def __init__(self):
        self.instances = []

     def __len__(self):
        # Restituisce il numero totale di esempi nel tuo dataset
        return len(self.instances)  

     def __getitem__(self, idx):
        # Restituisce un singolo esempio dal dataset
        instance = self.instances[idx]
        return instance.data, instance.label  # Assumi che ogni istanza abbia attributi 'data' e 'label'   
