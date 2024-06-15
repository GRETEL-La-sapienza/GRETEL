'''import numpy as np

from src.core.configurable import Configurable

class BaseManipulator(Configurable):
    
    def __init__(self, context, local_config, dataset):
        self.dataset = dataset
        super().__init__(context, local_config)
        
    def init(self):
        super().init()
        self.manipulated = False
        self._process()
         
    def _process(self):
        for instance in self.dataset.instances:
            print("Features del nodo all'ingresso:", instance.node_features.shape)
            try:
                node_features_map = self.node_info(instance)
                edge_features_map = self.edge_info(instance)
                graph_features_map = self.graph_info(instance)
                self.manipulate_features_maps((node_features_map, edge_features_map, graph_features_map))
                # overriding the features
                # resize in num_nodes x feature dim
                instance.node_features = self.__process_features(instance.node_features, node_features_map, self.dataset.node_features_map)
                instance.edge_features = self.__process_features(instance.edge_features, edge_features_map, self.dataset.edge_features_map)
                instance.graph_features = self.__process_features(instance.graph_features, graph_features_map, self.dataset.graph_features_map)
                print("Features del nodo dopo la manipolazione:", instance.node_features.shape)
            except Exception as e:
                print("Errore durante la manipolazione delle features:", str(e))
                break  # interrompe il ciclo per evitare ulteriori errori

           

    def _process_instance(self,instance):
        node_features_map = self.node_info(instance)
        edge_features_map = self.edge_info(instance)     
        graph_features_map = self.graph_info(instance)
        self.manipulate_features_maps((node_features_map, edge_features_map, graph_features_map))
        # overriding the features
        # resize in num_nodes x feature dim
        instance.node_features = self.__process_features(instance.node_features, node_features_map, self.dataset.node_features_map)
        instance.edge_features = self.__process_features(instance.edge_features, edge_features_map, self.dataset.edge_features_map)
        instance.graph_features = self.__process_features(instance.graph_features, graph_features_map, self.dataset.graph_features_map)

       
    def node_info(self, instance):
        return {}
    
    def graph_info(self, instance):
        return {}
    
    def edge_info(self, instance):
        return {}
    
    def manipulate_features_maps(self, feature_values):
        if not self.manipulated:
            node_features_map, edge_features_map, graph_features_map = feature_values
            self.dataset.node_features_map = self.__process_map(node_features_map, self.dataset.node_features_map)
            self.dataset.edge_features_map = self.__process_map(edge_features_map, self.dataset.edge_features_map)
            self.dataset.graph_features_map = self.__process_map(graph_features_map, self.dataset.graph_features_map)
            self.manipulated = True
             # Stampa della dimensione totale delle features riconosciute
            print("Dimensione totale delle features riconosciute (node_features):", len(self.dataset.node_features_map))
            print("Dimensione totale delle features riconosciute (edge_features):", len(self.dataset.edge_features_map))
            print("Dimensione totale delle features riconosciute (graph_features):", len(self.dataset.graph_features_map))
    
    def __process_map(self, curr_map, dataset_map):
        _max = max(dataset_map.values()) if dataset_map.values() else -1
        for key in curr_map:
            if key not in dataset_map:
                _max += 1
                dataset_map[key] = _max
        return dataset_map
    


    def __process_features(self, features, curr_map, dataset_map):
        if curr_map:
            if not isinstance(features, np.ndarray):
                features = np.array([])
            try:
                old_feature_dim = features.shape[1]
            except IndexError:
                old_feature_dim = 0
            # If the feature vector doesn't exist, then
            # here we're creating it for the first time
            if old_feature_dim:
                features = np.pad(features,
                                pad_width=((0, 0), (0, len(dataset_map) - old_feature_dim)),
                                constant_values=0)
            else:
                features = np.zeros((len(list(curr_map.values())[0]), len(dataset_map)))
                
            for key in curr_map:
                index = dataset_map[key]
                if isinstance(curr_map[key], list) or isinstance(curr_map[key], np.ndarray):
                    # Prepara un array vuoto per raccogliere i valori da media
                    values = []
                    for x in curr_map[key]:
                        if isinstance(x, (np.ndarray, list)) and len(x) > 0:
                            values.append(x[0])  # Assumi che x sia un array o una lista e prendi il primo elemento
                        else:
                            values.append(x)  # Usa direttamente x se non è una sequenza o è vuoto
                    # Calcola la media dei valori raccolti se la lista non è vuota
                    aggregated_value = np.mean(values) if values else 0
                    features[:, index] = aggregated_value
                else:
                    features[:, index] = curr_map[key]
                            
        return features
'''
import numpy as np

from src.core.configurable import Configurable

class BaseManipulator(Configurable):
    
    def __init__(self, context, local_config, dataset):
        self.dataset = dataset
        super().__init__(context, local_config)
        
    def init(self):
        super().init()
        self.manipulated = False
        self._process()
         
    def _process(self):
        for instance in self.dataset.instances:
            node_features_map = self.node_info(instance)
            edge_features_map = self.edge_info(instance)
            graph_features_map = self.graph_info(instance)
            self.manipulate_features_maps((node_features_map, edge_features_map, graph_features_map))
            # overriding the features
            # resize in num_nodes x feature dim
            instance.node_features = self.__process_features(instance.node_features, node_features_map, self.dataset.node_features_map)
            instance.edge_features = self.__process_features(instance.edge_features, edge_features_map, self.dataset.edge_features_map)
            instance.graph_features = self.__process_features(instance.graph_features, graph_features_map, self.dataset.graph_features_map)

           

    def _process_instance(self,instance):
        node_features_map = self.node_info(instance)
        edge_features_map = self.edge_info(instance)
        graph_features_map = self.graph_info(instance)
        self.manipulate_features_maps((node_features_map, edge_features_map, graph_features_map))
        # overriding the features
        # resize in num_nodes x feature dim
        instance.node_features = self.__process_features(instance.node_features, node_features_map, self.dataset.node_features_map)
        instance.edge_features = self.__process_features(instance.edge_features, edge_features_map, self.dataset.edge_features_map)
        instance.graph_features = self.__process_features(instance.graph_features, graph_features_map, self.dataset.graph_features_map)

       
    def node_info(self, instance):
        return {}
    
    def graph_info(self, instance):
        return {}
    
    def edge_info(self, instance):
        return {}
    
    def manipulate_features_maps(self, feature_values):
        if not self.manipulated:
            node_features_map, edge_features_map, graph_features_map = feature_values
            self.dataset.node_features_map = self.__process_map(node_features_map, self.dataset.node_features_map)
            self.dataset.edge_features_map = self.__process_map(edge_features_map, self.dataset.edge_features_map)
            self.dataset.graph_features_map = self.__process_map(graph_features_map, self.dataset.graph_features_map)
            self.manipulated = True
    
    def __process_map(self, curr_map, dataset_map):
        _max = max(dataset_map.values()) if dataset_map.values() else -1
        for key in curr_map:
            if key not in dataset_map:
                _max += 1
                dataset_map[key] = _max
        return dataset_map

    def __process_features(self, features, curr_map, dataset_map):
        if curr_map:
            if not isinstance(features, np.ndarray):
              features = np.array([])
            try:
              old_feature_dim = features.shape[1]
            except IndexError:
              old_feature_dim = 0
            # If the feature vector doesn't exist, then
            # here we're creating it for the first time
            if old_feature_dim:
              new_pad_width = max(0, len(dataset_map) - old_feature_dim)
              features = np.pad(features, pad_width=((0, 0), (0, new_pad_width)), constant_values=0)
                #features = np.pad(features,
                #                pad_width=((0, 0), (0, len(dataset_map) - old_feature_dim)),
                #                constant_values=0)
            else:
              features = np.zeros((len(list(curr_map.values())[0]), len(dataset_map)))

            for key in curr_map:
            # index = dataset_map[key]
            # features[:, index] = curr_map[key]
            #code modified for an error
            # Assumendo che curr_map[key] sia una sequenza e tu voglia assegnare questa sequenza a una colonna specifica di 'features'
                index = dataset_map[key] # Assicurati che questo sia un intero

                sequence = np.array(curr_map[key]) # Converti la sequenza in un array NumPy
                if sequence.ndim == 1: # Se la sequenza è unidimensionale
                   sequence = sequence.reshape(-1, 1) # Trasforma in colonna se necessario
                features[:, index:index+sequence.shape[1]] = sequence # Assegna la sequenza alla colonna


        return features
  

'''
    def __process_features(self, features, curr_map, dataset_map):
        print("Current Map (curr_map):", curr_map)
        if curr_map:
            if not isinstance(features, np.ndarray):
                features = np.array([])
            try:
                old_feature_dim = features.shape[1]
            except IndexError:
                old_feature_dim = 0
            # If the feature vector doesn't exist, then
            # here we're creating it for the first time
            if old_feature_dim:
                pad_width_value = len(dataset_map) - old_feature_dim
                if pad_width_value > 0:
                # Applica il padding solo se il numero di caratteristiche attuali è minore di quello richiesto
                    features = np.pad(features,
                                    pad_width=((0, 0), (0, pad_width_value)),
                                    constant_values=0)
            else:
                features = np.zeros((len(list(curr_map.values())[0]), len(dataset_map)))

            print("Current Map (curr_map):", curr_map) 
            for key in curr_map:
                index = dataset_map[key]
                print("Key (k):", key)
                print("Index corresponding to key in dataset_map:", index)
                value_to_assign = np.asarray(curr_map[key])
                value_to_assign = value_to_assign.ravel()
                if value_to_assign.ndim > 1 or value_to_assign.size != features.shape[0]:
                    raise ValueError(f"Data size mismatch or incorrect dimension for key {key}: expected size {features.shape[0]}, got size {value_to_assign.size} with dimension {value_to_assign.ndim}.")

                features[:, index] = value_to_assign
                #features[:, index] = curr_map[key]
                print("Features after assignment at index [all, {}]:".format(index), features[:, index])
  
                            
        return features
    





    def __process_features(self, features, curr_map, dataset_map):
        if curr_map:
            if not isinstance(features, np.ndarray):
                features = np.array([])
            try:
                old_feature_dim = features.shape[1]
            except IndexError:
                old_feature_dim = 0
            # If the feature vector doesn't exist, then
            # here we're creating it for the first time
            if old_feature_dim:
                features = np.pad(features,
                                pad_width=((0, 0), (0, len(dataset_map) - old_feature_dim)),
                                constant_values=0)
            else:
                features = np.zeros((len(list(curr_map.values())[0]), len(dataset_map)))
                
            for key in curr_map:
                index = dataset_map[key]
                features[:, index] = curr_map[key]
                            
        return features




    def __process_features(self, features, curr_map, dataset_map):
        if curr_map:
            if not isinstance(features, np.ndarray):
                features = np.array([])
            try:
                old_feature_dim = features.shape[1]
            except IndexError:
                old_feature_dim = 0

            required_feature_dim = len(dataset_map)
            if old_feature_dim > required_feature_dim:
                print("Warning: features have more dimensions than currently mapped. Adjusting...")
                features = features[:, :required_feature_dim]  # Tronca le caratteristiche extra
            elif old_feature_dim < required_feature_dim:
                pad_width_value = required_feature_dim - old_feature_dim
                features = np.pad(features,
                                pad_width=((0, 0), (0, pad_width_value)),
                                constant_values=0)

            for key in curr_map:
                index = dataset_map[key]
                features[:, index] = curr_map[key]
                                
        return features


    def __process_features(self, features, curr_map, dataset_map):
         
        if curr_map:
            if not isinstance(features, np.ndarray):
                features = np.array([])
            try:
                old_feature_dim = features.shape[1]
            except IndexError:
                old_feature_dim = 0
            # If the feature vector doesn't exist, then
            # here we're creating it for the first time
            
            if old_feature_dim:
                # Calcola il valore di pad_width
                pad_width_value = 5 - old_feature_dim

                # Stampa i valori per il debug
                print("old_feature_dim:", old_feature_dim)
                print("len(dataset_map):", len(dataset_map))
                print("Calculated pad_width_value:", pad_width_value)

                # Controlla se il pad_width_value è negativo
                if pad_width_value < 0:
                    print("Error: pad_width_value is negative, which is not allowed.")

                # Applica np.pad solo se pad_width_value è non negativo
                if pad_width_value >= 0:
                    features = np.pad(features,
                                    pad_width=((0, 0), (0, pad_width_value)),
                                    constant_values=0)
                else:
                    # Gestisci il caso in cui pad_width_value sia negativo
                    # Potresti voler assegnare a features un valore di default o sollevare un'eccezione
                    features = np.zeros((len(list(curr_map.values())[0]), len(dataset_map)))
            else:
                # Stampa le dimensioni iniziali di features e di dataset_map in caso di old_feature_dim assente
                print("old_feature_dim is None or 0")
                print("Creating zero matrix with dimensions:", len(list(curr_map.values())[0]), len(dataset_map))
                features = np.zeros((len(list(curr_map.values())[0]), len(dataset_map)))




            #if old_feature_dim:
            #    features = np.pad(features,
            #                    pad_width=((0, 0), (0, len(dataset_map) - old_feature_dim)),
            #                    constant_values=0)
            #else:
            #    features = np.zeros((len(list(curr_map.values())[0]), len(dataset_map)))
                
            for key in curr_map:
                index = dataset_map[key]
                features[:, index] = curr_map[key]
                            
        return features
    

  '''  