
import pandas as pd
import numpy as np
import os
from utils.dynamic_graph_temporal_signal import DynamicGraphTemporalSignalLen


class PowerGridDatasetLoader(object):
    def __init__(self, natural_folder):
        self._natural_folder = natural_folder
        self.voltages = {}
        self.edge_attr = []
        self.edge_index= []
        self.processed = False
        self.transformation_dict = {}

    
    def _process_one_case(self, name_folder):
        trans_data = pd.read_csv(os.path.join(name_folder, 'trans.csv'))
        trans_data.columns = [col.strip() for col in trans_data.columns]
        trans_data.rename(columns={'Time(s)': 'time'}, inplace=True)
        trans_data['time'] = pd.to_timedelta(trans_data['time'], unit='s') 

        trans_data.set_index('time', inplace= True)
        mean_data = trans_data.groupby(pd.Grouper(freq='0.005S')).mean()
        self.voltages_temp = {}
        self.edge_power_temp ={}
        self.edge_vars_temp = {}
        for i in mean_data.columns:
            name = i.strip().replace("'","").split(" ")
            if name[0] == 'VOLT':
                if int(name[1]) not in self.voltages_temp:
                    self.voltages_temp[int(name[1])] = mean_data[i].astype(float).values
            if name[0] == 'POWR':
                if name[5] =="1":
                    self.edge_power_temp[(int(name[1]), int(name[3]))] = mean_data[i].astype(float).values
            if name[0] == 'VARS':
                if name[5] =="1":
                    self.edge_vars_temp[(int(name[1]), int(name[3]))] = mean_data[i].astype(float).values
        return self.voltages_temp, self.edge_power_temp, self.edge_vars_temp



    def _transform_temp(self):
        transformed_dict = {}
        for key, value in self.voltages_temp.items():
            transformed_key = self.transformation_dict.get(key, None)
            if transformed_key is None:
                transformed_key = len(self.transformation_dict) 
                self.transformation_dict[key] = transformed_key  
            transformed_dict[transformed_key] = value

        self.edge_power_temp = {(self.transformation_dict[key[0]],self.transformation_dict[key[1]]): value for key, value in self.edge_power_temp.items() }
        self.edge_vars_temp ={(self.transformation_dict[key[0]],self.transformation_dict[key[1]]): value for key, value in self.edge_vars_temp.items() }
        edge_index_transformed = [(self.transformation_dict[i[0]], self.transformation_dict[i[1]]) for i in self.edge_index_temp]
        self.voltages_temp = transformed_dict
        self.edge_index_temp = edge_index_transformed
        return self.voltages_temp, self.edge_index_temp, self.edge_power_temp, self.edge_vars_temp

    
    def _update_mix_weights(self):
        #We want to create a matrix of shape (n_edges, n_timestamps, n_features)
        n_edges = len(self.edge_index_temp)
        n_timestamps = len(next(iter(self.voltages_temp.values())))
        edge_attr_temp = np.zeros((n_edges, n_timestamps, 2))
        
        for i, edge in enumerate(self.edge_index_temp):
            if edge in self.edge_power_temp:
                edge_attr_temp[i, :, 0] = self.edge_power_temp[edge]
            if edge in self.edge_vars_temp:
                edge_attr_temp[i, :, 1] = self.edge_vars_temp[edge]
    
        self.edge_attr.append(edge_attr_temp)

    def reconstruir_voltages(self, n):
        info_nodos = []
        for key, value in self.voltages.items():
            info_nodos.append(value[n].tolist())
        return info_nodos

    def _update_voltages(self):
        n_each = len(self.voltages_temp[list(self.voltages_temp.keys())[0]])
        n_situation = 0 if len(self.voltages) == 0 else len(self.voltages[list(self.voltages.keys())[0]]) 
        for i in self.voltages_temp:
            if i not in self.voltages:
                print("Node: ", i, " not included, including...")
                self.voltages[i] =[np.zeros(n_each) for _ in range(n_situation)] + [self.voltages_temp[i]]
            else:
                self.voltages[i].append(self.voltages_temp[i])
    

    def _check_nodes(self):
        edge_vars_keys = set(self.edge_vars_temp.keys())
        edge_power_keys = set(self.edge_power_temp.keys())
        unique_keys = edge_vars_keys.union(edge_power_keys)
        unique_nodes =set(integer for key in unique_keys for integer in key)
        self.edge_index_temp = list(unique_keys)
        not_present_ints = [integer for integer in unique_nodes if integer not in self.voltages_temp]
        not_present_2 = [integer for integer in self.voltages_temp if integer not in unique_nodes ]
        if len(not_present_ints) > 0:
            print("Nodes not present in voltages: ", not_present_ints)
            for i in not_present_ints:
                self.voltages_temp[i] = np.zeros(len(self.voltages_temp[list(self.voltages_temp.keys())[0]]))
        if len(not_present_2)>0:
            print("Nodes not present in edges: ", not_present_2)
            for i in not_present_2:
                self.voltages_temp.pop(i)
    
        return self.voltages_temp, self.edge_index_temp
    

  
    def _get_targets_and_features(self):
        #We want to get the targets and features, which will be: for each situation, we will predict the voltages of the nodes in the last 40 timestamps
        if not self.processed:
            self.process()
              
        n_situations = len(next(iter(self.voltages.values()))) 
        print("Number of situations: ", n_situations)
        n_keys = len(self.transformation_dict)

        n_timestamps = len(next(iter(self.voltages.values()))[0])
        print("Number of timestamps: ", n_timestamps)

        voltages_def = np.zeros((n_situations, n_keys, n_timestamps))

        for key, arrays_list in self.voltages.items():
            for situation, array in enumerate(arrays_list):
                voltages_def[situation, key, :] = array
        
        n_situations = self._limit
        #self.features = [voltages_def[i, :, :-self._target] for i in range(n_situations)]
        #self.targets = [voltages_def[i, : , -self._target:] for i in range(n_situations)]
        #(n_situations, n_edges, n_timestamps, n_features) 
        #self.edge_weights = [self.edge_attr[i][:,:-self._target,:].reshape(len(self.edge_index[0]), 2,n_timestamps-self._target) for i in range(n_situations)]
        
        self.features = [voltages_def[i, :, j:j+self._intro] for i in range(n_situations) for j in range(0,n_timestamps-self._target-self._intro, self._step)]
        self.targets = [voltages_def[i, : ,j+self._intro:j+self._intro+self._target] for i in range(n_situations) for j in range(0,n_timestamps-self._target-self._intro, self._step)]
        self.edge_weights = [self.edge_attr[i][:,j:j+self._intro,:].reshape(len(self.edge_index[0]), 2,self._intro) for i in range(n_situations) for j in range(0,n_timestamps-self._target-self._intro, self._step)]

        div = int(len(self.features)/n_situations)

        repeated_index = [self.edge_index[j] for j in range(n_situations) for k in range(div) for i in range(self._intro)]

        repeated_index = np.array(repeated_index).reshape((n_situations*div, len(self.edge_index[0]),2,  self._intro))
        self.edges = [repeated_index[i, :, :, :] for i in range(n_situations*div)]

        


    def process(self):
        print("Here")
        for root, dirs, files in os.walk(self._natural_folder):
            for folder in dirs:
                folder_name = os.path.join(root, folder)
                print("Processing: ", folder)
                self._process_one_case(folder_name)

                #We check if all the nodes present in the edges have voltage info and viceversa
                self._check_nodes()
                
                #We transform the nodes ids to correct numbering (easier)
                self._transform_temp()

                
                #We update the voltage matrix
                self._update_voltages()

                #We create the edge attribute matrix, corresponding to the edge_index_ordering
                self._update_mix_weights()
                self.edge_index.append([[i[0], i[1]] for i in self.edge_index_temp])

        self.processed = True
        return self.voltages, self.edge_index, self.edge_attr

    
    
    def get_dataset(self, target= 50, intro=200, step=50, limit=None):
        self._target = target
        self._intro = intro
        self._step = step
        
        if not self.processed:
            self.process()
        self._limit = len(next(iter(self.voltages.values()))) if limit is None else limit
        self._get_targets_and_features()
        dataset = DynamicGraphTemporalSignalLen(self.edges, self.edge_weights, self.features, self.targets, "PowerGridDataset")
        return dataset


