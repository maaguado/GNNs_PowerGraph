import pandas as pd
import numpy as np
import os
from utils.dynamic_graph_temporal_signal import DynamicGraphTemporalSignalLen
from sklearn.preprocessing import OneHotEncoder


class PowerGridDatasetLoader(object):
    """
    A class to load and preprocess power grid dataset for regression or classification tasks.

    Attributes:
        _natural_folder (str): The path to the folder containing dataset files.
        problem (str): The type of problem, either 'regression' or 'classification'.
        voltages (dict): Dictionary containing voltage data for each node.
        buses (list): List of bus numbers.
        types (list): List of bus types for classification problem.
        edge_attr (list): List of edge attributes.
        edge_index (list): List of edge indices.
        processed (bool): Flag indicating whether data processing has been done.
        transformation_dict (dict): Dictionary for node id transformation.
    """

    def __init__(self, natural_folder, problem="regression"):
        """
        Initialize PowerGridDatasetLoader with dataset folder and problem type.

        Args:
            natural_folder (str): The path to the folder containing dataset files.
            problem (str): The type of problem, either 'regression' or 'classification'.
        """
        self._natural_folder = natural_folder
        self.voltages = {}
        self.buses = []
        self.types = []
        self.edge_attr = []
        self.edge_index = []
        self.processed = False
        self.transformation_dict = {}
        self.problem = problem

    def _process_one_case(self, name_folder):
        """
        Process one dataset case.

        Args:
            name_folder (str): The name of the folder containing dataset files.

        Returns:
            Tuple: Tuple containing voltage, edge power, and edge vars data.
        """
        # Processing dataset files
        
        info = pd.read_csv(os.path.join(name_folder, 'info.csv'),index_col=0, header=None).T
        self.buses.append(int(info['bus1'].values[0].strip()))
        self.types.append(info['type'].values[0].strip())

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

    def _preprocess_targets(self):
        """Preprocess targets for classification or regression."""
        if (self.problem.split("_")[1]=='type'):
            encoder = OneHotEncoder(sparse_output=False)
            self.processed_targets = encoder.fit_transform(np.array(self.types).reshape(-1, 1))

        elif (self.problem.split("_")[1]=='bus'):
            self.buses = [self.transformation_dict[i] for i in self.buses]
            encoder = OneHotEncoder(sparse_output=False)
            self.processed_targets = encoder.fit_transform(np.array(self.buses).reshape(-1, 1))

    def _transform_temp(self):
        """Transform temporary data."""
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
        """Update edge attribute matrix."""
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
        """
        Reconstruct voltages for a specific situation.

        Args:
            n (int): Situation index.

        Returns:
            list: List containing voltage information.
        """
        info_nodos = []
        for key, value in self.voltages.items():
            info_nodos.append(value[n].tolist())
        return info_nodos

    def _update_voltages(self):
        """Update voltage matrix."""
        n_each = len(self.voltages_temp[list(self.voltages_temp.keys())[0]])
        n_situation = 0 if len(self.voltages) == 0 else len(self.voltages[list(self.voltages.keys())[0]]) 
        for i in self.voltages_temp:
            if i not in self.voltages:
                print("Node: ", i, " not included, including...")
                self.voltages[i] =[np.zeros(n_each) for _ in range(n_situation)] + [self.voltages_temp[i]]
            else:
                self.voltages[i].append(self.voltages_temp[i])

    def _check_nodes(self):
        """Check nodes present in edges and voltages."""
        edge_vars_keys = set(self.edge_vars_temp.keys())
        edge_power_keys = set(self.edge_power_temp.keys())
        unique_keys = edge_vars_keys.union(edge_power_keys)
        unique_nodes = set(integer for key in unique_keys for integer in key)
        self.edge_index_temp = list(unique_keys)
        not_present_ints = [integer for integer in unique_nodes if integer not in self.voltages_temp]
        not_present_2 = [integer for integer in self.voltages_temp if integer not in unique_nodes]
        if len(not_present_ints) > 0:
            print("Nodes not present in voltages: ", not_present_ints)
            for i in not_present_ints:
                self.voltages_temp[i] = np.zeros(len(self.voltages_temp[list(self.voltages_temp.keys())[0]]))
        if len(not_present_2) > 0:
            print("Nodes not present in edges: ", not_present_2)
            for i in not_present_2:
                self.voltages_temp.pop(i)
    
        return self.voltages_temp, self.edge_index_temp



    def _get_targets_and_features(self):
        """
        Get the targets and features.

        This method retrieves the targets and features from the processed data based on the problem type.

        Returns:
            None
        """
        # If data is not processed yet, call the process method to prepare the dataset
        if not self.processed:
            self.process()
              
        # Get the number of situations (timestamps) in the dataset
        n_situations = len(next(iter(self.voltages.values()))) 
        print("Number of situations: ", n_situations)

        # Get the number of nodes (keys) and timestamps
        n_keys = len(self.transformation_dict)
        n_timestamps = len(next(iter(self.voltages.values()))[0])
        print("Number of timestamps: ", n_timestamps)

        # Initialize an array to store the voltage data for each situation
        voltages_def = np.zeros((n_situations, n_keys, n_timestamps))

        # Populate the voltages_def array with voltage data
        for key, arrays_list in self.voltages.items():
            for situation, array in enumerate(arrays_list):
                voltages_def[situation, key, :] = array
        


        indices = np.where(np.array(self.types)==self._type)[0] if self._type is not None else range(n_situations)

        print("Number of situations of the selected type: ", len(indices))


        voltages_def = voltages_def[indices, :, :]
        edge_attr = [self.edge_attr[i] for i in indices]
        edge_index = [self.edge_index[i] for i in indices]
        # Set the number of situations to the limit or total number of situations if no limit is specified
        n_situations = min(self._limit, len(indices)-self.start) if self._limit is not None else len(indices)-self.start
        
        # If the problem type is regression
        if (self.problem.split("_")[0]=='regression'):
            # If considering only one timestamp per situation
            if (self._one_situation):

                situations_each = indices
                # Assign features as voltages up to the target timestamp
                self.features = [voltages_def[i, :, :-self._target] for i in range(self.start, self.start+n_situations)]
                # Assign targets as voltages from the target timestamp onwards
                self.targets = [voltages_def[i, : , -self._target:] for i in range(self.start, self.start+ n_situations)]
                # Assign edge weights as edge attributes excluding the target timestamp
                self.edge_weights = [edge_attr[i][:,:-self._target,:].reshape(n_timestamps-self._target, len(edge_index[0]), 2) for i in range(self.start,self.start+ n_situations)]
                # Repeat the edge index for each situation
                repeated_index = [edge_index[j] for j in range(self.start, self.start+n_situations) for i in range(n_timestamps-self._target)]
                repeated_index = np.array(repeated_index).reshape((n_situations, n_timestamps-self._target, len(edge_index[0]),2  ))
                self.edges = [repeated_index[i, :, :, :] for i in range(self.start, self.start+n_situations)]
            else:
                situations_each = [indices[i] for i in range(self.start, self.start+n_situations) for j in range(0,n_timestamps-self._target-self._intro, self._step)]
                # Assign features as voltages over a sliding window
                self.features = [voltages_def[i, :, j:j+self._intro] for i in range(self.start, self.start+n_situations) for j in range(0,n_timestamps-self._target-self._intro, self._step)]
                # Assign targets as voltages over the next target timestamps
                self.targets = [voltages_def[i, : ,j+self._intro:j+self._intro+self._target] for i in range(self.start, self.start+n_situations) for j in range(0,n_timestamps-self._target-self._intro, self._step)]
                # Assign edge weights as edge attributes over the introduction period
                self.edge_weights = [edge_attr[i][:,j:j+self._intro,:].reshape(self._intro,len(edge_index[0]), 2) for i in range(self.start, self.start+n_situations) for j in range(0,n_timestamps-self._target-self._intro, self._step)]
                div = int(len(self.features)/(n_situations ))
                repeated_index = [edge_index[j] for j in range(self.start, self.start+n_situations) for k in range(div) for i in range(self._intro)]
                repeated_index = np.array(repeated_index).reshape(((n_situations)*div,  self._intro, len(self.edge_index[0]),2))
                self.edges = [repeated_index[i, :, :, :] for i in range((n_situations)*div)]
        
        # If the problem type is classification
        elif (self.problem.split("_")[0]=='classification'):
            self._preprocess_targets()
            processed_targets = self.processed_targets[indices, :]
            situations_each = range(n_situations)
            # Assign features as voltages for each situation
            self.features = [voltages_def[i, :, :] for i in range(n_situations)]
            # Assign targets as bus types or bus numbers depending on the problem type
            self.targets = [processed_targets[i,:]  for i in range(n_situations)]
            # Assign edge weights as edge attributes for each situation
            self.edge_weights = [edge_attr[i][:,:,:].reshape(n_timestamps, len(self.edge_index[0]), 2) for i in range(n_situations)]
            # Repeat the edge index for each situation
            repeated_index = [edge_index[j] for j in range(n_situations) for i in range(n_timestamps)]
            repeated_index = np.array(repeated_index).reshape((n_situations, n_timestamps, len(self.edge_index[0]),2  ))
            self.edges = [repeated_index[i, :, :, :] for i in range(n_situations)]
        return situations_each


    def process(self):
        """
        Process the dataset.

        This method processes the dataset by iterating through each case folder,
        processing the data, checking nodes, transforming temporary data,
        updating voltages, and creating the edge attribute matrix.

        Returns:
            Tuple: Tuple containing voltage data, edge indices, and edge attributes.
        """
        for root, dirs, files in os.walk(self._natural_folder):
            for folder in dirs:
                folder_name = os.path.join(root, folder)
                print("Processing: ", folder)

                if (not os.path.exists(os.path.join(folder_name, 'info.csv'))):
                    print("Skipping ", folder)
                    continue
            
                self._process_one_case(folder_name)

                # We check if all the nodes present in the edges have voltage info and vice versa
                self._check_nodes()
                
                # We transform the nodes ids to correct numbering (easier)
                self._transform_temp()

                # We update the voltage matrix
                self._update_voltages()

                # We create the edge attribute matrix, corresponding to the edge_index_ordering
                self._update_mix_weights()
                self.edge_index.append([[i[0], i[1]] for i in self.edge_index_temp])
    
        
        self.processed = True
        return self.voltages, self.edge_index, self.edge_attr

    def get_dataset(self, target=50, intro=200, step=50, limit=None, one_ts_per_situation=False, start=0, type=None):
        """
        Get the processed dataset.

        Args:
            target (int): Target length.
            intro (int): Introduction length.
            step (int): Step length.
            limit (int): Limit for the number of situations.
            one_ts_per_situation (bool): Whether to consider only one timestamp per situation.

        Returns:
            DynamicGraphTemporalSignalLen: Processed dataset.
        """
        self._target = target
        self._intro = intro
        self._step = step
        self._type = type
        self._one_situation = one_ts_per_situation
        self.start = start
        if not self.processed:
            self.process()
        self._limit = len(next(iter(self.voltages.values()))) if limit is None else limit
        situations_each = self._get_targets_and_features()
        dataset = DynamicGraphTemporalSignalLen(self.edges, self.edge_weights, self.features, self.targets, "PowerGridDataset")
        return dataset, situations_each
