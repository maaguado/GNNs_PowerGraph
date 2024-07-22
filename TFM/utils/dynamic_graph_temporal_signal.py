import torch
import numpy as np
from typing import List, Union
from torch_geometric.data import Data
import torch_geometric_temporal.signal as signal


Edge_Indices = List[Union[np.ndarray, None]]
Edge_Weights = List[Union[np.ndarray, None]]
Node_Features = List[Union[np.ndarray, None]]
Targets = List[Union[np.ndarray, None]]
Additional_Features = List[np.ndarray]



class DynamicGraphTemporalSignalLen(signal.DynamicGraphTemporalSignal):
    def __init__(
        self,
        edge_indices: Edge_Indices,
        edge_weights: Edge_Weights,
        features: Node_Features,
        targets: Targets,
        name: str,
        ndiv: int = 1,
        **kwargs: Additional_Features
    ):
        super().__init__(edge_indices, edge_weights, features, targets, **kwargs)
        self.name = name
        self.ndiv = ndiv

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, time_index: Union[int, slice]):
        if isinstance(time_index, slice):
            snapshot = DynamicGraphTemporalSignalLen(
                self.edge_indices[time_index],
                self.edge_weights[time_index],
                self.features[time_index],
                self.targets[time_index],
                self.name,
                **{key: getattr(self, key)[time_index] for key in self.additional_feature_keys}
            )
        else:
            x = self._get_features(time_index)
            edge_index = self._get_edge_index(time_index)
            edge_weight = self._get_edge_weight(time_index)
            y = self._get_target(time_index)
            additional_features = self._get_additional_features(time_index)

            snapshot = Data(x=x, edge_index=edge_index, edge_attr=edge_weight,
                            y=y, **additional_features)
        return snapshot
