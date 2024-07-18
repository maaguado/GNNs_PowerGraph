import torch.nn.functional as F
import pandas as pd
import seaborn as sns
sns.set_palette("coolwarm_r")
import numpy as np
import os, sys
import itertools
import wandb
import argparse
import random 

import json

sys.path.insert(1, "/usr/src/app/GNNs_PowerGraph/TFM")
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable


from utils import powergrid


import torch
import torch.nn.functional as F

from utils.trainer import TrainerMTGNN
from torch_geometric_temporal.nn.attention import MTGNN

import itertools
def entrenar_y_evaluar_modelos_mtgnn(param_grid, dataset, dataloader_params, num_early_stop, num_epochs, problem="", device=torch.device("cpu"), path_save_experiment=None):
    resultados_list = []
    
    n_nodes = dataset.features[0].shape[0]
    n_target = dataset.targets[0].shape[1]
    n_features = dataset[0].x.shape[1]

    mejor_loss_test = float('inf')
    mejor_trainer = None
    mejores_parametros = None
    mejores_resultados = None

    n_iter = 50 

    for _ in tqdm(range(n_iter)):
        gcn_depth = random.choice(param_grid['gcn_depth'])
        conv_channels = random.choice(param_grid['conv_channels'])
        kernel_size = random.choice(param_grid['kernel_size'])
        dropout = random.choice(param_grid['dropout'])
        gcn_true = random.choice(param_grid['gcn_true'])
        build_adj = random.choice(param_grid['build_adj'])
        propalpha = random.choice(param_grid['propalpha'])
        out_channels = random.choice(param_grid['out_channels'])

        try:        
            gcn_depth, conv_channels, kernel_size, dropout, gcn_true, build_adj, propalpha, out_channels = config
            print(f"Entrenando modelo con gcn_depth={gcn_depth}, conv_channels={conv_channels}, kernel_size={kernel_size}, dropout={dropout}, gcn_true={gcn_true}, build_adj={build_adj}, propalpha={propalpha}, out_channels={out_channels}")
            wandb.init(project='mtgnn_'+problem, entity='maragumar01')
            model = RecurrentGCN(
                name="MTGNN", 
                node_count=n_nodes, 
                node_features=n_features, 
                n_target=n_target,
                conv_channels=conv_channels,
                residual_channels=conv_channels, 
                out_channels=out_channels,
                skip_channels=conv_channels // 2,  # Ejemplo de cómo definir skip channels
                end_channels=n_target,  # Para conectar con la salida
                gcn_depth=gcn_depth,
                kernel_size=kernel_size,
                dropout=dropout,
                gcn_true=gcn_true,
                build_adj=build_adj,
                propalpha=propalpha
            )
            wandb.config.update({
                'gcn_depth': gcn_depth,
                'conv_channels': conv_channels,
                'kernel_size': kernel_size,
                'dropout': dropout,
                'gcn_true': gcn_true,
                'build_adj': build_adj,
                'propalpha': propalpha,
                'out_channels': out_channels
            })
            trainer = TrainerMTGNN(model, dataset, device, f"../results/{problem}", dataloader_params)

            losses, eval_losses, r2scores = trainer.train(num_epochs=num_epochs, steps=200, num_early_stop=num_early_stop)
            r2score_tst, losses_tst, loss_nodes, _, _ = trainer.test()

            results_intermedio = {
                "gcn_depth": gcn_depth,
                "conv_channels": conv_channels,
                "kernel_size": kernel_size,
                "dropout": dropout,
                "gcn_true": gcn_true,
                "build_adj": build_adj,
                "propalpha": propalpha,
                "out_channels": out_channels,
                "loss_final": losses[-1],
                "r2_eval_final": np.mean(r2scores[-1]),
                "loss_eval_final": np.mean(eval_losses[-1]),
                "r2_test": np.mean(r2score_tst),
                "loss_test": np.mean(losses_tst),
                "loss_nodes": np.mean(loss_nodes, axis=0).tolist()
            }
            
            resultados_list.append(results_intermedio)
            wandb.log({"loss": losses[-1], "r2_eval": np.mean(r2scores[-1]), "loss_eval": np.mean(eval_losses[-1]), "r2_test": np.mean(r2score_tst), "loss_test": np.mean(losses_tst)})

            if np.mean(losses_tst) < mejor_loss_test:
                mejor_loss_test = np.mean(losses_tst)
                mejor_trainer = trainer
                mejores_parametros = {
                    "gcn_depth": gcn_depth,
                    "conv_channels": conv_channels,
                    "kernel_size": kernel_size,
                    "dropout": dropout,
                    "gcn_true": gcn_true,
                    "build_adj": build_adj,
                    "propalpha": propalpha,
                    "out_channels": out_channels
                }
                mejores_resultados = results_intermedio
                mejor_trainer.save_model(params=mejores_parametros, path_save_experiment= path_save_experiment)
            print("Resultados intermedios: ", results_intermedio)
            wandb.finish()
        except Exception as e:
            print(f"Error entrenando modelo con gcn_depth={gcn_depth}, conv_channels={conv_channels}, kernel_size={kernel_size}, dropout={dropout}, gcn_true={gcn_true}, build_adj={build_adj}, propalpha={propalpha}, out_channels={out_channels}, exception: {e}")
            wandb.finish()
    resultados_gt = pd.DataFrame(resultados_list)
    
    return mejor_trainer, mejores_parametros, mejores_resultados, resultados_gt


class RecurrentGCN(torch.nn.Module):
    def __init__(self, name, node_count, node_features, n_target, conv_channels, residual_channels, out_channels,skip_channels, end_channels,dilation_exponential=1, kernel_size=5, layers=2, propalpha=0.5, tanhalpha=0.3, dropout=0.3, layer_norm_affine=True, subgraph_size=3, gcn_depth=2, gcn_true=True, build_adj=True):
        super(RecurrentGCN, self).__init__()
        self.name = name
        self.n_nodes = node_count
        self.n_features = node_features
        self.n_target = n_target
        self.out_channels = out_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.end_channels = end_channels

        self.dilation_exponential = dilation_exponential
        self.layers = layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.propalpha = propalpha
        self.tanhalpha = tanhalpha
        self.layer_norm_affine = layer_norm_affine
        self.kernel_set = [kernel_size] * conv_channels
        self.subgraph_size = subgraph_size

        self.gcn_true = gcn_true
        self.build_adj = build_adj
        self.gcn_depth = gcn_depth
        self.recurrent = MTGNN(gcn_true=gcn_true, 
                   build_adj=build_adj, 
                   gcn_depth =gcn_depth, 
                   num_nodes=node_count, 
                   kernel_set = self.kernel_set, 
                   kernel_size =kernel_size, 
                   dropout=dropout, 
                   subgraph_size=subgraph_size, 
                   node_dim =1, 
                   dilation_exponential=dilation_exponential, 
                   conv_channels=conv_channels, 
                   residual_channels=residual_channels, 
                   skip_channels=skip_channels, 
                   end_channels=end_channels, 
                   seq_length=node_features, 
                   in_dim=1, out_dim=out_channels, 
                   layers=layers, propalpha=propalpha, 
                   tanhalpha=tanhalpha, 
                   layer_norm_affline=layer_norm_affine)
        
        self.linear = torch.nn.Linear(out_channels, n_target)

    def forward(self, x, matrix):
        h = self.recurrent(x, matrix)
        h = h.squeeze(0).squeeze(2).permute(1,0)
        h = self.linear(h)
        return h

parser = argparse.ArgumentParser()
parser.add_argument(
    '--problem',
    '-p',
    required=True,
    type=str,
    dest='problem',
    help='Name of the problem'
)
args = parser.parse_args()
problem = args.problem
# ------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------- #
#                             Read json parameters                             #
# ---------------------------------------------------------------------------- #




path = os.getcwd()

sys.path.insert(1, "/".join(path.split("/")[0:-1]))


folder_path = "/usr/src/app/GNNs_PowerGraph/TFM/datos"
results_save_path = "../results"
name_model = "MTGNN"
loader = powergrid.PowerGridDatasetLoader(folder_path, problem="regression")
_,_,_ =loader.process(verbose=False)
limit = 300

dataloader_params2 = {
            "batch_size": 5,
            "data_split_ratio": [0.7, 0.15, 0.15],
            "seed": 42,
            "keep_same": True,
            "use_batch":False
}
param_grid = {
    'gcn_depth': [1,2, 3],                  
    'conv_channels': [4, 8, 16], 
    'out_channels': [4, 8, 16],          
    'kernel_size': [3],                
    'dropout': [0.25],              
    'gcn_true': [True],                
    'build_adj': [True],              
    'propalpha': [0.15]            
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_early_stop = 10
num_epochs = 50
lr = 0.01
hidden_size =100


### Gen trip
print(f"Ajustando modelo para {problem}...")
dataset_gt, situations_gt = loader.get_dataset( target= 20, intro=100, step=20, one_ts_per_situation=False, start = 1, type=problem)
n_div_gt = loader.div
n_nodes =dataset_gt.features[0].shape[0]
n_target = dataset_gt.targets[0].shape[1]
n_features = dataset_gt[0].x.shape[1]

path_save_experiment_gt = results_save_path+f"/{problem}"+ f"/ajustes/{name_model}_results.csv"

trainer_gt,params_gt, resultados_final_gt, resultados_gt = entrenar_y_evaluar_modelos_mtgnn(param_grid, dataset_gt, dataloader_params2, num_early_stop, num_epochs, problem=problem, path_save_experiment=path_save_experiment_gt)
losses_tst, r2score_tst, loss_nodes, predictions, real = trainer_gt.test()

resultados_gt.to_csv(path_save_experiment_gt, index=False)
trainer_gt.save_model(params=params_gt, path_save_experiment= path_save_experiment_gt)

