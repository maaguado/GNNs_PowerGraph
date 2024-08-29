import torch.nn.functional as F
import pandas as pd
import seaborn as sns
sns.set_palette("coolwarm_r")
import numpy as np
import os, sys
import itertools
import wandb
import random
import argparse


sys.path.insert(1, "/Users/maguado/Documents/UGR/Master/TFM/repo/GNNs_PowerGraph/TFM")
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable


from utils import powergrid


import torch
import torch.nn.functional as F
from  utils.stgcn import STConv
from utils.trainer import TrainerSTConv

import itertools


def entrenar_y_evaluar_modelos_stconv(param_grid, dataset, dataloader_params, num_early_stop, num_epochs, problem="", path_save_experiment=None):
    
    resultados_list = []
    
    n_div_bt = loader.div
    n_nodes =dataset.features[0].shape[0]
    n_target = dataset.targets[0].shape[1]
    n_features = dataset[0].x.shape[1]

    #Vamos a guardar el mejor modelo
    mejor_loss_test = float('inf')
    mejor_trainer = None
    mejores_parametros = None
    mejores_resultados = None
    
    device =torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    n_iter = 15 
    for _ in tqdm(range(n_iter)):
        # Selecciona aleatoriamente los parámetros
        out_channels = random.choice(param_grid["out_channels"])
        kernel_size = random.choice(param_grid["kernel_size"])
        hidden_channels = random.choice(param_grid["hidden_channels"])
        normalization = random.choice(param_grid["normalization"])
        try:
            print(f"Entrenando modelo con out_channels={out_channels}, kernel_size={kernel_size}, hidden_channels={hidden_channels}, normalization={normalization}")
            model_bt = RecurrentGCN(name="STConv", node_features=n_features, node_count=n_nodes, n_target=n_target, out_channels=out_channels,k=2, kernel_size=kernel_size, hidden_channels=hidden_channels, normalization=normalization)

            wandb.init(project='stconv_'+problem, entity='maragumar01')
            trainer_bt = TrainerSTConv(model_bt, dataset,device, f"../results/{problem}", dataloader_params)
            wandb.config.update({
                'out_channels': out_channels,
                'kernel_size': kernel_size,
                'hidden_channels': hidden_channels,
                'normalization': normalization
            })


            losses,eval_losses, r2scores  = trainer_bt.train(num_epochs=num_epochs, steps=50, num_early_stop=num_early_stop)
            r2score_tst,losses_tst, loss_nodes, _, _ = trainer_bt.test()
        
            results_intermedio = {
                "Out channels": out_channels,
                "Kernel size": kernel_size,
                "Hidden channels": hidden_channels,
                "Normalization": normalization,
                "loss_final": losses[-1],
                "r2_eval_final": np.mean(r2scores[-1]),
                "loss_eval_final": np.mean(eval_losses[-1]),
                "r2_test": np.mean(r2score_tst),
                "loss_test": np.mean(losses_tst),
                "loss_nodes": np.mean(loss_nodes, axis=0).tolist()
            }
            # Añade los resultados a la lista
            resultados_list.append(results_intermedio)
            wandb.log({"loss": losses[-1], "r2_eval": np.mean(r2scores[-1]), "loss_eval": np.mean(eval_losses[-1]), "r2_test": np.mean(r2score_tst), "loss_test": np.mean(losses_tst)})

            if np.mean(losses_tst) < mejor_loss_test:
                mejor_loss_test = np.mean(losses_tst)
                mejor_trainer = trainer_bt
                mejores_parametros = {"Out channels": out_channels, "Kernel size": kernel_size, "Hidden channels": hidden_channels, "Normalization": normalization}
                mejores_resultados = results_intermedio
                mejor_trainer.save_model(params=mejores_parametros, path_save_experiment= path_save_experiment)
            wandb.finish()
            print("Resultados intermedios: ", results_intermedio)
        except Exception as e:
            print(f"Error entrenando modelo con out_channels={out_channels}, kernel_size={kernel_size}, hidden_channels={hidden_channels}, normalization={normalization}, excepcion: {e}")
            wandb.finish()
    resultados_gt = pd.DataFrame(resultados_list)
    
    return mejor_trainer, mejores_parametros, mejores_resultados, resultados_gt



class RecurrentGCN(torch.nn.Module):
    def __init__(self, name, node_features, node_count, n_target, out_channels,k, kernel_size, hidden_channels, normalization="sym"):
        self.name  =name
        self.n_nodes = node_count
        self.n_target = n_target
        self.n_features = node_features
        self.out_channels = out_channels
        self.k = k
        self.kernel_size = kernel_size
        self.normalization = normalization
        self.hidden_channels = hidden_channels
        super(RecurrentGCN, self).__init__()
        self.recurrent = STConv(num_nodes=node_count,
                                in_channels=1,
                                hidden_channels=hidden_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                K=k,  # Grado del polinomio Chebyshev
                                normalization=normalization 
                            )

        self.linear = torch.nn.Linear(self.out_channels, self.n_target)


    def forward(self, x, edge_index, edge_weight):

        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = h.squeeze(0).mean(dim=0)
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

path = os.getcwd()

sys.path.insert(1, "/".join(path.split("/")[0:-1]))


folder_path = "/Users/maguado/Documents/UGR/Master/TFM/repo/GNNs_PowerGraph/TFM/datos"
results_save_path = "../results"
name_model = "STConv"
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
    "out_channels": [32, 64],
    "kernel_size": [3,5,7],
    "normalization": ["sym", "rw"],
    "hidden_channels": [16, 32, 64],
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
trainer_gt,params_gt, resultados_final_gt, resultados_gt = entrenar_y_evaluar_modelos_stconv(param_grid, dataset_gt, dataloader_params2, num_early_stop, num_epochs, problem=problem, path_save_experiment=path_save_experiment_gt)
losses_tst, r2score_tst, loss_nodes, predictions, real = trainer_gt.test()

resultados_gt.to_csv(path_save_experiment_gt, index=False)
trainer_gt.save_model(params=params_gt, path_save_experiment= path_save_experiment_gt)

