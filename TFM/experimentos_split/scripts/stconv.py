import torch.nn.functional as F
import pandas as pd
import seaborn as sns
sns.set_palette("coolwarm_r")
import numpy as np
import os, sys
import itertools
import wandb

sys.path.insert(1, "/usr/src/app/GNNs_PowerGraph/TFM")
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


def entrenar_y_evaluar_modelos_stconv(param_grid, dataset, dataloader_params, num_early_stop, num_epochs, problem=""):
    
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

    for out_channels, kernel_size, hidden_channels, normalization in tqdm(list(itertools.product(param_grid["out_channels"], param_grid['kernel_size'],param_grid['hidden_channels'], param_grid['normalization']))):
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
        wandb.finish()
        print("Resultados intermedios: ", results_intermedio)

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

    
    


path = os.getcwd()

sys.path.insert(1, "/".join(path.split("/")[0:-1]))


folder_path = "/usr/src/app/GNNs_PowerGraph/TFM/datos"
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
    "out_channels": [16, 32, 64],
    "kernel_size": [3,5,7],
    "normalization": ["sym", "rw"],
    "hidden_channels": [16, 32, 64],
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_early_stop = 10
num_epochs = 1
lr = 0.01
hidden_size =100


### Gen trip
print("Ajustando modelo para gen_trip...")
problem_gt = "gen_trip"
dataset_gt, situations_gt = loader.get_dataset( target= 20, intro=100, step=20, one_ts_per_situation=False, start = 1, type=problem_gt)
n_div_gt = loader.div
n_nodes =dataset_gt.features[0].shape[0]
n_target = dataset_gt.targets[0].shape[1]
n_features = dataset_gt[0].x.shape[1]


trainer_gt,params_gt, resultados_final_gt, resultados_gt = entrenar_y_evaluar_modelos_stconv(param_grid, dataset_gt, dataloader_params2, num_early_stop, num_epochs, problem=problem_gt)
losses_tst, r2score_tst, loss_nodes, predictions, real = trainer_gt.test()

path_save_experiment_gt = results_save_path+f"/{problem_gt}"+ f"/ajustes/{name_model}_results.csv"
resultados_gt.to_csv(path_save_experiment_gt, index=False)
trainer_gt.save_model(params=params_gt, path_save_experiment= path_save_experiment_gt)


### Bus trip
print("Ajustando modelo para bus_trip...")
problem_bt = "bus_trip"
dataset_bt, situations_bt = loader.get_dataset( target= 20, intro=100, step=20, one_ts_per_situation=False, start = 1, type=problem_bt)
n_div_bt = loader.div

trainer_bt,params_bt,resultados_final_bt, resultados_bt = entrenar_y_evaluar_modelos_stconv(param_grid, dataset_bt, dataloader_params2, num_early_stop=num_early_stop, num_epochs=num_epochs, problem=problem_bt)
_, _, _, predictions_bt_ajuste, real_bt_ajuste = trainer_bt.test()
path_save_experiment_bt = results_save_path+f"/{problem_bt}"+ f"/ajustes/{name_model}_results.csv"
resultados_bt.to_csv(path_save_experiment_bt, index=False)
trainer_bt.save_model(path_save_experiment=path_save_experiment_bt, params=params_bt)


# Bus fault
print("Ajustando modelo para bus_fault...")
problem_bf = "bus_fault"
dataset_bf, situations_bf = loader.get_dataset( target= 20, intro=100, step=20, one_ts_per_situation=False, start = 1, type=problem_bf)
n_div_bf = loader.div

num_epochs = 100
num_early_stop = 10
trainer_bf,params_bf,resultados_final_bf, resultados_bf = entrenar_y_evaluar_modelos_stconv(param_grid, dataset_bf, dataloader_params2, num_early_stop=num_early_stop, num_epochs=num_epochs, problem=problem_bf)
_, _, _, predictions_bf_ajuste, real_bf_ajuste = trainer_bf.test()
path_save_experiment_bf = results_save_path+f"/{problem_bf}"+ f"/ajustes/{name_model}_results.csv"
resultados_bf.to_csv(path_save_experiment_bf, index=False)
trainer_bf.save_model(path_save_experiment=path_save_experiment_bf, params=params_bf)


# Branch fault
print("Ajustando modelo para branch_fault...")
problem_brf = "branch_fault"
dataset_brf, situations_brf = loader.get_dataset( target= 20, intro=100, step=20, one_ts_per_situation=False, start = 1, type=problem_brf)
n_div_brf = loader.div

trainer_brf,params_brf,resultados_final_brf, resultados_brf = entrenar_y_evaluar_modelos_stconv(param_grid, dataset_brf, dataloader_params2, num_early_stop=num_early_stop, num_epochs=num_epochs, problem=problem_brf)
_, _, _, predictions_brf_ajuste, real_brf_ajuste = trainer_brf.test()
path_save_experiment_brf = results_save_path+f"/{problem_brf}"+ f"/ajustes/{name_model}_results.csv"
resultados_brf.to_csv(path_save_experiment_brf, index=False)
trainer_brf.save_model(path_save_experiment=path_save_experiment_brf, params=params_brf)

# Branch trip
print("Ajustando modelo para branch_trip...")
problem_brt = "branch_trip"
dataset_brt, situations_brt = loader.get_dataset( target= 20, intro=100, step=20, one_ts_per_situation=False, start = 1, type=problem_brt)
n_div_brt = loader.div
trainer_brt,params_brt,resultados_final_brt, resultados_brt = entrenar_y_evaluar_modelos_stconv(param_grid, dataset_brt, dataloader_params2, num_early_stop=num_early_stop, num_epochs=num_epochs, problem=problem_brt)
_, _, _, predictions_brt_ajuste, real_brt_ajuste = trainer_brt.test()
path_save_experiment_brt = results_save_path+f"/{problem_brt}"+ f"/ajustes/{name_model}_results.csv"
resultados_brt.to_csv(path_save_experiment_brt, index=False)
trainer_brt.save_model(path_save_experiment=path_save_experiment_brt, params=params_brt)