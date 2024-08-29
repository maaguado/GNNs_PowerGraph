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
from utils.astgcn import ASTGCN
from utils.trainer import TrainerMSTGCN


def entrenar_y_evaluar_modelos_astgcn(param_grid, dataset, dataloader_params, num_early_stop, num_epochs, problem="", device = torch.device("cpu"), path_save_experiment=None):
    resultados_list = []

    
    n_nodes = dataset.features[0].shape[0]
    n_target = dataset.targets[0].shape[1]
    n_features = dataset[0].x.shape[1]

    mejor_loss_test = float('inf')
    mejor_trainer = None
    mejores_parametros = None
    mejores_resultados = None
    

    for nb_block, filter_, time_strides in tqdm(list(itertools.product(param_grid['nb_block'], param_grid['filter'], param_grid['time_strides']))):
        try:  
            print(f"Entrenando modelo con nb_block={nb_block}, nb_chev_filter={filter_}, nb_time_filter={filter_}, time_strides={time_strides}")        
            wandb.init(project='astgcn_'+problem, entity='maragumar01')
            wandb.config.update({
                'nb_block': nb_block,
                'filter_size': filter_,
                'time_strides': time_strides
            })

            model = RecurrentGCN(name="ASTGCN", node_features=n_features, node_count=n_nodes, n_target=n_target, nb_block=nb_block, k=2, nb_chev_filter = filter_, nb_time_filter =filter_, time_strides = time_strides)
            
            trainer = TrainerMSTGCN(model, dataset, device, f"../results/{problem}", dataloader_params)

            losses, eval_losses, r2scores = trainer.train(num_epochs=num_epochs, steps=200, num_early_stop=num_early_stop)
            r2score_tst, losses_tst, loss_nodes, _, _ = trainer.test()
        
            results_intermedio = {
                "nb_block": nb_block,
                "nb_chev_filter": filter_,
                "nb_time_filter": filter_,
                "time_strides": time_strides,
                "loss_final": losses[-1],
                "r2_eval_final": np.mean(r2scores[-1]),
                "loss_eval_final": np.mean(eval_losses[-1]),
                "r2_test": np.mean(r2score_tst),
                "loss_test": np.mean(losses_tst),
                "loss_nodes": np.mean(loss_nodes, axis=0).tolist()
            }
            wandb.log({"loss": losses[-1], "r2_eval": np.mean(r2scores[-1]), "loss_eval": np.mean(eval_losses[-1]), "r2_test": np.mean(r2score_tst), "loss_test": np.mean(losses_tst)})
            resultados_list.append(results_intermedio)

            if np.mean(losses_tst) < mejor_loss_test:
                mejor_loss_test = np.mean(losses_tst)
                mejor_trainer = trainer
                mejores_parametros = {
                    "nb_block": nb_block,
                    "nb_chev_filter": filter_,
                    "nb_time_filter": filter_,
                    "time_strides": time_strides
                }
                mejores_resultados = results_intermedio
                mejor_trainer.save_model(params=mejores_parametros, path_save_experiment= path_save_experiment)

            print("Resultados intermedios: ", results_intermedio)
            wandb.finish()
        except Exception as e:
            print(f"Error entrenando modelo con nb_block={nb_block}, nb_chev_filter={filter_}, nb_time_filter={filter_}, time_strides={time_strides}: {e}")
            wandb.finish()
    resultados_gt = pd.DataFrame(resultados_list)
   
    return mejor_trainer, mejores_parametros, mejores_resultados, resultados_gt


class RecurrentGCN(torch.nn.Module):
    def __init__(self, name, node_features, node_count, n_target, nb_block, k=1, nb_chev_filter = 2, nb_time_filter =2, time_strides = 2):
        self.name  =name
        self.n_nodes = node_count
        self.n_target = n_target
        self.n_features = node_features
        self.nb_block = nb_block
        self.k = k
        self.nb_chev_filter = nb_chev_filter
        self.nb_time_filter = nb_time_filter
        self.time_strides = time_strides

    
        super(RecurrentGCN, self).__init__()
        self.recurrent = ASTGCN(in_channels=1, 
                                num_for_predict=n_target, 
                                len_input=node_features,
                                K=k, 
                                nb_block=nb_block, 
                                num_of_vertices=node_count,
                                nb_chev_filter=nb_chev_filter,
                                nb_time_filter=nb_time_filter, 
                                time_strides=time_strides)



    def forward(self, x, edge_index):
        h = self.recurrent(x, edge_index)
        h = F.relu(h)
        return h
    
import argparse
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


folder_path = "/usr/src/app/GNNs_PowerGraph/TFM/datos"
results_save_path = "../results"
name_model = "ASTGCN"
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
    "nb_block": [1, 2, 3],
    "filter": [2,4,8],
    "time_strides": [1, 2, 4,5]
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
trainer_gt,params_gt, resultados_final_gt, resultados_gt = entrenar_y_evaluar_modelos_astgcn(param_grid, dataset_gt, dataloader_params2, num_early_stop, num_epochs, problem=problem, path_save_experiment=path_save_experiment_gt)
losses_tst, r2score_tst, loss_nodes, predictions, real = trainer_gt.test()

resultados_gt.to_csv(path_save_experiment_gt, index=False)
trainer_gt.save_model(params=params_gt, path_save_experiment= path_save_experiment_gt)

