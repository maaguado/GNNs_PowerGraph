
import os
import json
from sklearn.metrics import r2_score
from utils import pygt_loader
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import shutil
import numpy as np
import pandas as pd


def train_test_val_split(dataset, data_split_ratio, random_seed=0, batch_size=64, keep_same=False, use_batch=False):
    
    
    train_ratio = data_split_ratio[0]
    val_ratio = data_split_ratio[1]
    test_ratio = data_split_ratio[2]
    if train_ratio + val_ratio + test_ratio != 1:
        raise ValueError("The sum of the ratios must be 1")
    

    torch.manual_seed(random_seed)
    
    total_series = len(dataset.features)
    train_size = int(train_ratio * total_series)
    val_size = int(val_ratio * total_series)

    if (keep_same):
    # Calcular el índice de la serie temporal más cercana al índice que corresponde al ratio de entrenamiento
        train_index = round(train_ratio * total_series)


        # Ajustar el índice de entrenamiento si es necesario para asegurar que se mantienen separadas las simulaciones
        if train_index % 15 > 7:
            train_index += 15 - (train_index % 15)  # Si está más cerca del final de una simulación, avanzamos a la siguiente simulación completa
        else:
            train_index -= train_index % 15  # Si está más cerca del inicio de una simulación, retrocedemos al inicio de la simulación anterior

        # Calcular los índices de validación y test - comprobamos si es múltiplo de 15 y ajustamos si es necesario
        val_index = train_index + val_size
        if val_index % 15 > 7:
            val_index += 15 - (val_index % 15)  # Si está más cerca del final de una simulación, avanzamos a la siguiente simulación completa
        else:
            val_index -= val_index % 15
    else:
        train_index = train_size
        val_index = train_size + val_size
    
    train_dataset = dataset[:train_index]
    val_dataset = dataset[train_index:val_index]
    test_dataset = dataset[val_index:]

    
    print("\n==================== DATASET INFO ===================\n")
    print(f"Train dataset: {len(train_dataset)}")
    print(f"Validation dataset: {len(val_dataset)}")
    print(f"Test dataset: {len(test_dataset)}")
    if(use_batch):
        dataloaders = {}
        dataloaders['train'] = pygt_loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
        dataloaders['val'] = pygt_loader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        dataloaders['test'] = pygt_loader.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        
        print("\n==================== DATALOADER INFO ===================\n")
        for split, dataloader in dataloaders.items():
            print(f"DataLoader for {split} set:")
            print(f"Number of batches: {len(dataloader)}")
            
        return dataloaders
    else:
        return {'train':train_dataset, 'val':val_dataset, 'test':test_dataset}




class TrainerModel(object):
    def __init__(self, model, dataset,device, save_dir, dataloader_params):
        self.model = model
        self.dataset = dataset 
        self.loader = train_test_val_split(dataset, batch_size=dataloader_params["batch_size"],
                                                   data_split_ratio=dataloader_params["data_split_ratio"],
                                                   random_seed=dataloader_params["seed"],
                                                   keep_same=dataloader_params["keep_same"],
                                                   use_batch=dataloader_params["use_batch"])
        self.device = device
        self.optimizer = None
        self.name = model.name
        self.h = None
        self.c = None
        self.save_path = save_dir
        self.resultados_final = {"Modelo": self.name, 
                            "Params": None, 
                           "Fichero_resultados_experimento": None, 
                            "Loss_tst": 0,
                            "R2_tst": 0,
                            "Loss_nodes": 0,
                            "R2_eval": 0,
                            "Loss_eval": 0,
                            "Loss_final": 0}

    def __loss__(self, logits, labels):
        return F.mse_loss(logits, labels)
    

    def _train_loop_snap(self):
        counter = 0
        accumulated_loss = 0.0
        losses = []
        for snapshot in self.loader["train"]:
            snapshot = snapshot.to(self.device)
            loss = self._train_snap(snapshot)

            loss.backward()
            accumulated_loss += loss.item()
            counter += 1

            if counter % self.steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses.append(accumulated_loss / self.steps)
                accumulated_loss = 0.0
        if counter % self.steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            losses.append(accumulated_loss / (counter % self.steps))
        return losses


    def train(self, num_epochs, num_early_stop, batch, steps=50):
 
        self.steps = steps
        self.batch = batch
        self.num_epochs = num_epochs
        self.optimizer = Adam(self.model.parameters())
        lr_schedule = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.1, patience=2, min_lr=0.001
        )
        self.model.to(self.device)
        best_eval_r2score = -100.0
        best_eval_loss = 0.0
        early_stop_counter = 0
        eval_losses, r2scores = [], []
        losses = []
        for epoch in range(num_epochs):
            is_best = False
            self.model.train()
            losses_epoch = []
            if batch:
                for batch in self.loader["train"]:
                    batch = batch.to(self.device)
                    loss = self._train_batch(batch)
                    losses_epoch.append(loss)
            else:
                losses_epoch = self._train_loop_snap()
            train_loss = torch.FloatTensor(losses_epoch).mean().item()
            losses.append(train_loss)

            ############### EVALUATION ################

            with torch.no_grad():
                eval_loss, eval_r2score = self.eval()
                eval_losses.append(eval_loss)
                r2scores.append(eval_r2score)

            print(f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Eval Loss: {eval_loss:.4f} | "
            f"Eval R2: {eval_r2score:.4f} | ")
            
            ############### EARLY STOPPING ################

            if num_early_stop > 0:
                    if eval_loss <= best_eval_loss:
                        best_eval_loss = eval_loss
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                    if epoch > num_epochs / 2 and early_stop_counter > num_early_stop:
                        break
            if lr_schedule:
                lr_schedule.step(eval_loss)
        self.resultados_final['loss_final'] = losses[-1]
        self.resultados_final['r2_eval_final'] = r2scores[-1]
        self.resultados_final['loss_eval_final'] = eval_losses[-1]
        return losses, eval_losses, r2scores


    def eval(self):
        self.model.to(self.device)
        self.model.eval()
        
        losses_eval, r2scores = [], []

        if self.batch:
            for batch in self.loader["eval"]:
                batch = batch.to(self.device)
                loss, r2_score = self._eval_batch(batch, test = False)
                r2scores.append(r2_score)
                losses_eval.append(loss)
        else:
            self.h = None
            self.c = None
            for snapshot in self.loader["eval"]:
                snapshot = snapshot.to(self.device)
                loss, r2_score = self._eval_snap(snapshot, test=False)
                r2scores.append(r2_score)
                losses_eval.append(loss)

    
        eval_loss = torch.FloatTensor(losses_eval).mean().item()
        eval_r2score = torch.FloatTensor(r2scores).mean().item()
        return eval_loss, eval_r2score
    


    def test(self, load_model = False):

        if load_model:
            state_dict = torch.load(os.path.join(self.save_path, self.name + ".pt"))

            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)

        self.model.eval()
        losses, r2scores, preds, loss_nodes = [], [], [], []
        if self.batch:
            for batch in self.loader["test"]:
                batch = batch.to(self.device)
                loss, r2, batch_preds, batch_real, loss_per_node = self._eval_batch(batch, test=True)
                preds.append(batch_preds)
                r2scores.append(r2)
                losses.append(loss)
                loss_nodes.append(loss_per_node)
                real.append(batch_real)
            else:
                for snapshot in self.loader["test"]:
                    snapshot = snapshot.to(self.device)
                    loss, r2, snapshot_preds, real, loss_per_node = self._eval_snap(snapshot)
                    preds.append(snapshot_preds)
                    r2scores.append(r2)
                    losses.append(loss)
                    loss_nodes.append(loss_per_node)
                    real.append(real)

        test_loss = torch.tensor(losses).mean().item()
        test_r2score = np.mean(r2scores)

        print(
            f"test loss: {test_loss:.6f}, test r2score {test_r2score:.6f}"
        )

    
        self.resultados_final['Loss_tst'] = np.mean(test_loss)
        self.resultados_final['R2_tst'] = np.mean(test_r2score)
        self.resultados_final['Loss_nodes'] = np.mean(loss_nodes, axis=0)
        
        return test_r2score, test_loss, loss_nodes, preds, real
    
    
    
    def _train_batch(self, batch):
        x = batch.x.view(len(batch), self.model.n_nodes, self.model.n_features).to(self.device)
        y = batch.y.to(self.device)

        if self.name in ['LSTM']:
            y_hat = self.model(x)
        if self.name == "AGCRN":
            y_hat, h = self.model(x, self.h)
            self.h = h
        loss = self.__loss__(y_hat.view(-1, self.model.n_target), y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()



    def _train_snap(self, snapshot):
        if self.name in ['LSTM']:
            x = snapshot.x.view( self.model.n_nodes, self.model.n_features)[None,:,:].to(self.device)
            y = snapshot.y[None,:,:].to(self.device)
            y_hat = self.model(x)
            
        
        if self.name == "DryGrEncoder":
            x = snapshot.x.to(self.device)
            edge_index = snapshot.edge_index.permute(2, 1, 0).to(self.device) 
            edge_attr = snapshot.edge_attr.permute(1, 0, 2).to(self.device) 
            y = snapshot.y.to(self.device) 
            y_hat, self.h, self.c = self.model(x, edge_index[:,:,0],edge_attr, self.h, self.c)
        loss = self.__loss__(y_hat, y)
        return loss


    def _eval_batch(self, batch, test = True):
        x = batch.x.view(len(batch), self.model.n_nodes, self.model.n_features)
        if self.name in ['LSTM']:
            y_hat = self.model(x)
        if self.name == "AGCRN":
            y_hat, h = self.model(x, self.h)
            self.h = h
        
        labels = batch.y
        logits = y_hat.view(-1, self.model.n_target)
           
        if test:
            r2 = r2_score(labels.cpu(), logits.cpu())
            loss = self.__loss__(logits, labels).item()
            loss_per_node = F.mse_loss(logits,labels , reduction='none')
            loss_per_node= loss_per_node.view(len(batch), self.model.n_nodes, self.model.n_target).mean(dim=0).mean(dim=1).cpu().detach().numpy()
            preds = y_hat.view(len(batch), self.model.n_nodes, self.model.n_target)
            real = batch.y.view(len(batch), self.model.n_nodes, self.model.n_target).cpu().detach().numpy()
            return loss, r2, preds, real, loss_per_node
        else:
            return loss, r2
        
        


    def _eval_snap(self, snapshot, test=True):
        if self.name in ['LSTM']:
            x = snapshot.x.view(self.model.n_nodes, self.model.n_features)[None,:,:].to(self.device)
            y = snapshot.y[None,:,:].to(self.device)
            y_hat = self.model(x)
            loss = F.mse_loss(y_hat, y).item()
            r2 = r2_score(y.squeeze(0).detach().cpu(), y_hat.squeeze(0).detach().cpu())
            loss_per_node = F.mse_loss(y_hat, y, reduction='none')
            loss_per_node = loss_per_node.view(1, self.model.n_nodes, self.model.n_target).mean(dim=0).mean(dim=1).cpu().detach().numpy()

        if self.name == "DryGrEncoder":
            x = snapshot.x.to(self.device)  
            edge_index = snapshot.edge_index.permute(2, 1, 0).to(self.device)  # [2, num_edges, num_time_steps]-> [2, 30, 100]
            edge_attr = snapshot.edge_attr.permute(1, 0, 2).to(self.device)  # [num_edges, num_time_steps, num_edge_features] -> [30, 100, 2]
            y = snapshot.y.to(self.device)
            y_hat, self.h, self.c = self.model(x, edge_index[:,:,0],edge_attr, self.h, self.c)
            loss = F.mse_loss(y_hat, y).item()
            loss_per_node = F.mse_loss(y_hat, y, reduction='none')
            loss_per_node = loss_per_node.view(1, self.model.n_nodes, self.model.n_target).mean(dim=1).cpu().detach().numpy()
        
        if test:
            preds = y_hat.view(1, self.model.n_nodes, self.model.n_target)
            real = y.view(1, self.model.n_nodes, self.model.n_target).cpu().detach().numpy()
            return loss, r2, preds, real, loss_per_node
        else:
            return loss, r2



    def save_model(self):
        print("\n==================== GUARDANDO RESULTADOS ===================\n")
 
        path_modelo = os.path.join(self.save_path, self.name + ".pt")
        path_general_problem = os.path.join(self.save_path, "results.csv")
        torch.save(self.model.state_dict(), path_modelo)


        if os.path.exists(path_general_problem):
            df = pd.read_csv(path_general_problem)
        else:
            # Crear un DataFrame vacío con las columnas del diccionario
            df = pd.DataFrame(columns=self.resultados_final.keys())

        new_data_df = pd.DataFrame([self.resultados_final])

        df = pd.concat([df, new_data_df], ignore_index=True)
        print(df)
        df.to_csv(path_general_problem, index=False)

        print("\n==================== RESULTADOS GUARDADOS ===================\n")
    