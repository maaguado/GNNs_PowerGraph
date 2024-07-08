
import os
import json
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from utils import pygt_loader
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import shutil
import numpy as np
import pandas as pd
from utils.train_lstm import  LSTMModel


def train_test_val_split(dataset, data_split_ratio, random_seed=0, batch_size=64, keep_same=False, use_batch=False, verbose=True):
    
    
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

    if verbose:
        print("\n==================== DATASET INFO ===================\n")
        print(f"Train dataset: {len(train_dataset)}")
        print(f"Validation dataset: {len(val_dataset)}")
        print(f"Test dataset: {len(test_dataset)}")
    if(use_batch):
        dataloaders = {}
        dataloaders['train'] = pygt_loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
        dataloaders['val'] = pygt_loader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        dataloaders['test'] = pygt_loader.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        
        if verbose:
            print("\n==================== DATALOADER INFO ===================\n")
            for split, dataloader in dataloaders.items():
                print(f"DataLoader for {split} set:")
                print(f"Number of batches: {len(dataloader)}")
            
        return dataloaders
    else:
        return {'train':train_dataset, 'val':val_dataset, 'test':test_dataset}




class TrainerModel(object):
    def __init__(self, model, dataset, device, save_dir, dataloader_params, verbose=True, is_classification=False):
        self.model = model
        self.dataset = dataset
        self.loader = train_test_val_split(dataset, batch_size=dataloader_params["batch_size"],
                                           data_split_ratio=dataloader_params["data_split_ratio"],
                                           random_seed=dataloader_params["seed"],
                                           keep_same=dataloader_params["keep_same"],
                                           use_batch=dataloader_params["use_batch"], verbose=verbose)
        self.device = device
        self.is_classification = is_classification
        self.optimizer = None
        self.name = model.name
        self.save_path = save_dir
        self.resultados_final = {"Modelo": self.name,
                                 "Params": None,
                                 "Fichero_resultados_experimento": None,
                                 "Loss_tst": 0,
                                 "Loss_eval": 0,
                                 "Loss_final": 0}

        if self.is_classification:
            self.resultados_final.update({"Accuracy_eval": 0, "Precision_eval": 0, "Recall_eval": 0, "F1_eval": 0, "Accuracy_tst": 0, "Precision_tst": 0, "Recall_tst": 0, "F1_tst": 0})
        else:
            self.resultados_final.update({"R2_tst": 0, "R2_eval": 0, "Loss_nodes": 0})


    def __loss__(self, logits, labels):
        if self.is_classification:
            return F.cross_entropy(logits, labels)
        return F.mse_loss(logits, labels)
    

    def _train_loop(self):
        pass


    def _eval_loop(self, test):
        pass


    def train(self, num_epochs, steps = None, num_early_stop=0):
 
        if steps is not None:
            self.steps = steps
        else:
            self.steps = len(self.loader["train"]) if self.batch else -1

        self.num_epochs = num_epochs
        self.optimizer = Adam(self.model.parameters())
        lr_schedule = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.1, patience=5, min_lr=0.001
        )
    
        self.model.to(self.device)
        best_eval_loss = 0.0
        early_stop_counter = 0

        eval_losses = []
        if self.is_classification:
            accs, precisions, recalls, f1s = [], [], [], []
        else:
            r2scores = []
        losses = []
        print("\n==================== TRAIN INFO ===================\n")
        for epoch in range(num_epochs):
            is_best = False
            self.model.train()
            losses_epoch = self._train_loop()
            train_loss = torch.FloatTensor(losses_epoch).mean().item()
            losses.append(train_loss)

            ############### EVALUATION ################

            with torch.no_grad():
                if self.is_classification:
                    eval_loss, acc, precision, recall, f1 = self.eval()
                    eval_losses.append(eval_loss)
                    accs.append(acc)
                    precisions.append(precision)
                    recalls.append(recall)
                    f1s.append(f1)
                else:
                    eval_loss, eval_r2score = self.eval()
                    eval_losses.append(eval_loss)
                    r2scores.append(eval_r2score)

            lr = self.optimizer.param_groups[0]['lr']
            result_str = f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f} | "

            if self.is_classification:
                result_str += (f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | "
                            f"F1-Score: {f1:.4f} | LR: {lr:.4f} | ")
            else:
                result_str += f"Eval R2: {eval_r2score:.4f} | LR: {lr:.4f} | "
            print(result_str)
            
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
        self.resultados_final['Loss_final'] = losses[-1]
        
        self.resultados_final['Loss_eval'] = eval_losses[-1]
        if self.is_classification:
            self.resultados_final["Accuracy_eval"] = accs[-1]
            self.resultados_final["Precision_eval"] = precisions[-1]
            self.resultados_final["Recall_eval"] = recalls[-1]
            self.resultados_final["F1_eval"] = f1s[-1]
            return losses, eval_losses, accs, precisions, recalls, f1s
        else:
            self.resultados_final['R2_eval'] = r2scores[-1]
            return losses, eval_losses, r2scores


    def eval(self):
        self.model.to(self.device)
        self.model.eval()
        if self.is_classification:
            losses_eval, acc, precision, recall, f1 = self._eval_loop(test=False)
            eval_loss = torch.FloatTensor(losses_eval).mean().item()
            return eval_loss, acc, precision, recall, f1
        else:
            losses_eval, r2scores = self._eval_loop(test=False)
            eval_loss = torch.FloatTensor(losses_eval).mean().item()
            eval_r2score = torch.FloatTensor(r2scores).mean().item()
            return eval_loss, eval_r2score
    


    def test(self, load_model = False, target_names = None):

        if load_model:
            state_dict = torch.load(os.path.join(self.save_path, self.name + ".pt"))

            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)

        print("\n==================== TEST INFO ===================\n")

        self.model.eval()
        if self.is_classification:
            losses, acc, precision, recall, f1, preds, real = self._eval_loop(test=True)

            self.resultados_final["Accuracy_tst"] = acc
            self.resultados_final["Precision_tst"] = precision
            self.resultados_final["Recall_tst"] = recall
            self.resultados_final["F1_tst"] = f1
            print(classification_report(real, preds, target_names=target_names, zero_division=0)) 
        else:
            losses, r2scores, loss_nodes, preds, real = self._eval_loop(test=True)
            test_r2score = np.mean(r2scores)
            self.resultados_final['R2_tst'] = np.mean(test_r2score)
            self.resultados_final['Loss_nodes'] = np.mean(loss_nodes, axis=0)

        print("preds: ", preds[0].shape)
        test_loss = torch.tensor(losses).mean().item()
        self.resultados_final['Loss_tst'] = test_loss

        if self.is_classification:
            print(f"test loss: {test_loss:.6f}, "
                f"test accuracy: {acc:.4f}, "
                f"test precision: {precision:.4f}, "
                f"test recall: {recall:.4f}, "
                f"test F1-score: {f1:.4f}")
        else:
            print(f"test loss: {test_loss:.6f}, test R2 score: {test_r2score:.6f}")

        return (acc,precision,recall,f1, test_loss, preds, real) if self.is_classification else (test_r2score, test_loss, loss_nodes, preds, real)

            
    def save_model(self, path_save_experiment=None, params=None):

        self.resultados_final['Params'] = params
        self.resultados_final['Fichero_resultados_experimento'] = path_save_experiment
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
    


class TrainerLSTMModel(TrainerModel):
    def __init__(self, model, dataset, device, save_dir, dataloader_params, batch=True, verbose=True, is_classification=False):
        super().__init__(model, dataset, device, save_dir, dataloader_params, verbose=verbose, is_classification=is_classification)
        self.is_classification = is_classification
        self.batch = batch
        if dataloader_params['use_batch'] and not batch:
            print("WARNING: The model is not batched but the dataloader is batched. Changing to batched model.")
            self.batch = True
        if not dataloader_params['use_batch'] and batch:
            print("WARNING: The model is batched but the dataloader is not batched. Changing to not batched model.")
            self.batch = False
        
    def _train_loop(self):
        losses_epoch = []
        if self.batch:
            for batch in self.loader["train"]:
                batch = batch.to(self.device)
                loss = self._train_batch(batch)
                losses_epoch.append(loss)
        else:
            counter = 0
            accumulated_loss = 0.0
            for snapshot in self.loader["train"]:
                snapshot = snapshot.to(self.device)
                loss = self._train_snap(snapshot)

                loss.backward()
                accumulated_loss += loss.item()
                counter += 1
                if counter % self.steps == 0:
                    #print("Step", counter, "updating...")
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    losses_epoch.append(accumulated_loss / self.steps)
                    accumulated_loss = 0.0
            if counter % self.steps != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses_epoch.append(accumulated_loss / (counter % self.steps))
        return losses_epoch


    def _eval_loop(self, test):
        losses_eval, preds, real = [], [], []
        if self.is_classification:
            accs, precisions, recalls, f1s = [], [], [], []
        else:
            r2scores, loss_per_node = [], []
        evaluation_set = "test" if test else "val"
        for item in self.loader[evaluation_set]:
            item = item.to(self.device)
            if test:
                if self.is_classification:
                    loss, preds_prelim, real_prelim = self._eval_batch(item, test = test) if self.batch else self._eval_snap(item, test = test)
                else:
                    loss, r2_score, preds_prelim, real_prelim, loss_per_node_prelim = self._eval_batch(item, test = test) if self.batch else self._eval_snap(item, test = test)
                    r2scores.append(r2_score)
                    loss_per_node.append(loss_per_node_prelim)
                preds.append(preds_prelim)
                real.append(real_prelim)
                losses_eval.append(loss)
            else:   
                
                if self.is_classification:
                    loss, preds_prelim, real_prelim = self._eval_batch(item, test = test) if self.batch else self._eval_snap(item, test = test)
                else:
                    loss, r2_score = self._eval_batch(item, test = test) if self.batch else self._eval_snap(item, test = test)
                    r2scores.append(r2_score)
                preds.append(preds_prelim)
                real.append(real_prelim)
                losses_eval.append(loss)

        if self.is_classification:
            real = np.concatenate(real).flatten() if self.batch else real
            preds = np.concatenate(preds).flatten()if self.batch else preds
            acc = accuracy_score(real, preds)
            precision = precision_score(real, preds, average='macro', zero_division=0)
            recall = recall_score(real, preds, average='macro', zero_division=0)
            f1 = f1_score(real, preds, average='macro', zero_division=0)
            return (losses_eval, acc, precision, recall, f1, preds, real) if test else (losses_eval, acc, precision, recall, f1)
        else:
            return (losses_eval, r2scores, loss_per_node, preds, real) if test else (losses_eval, r2scores)
    


    def _train_batch(self, batch):
        x = batch.x.view(len(batch), self.model.n_nodes, self.model.n_features).to(self.device)
        if self.is_classification:
            y = batch.y.to(self.device).view(-1, self.model.n_target)
        else:
            y = batch.y.to(self.device)
        y_hat = self.model(x)
        loss = self.__loss__(y_hat.view(-1, self.model.n_target), y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def _train_snap(self, snapshot):
        x = snapshot.x.view( self.model.n_nodes, self.model.n_features)[None,:,:].to(self.device)
        y = snapshot.y[None,:,:].to(self.device) if not self.is_classification else snapshot.y.to(self.device)
        y_hat = self.model(x).squeeze(0)
        loss = self.__loss__(y_hat, y)
        return loss

        

    def _eval_batch(self, batch, test):
        x = batch.x.view(len(batch), self.model.n_nodes, self.model.n_features).to(self.device)
        y = batch.y.to(self.device)

        y_hat = self.model(x)
        
        if self.is_classification:
            logits = y_hat.view(-1, self.model.n_target)
            y = y.view(-1, self.model.n_target)
            loss = self.__loss__(logits, y).item()
            preds = logits.argmax(dim=1).cpu().detach().numpy()
            real = y.argmax(dim=1).cpu().detach().numpy()
            return (loss,preds, real)
        else:
            logits = y_hat.view(-1, self.model.n_target)
            loss = self.__loss__(logits, y).item()
            r2 = r2_score(y.cpu(), logits.cpu())
            if test:
                loss_per_node = F.mse_loss(logits,batch.y , reduction='none')
                loss_per_node= loss_per_node.view(len(batch), self.model.n_nodes, self.model.n_target).mean(dim=0).mean(dim=1).cpu().detach().numpy()
                preds = y_hat.view(len(batch), self.model.n_nodes, self.model.n_target).cpu().detach().numpy()
                real = batch.y.view(len(batch), self.model.n_nodes, self.model.n_target).cpu().detach().numpy()
                return loss, r2, preds, real, loss_per_node
            else:
                return loss, r2
            
    def _eval_snap(self, snapshot, test):
        x = snapshot.x.view(self.model.n_nodes, self.model.n_features)[None,:,:].to(self.device)
        y = snapshot.y.to(self.device)
        y_hat = self.model(x)
        
        if self.is_classification:
            logits = y_hat.squeeze(0)
            loss = self.__loss__(logits, y).item()
            preds = logits.argmax(dim=0).cpu().detach()
            real = y.squeeze(0).argmax(dim=0).cpu().detach()
            return loss, preds, real
        else:
            loss = F.mse_loss(y_hat, y).item()
            r2 = r2_score(y.cpu(), y_hat.cpu())
            loss_per_node = F.mse_loss(y_hat, y, reduction='none')
            loss_per_node = loss_per_node.view(1, self.model.n_nodes, self.model.n_output).mean(dim=0).mean(dim=1).cpu().detach().numpy()
            if test:
                preds = y_hat.view(1, self.model.n_nodes, self.model.n_target).cpu().detach().numpy()
                real = y.view(1, self.model.n_nodes, self.model.n_target).cpu().detach().numpy()
                return loss, r2, preds, real, loss_per_node
            else:
                return loss, r2
    

class TrainerDryGrEncoder(TrainerModel):
    def __init__(self, model, dataset,device, save_dir, dataloader_params, verbose=True):
        if (dataloader_params['use_batch']):
            print("WARNING: The model is not batched but the dataloader is batched. Changing to not batched dataset.")
            dataloader_params['use_batch'] = False
        super().__init__(model, dataset,device, save_dir, dataloader_params, verbose=verbose)
        
        self.h = None
        self.c = None


    def _train_loop(self):
        losses_epoch = []
        self.h = None
        self.c = None
        counter = 0
        accumulated_loss = 0.0
        for snapshot in self.loader["train"]:
            snapshot = snapshot.to(self.device)
            loss = self._train_snap(snapshot)

            loss.backward(retain_graph=False)
            accumulated_loss += loss.item()
            counter += 1
            if counter % self.steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses_epoch.append(accumulated_loss / self.steps)
                accumulated_loss = 0.0
        if counter % self.steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            losses_epoch.append(accumulated_loss / (counter % self.steps))
        return losses_epoch


    def _eval_loop(self, test):
        losses_eval, r2scores = [], []
        preds, real, loss_per_node = [], [], []
        evaluation_set = "test" if test else "val"
        self.h = None
        self.c = None

        for item in self.loader[evaluation_set]:
            item = item.to(self.device)
            if test:
                loss, r2_score, preds_prelim, real_prelim, loss_per_node_prelim = self._eval_snap(item, test = test)
                preds.append(preds_prelim)
                real.append(real_prelim)
                loss_per_node.append(loss_per_node_prelim)
            else:   
                loss, r2_score = self._eval_snap(item, test = test)
            r2scores.append(r2_score)
            losses_eval.append(loss)

        if test:
            return losses_eval, r2scores, loss_per_node, preds, real
        return losses_eval, r2scores
    


    

    def _train_snap(self, snapshot):
        x = snapshot.x.to(self.device) 
        edge_index = snapshot.edge_index.permute(2, 1, 0).to(self.device) 
        edge_attr = snapshot.edge_attr.permute(1, 0, 2).to(self.device) 
        y = snapshot.y.to(self.device) 
        y_hat, self.h, self.c = self.model(x, edge_index[:,:,0],edge_attr, self.h, self.c)
        self.h = self.h.detach()    
        self.c = self.c.detach()
        loss = self.__loss__(y_hat, y)
        return loss
        
    
    def _eval_snap(self, snapshot, test=True):
        
        x = snapshot.x.to(self.device)  
        edge_index = snapshot.edge_index.permute(2, 1, 0).to(self.device)  # [2, num_edges, num_time_steps]-> [2, 30, 100]
        edge_attr = snapshot.edge_attr.permute(1, 0, 2).to(self.device)  # [num_edges, num_time_steps, num_edge_features] -> [30, 100, 2]
        y = snapshot.y.to(self.device)
        y_hat, self.h, self.c = self.model(x, edge_index[:,:,0],edge_attr, self.h, self.c)
        loss = F.mse_loss(y_hat, y).item()
        loss_per_node = F.mse_loss(y_hat, y, reduction='none')
        loss_per_node = loss_per_node.view(1, self.model.n_nodes, self.model.n_target).mean(dim=1).cpu().detach().numpy()
        r2 = r2_score(y.detach().cpu(), y_hat.detach().cpu())
        if test:
            preds = y_hat.view(1, self.model.n_nodes, self.model.n_target).cpu().detach().numpy()
            real = y.view(1, self.model.n_nodes, self.model.n_target).cpu().detach().numpy()
            return loss, r2, preds, real, loss_per_node
        else:
            return loss, r2


class TrainerAGCRN(TrainerLSTMModel):
    def __init__(self, model, dataset,device, save_dir, dataloader_params, verbose=True, is_classification=False):
        if (not dataloader_params['use_batch']):
            print("WARNING: The model is batched but the dataloader is not batched. Changing to batched dataset.")
            dataloader_params['use_batch'] = True
        super().__init__(model, dataset,device, save_dir, dataloader_params, batch=True, verbose=verbose, is_classification=is_classification)
        self.h = None
        self.e = torch.empty(self.model.n_nodes, self.model.embedding_dim).to(self.device)
        torch.nn.init.xavier_uniform_(self.e)




    def _eval_loop(self, test):
        self.h = None
        return super()._eval_loop(test)
    


    def _train_batch(self, batch):
        x = batch.x.view(len(batch), self.model.n_nodes, self.model.n_features)
        y_hat, self.h = self.model(x, self.e, self.h)
        loss = self.__loss__(y_hat.view(-1, self.model.n_target), batch.y)
        loss.backward()
        loss = loss.detach()
        self.h = self.h.detach()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()



    def _eval_batch(self, batch, test):
        x = batch.x.view(len(batch), self.model.n_nodes, self.model.n_features)
        y_hat,self.h = self.model(x, self.e, self.h)
        loss = self.__loss__(y_hat.view(-1, self.model.n_target), batch.y).item()

        y = batch.y.view(-1, self.model.n_target)
        logits = y_hat.view(-1, self.model.n_target)
        if self.is_classification:
            loss = self.__loss__(logits, y).item()
            preds = logits.argmax(dim=1).cpu().detach().numpy()
            real = y.argmax(dim=1).cpu().detach().numpy()
            return (loss,preds, real)
        else:
            r2 = r2_score(y.cpu(), logits.cpu())
            if test:
                loss_per_node = F.mse_loss(logits,batch.y , reduction='none')
                loss_per_node= loss_per_node.view(len(batch), self.model.n_nodes, self.model.n_target).mean(dim=0).mean(dim=1).cpu().detach().numpy()
                preds = y_hat.view(len(batch), self.model.n_nodes, self.model.n_target).cpu().detach().numpy()
                real = batch.y.view(len(batch), self.model.n_nodes, self.model.n_target).cpu().detach().numpy()
                return loss, r2, preds, real, loss_per_node
            else:
                return loss, r2




class TrainerMPNNLSTM(TrainerModel):
    def __init__(self, model, dataset,device, save_dir, dataloader_params, verbose=True, is_classification=False):
        if (dataloader_params['use_batch']):
            print("WARNING: The model is not batched but the dataloader is batched. Changing to not batched dataset.")
            dataloader_params['use_batch'] = False
        super().__init__(model, dataset,device, save_dir, dataloader_params, verbose=verbose, is_classification=is_classification)
        

    def _train_loop(self):
        losses_epoch = []
        counter = 0
        accumulated_loss = 0.0
        for snapshot in self.loader["train"]:
            snapshot = snapshot.to(self.device)
            loss = self._train_snap(snapshot)

            loss.backward()
            accumulated_loss  += loss.item()
            counter += 1
            if counter % self.steps == 0:
                #print("Step", counter, "updating...")
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses_epoch.append(accumulated_loss / self.steps)
                accumulated_loss = 0.0
        if counter % self.steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            losses_epoch.append(accumulated_loss / (counter % self.steps))
        return losses_epoch


    def _eval_loop(self, test):
        losses_eval, preds, real = [], [], []
        if self.is_classification:
            accs, precisions, recalls, f1s = [], [], [], []
        else:
            r2scores, loss_per_node = [], []
        evaluation_set = "test" if test else "val"


        for item in self.loader[evaluation_set]:
            item = item.to(self.device)
            if test:
                if self.is_classification:
                    loss, preds_prelim, real_prelim = self._eval_snap(item, test = test)
                else:
                    loss, r2_score, preds_prelim, real_prelim, loss_per_node_prelim = self._eval_snap(item, test = test)
                    r2scores.append(r2_score)
                    loss_per_node.append(loss_per_node_prelim)
                preds.append(preds_prelim)
                real.append(real_prelim)
                losses_eval.append(loss)
            else:   
                if self.is_classification:
                    loss, preds_prelim, real_prelim = self._eval_snap(item, test = test)
                    preds.append(preds_prelim)
                    real.append(real_prelim)
                else:
                    loss, r2_score = self._eval_snap(item, test = test)
                    r2scores.append(r2_score)
                losses_eval.append(loss)
            
        if self.is_classification:
            acc = accuracy_score(real, preds)
            precision = precision_score(real, preds, average='macro', zero_division=0)
            recall = recall_score(real, preds, average='macro', zero_division=0)
            f1 = f1_score(real, preds, average='macro', zero_division=0)
            return (losses_eval, acc, precision, recall, f1, preds, real) if test else (losses_eval, acc, precision, recall, f1)
        else:
            return (losses_eval, r2scores, loss_per_node, preds, real) if test else (losses_eval, r2scores)
    



    def _train_snap(self, snapshot):
        x = snapshot.x.to(self.device) 
        edge_index = snapshot.edge_index.permute(2, 1, 0).to(self.device) 
        edge_attr = snapshot.edge_attr.permute(1, 0, 2).to(self.device) 
        y = snapshot.y.to(self.device) 
        y_hat= self.model(x, edge_index[:,:,0],edge_attr)
        loss = self.__loss__(y_hat, y)
        return loss
        
    
    def _eval_snap(self, snapshot, test=True):
        
        x = snapshot.x.to(self.device)  
        edge_index = snapshot.edge_index.permute(2, 1, 0).to(self.device)  # [2, num_edges, num_time_steps]-> [2, 30, 100]
        edge_attr = snapshot.edge_attr.permute(1, 0, 2).to(self.device)  # [num_edges, num_time_steps, num_edge_features] -> [30, 100, 2]
        y = snapshot.y.to(self.device)
        y_hat = self.model(x, edge_index[:,:,0],edge_attr)
        if self.is_classification:
            loss = self.__loss__(y_hat, y).item()
            preds = y_hat.argmax(dim=0).cpu().detach()
            real = y.squeeze(0).argmax(dim=0).cpu().detach()
            return loss, preds, real
        else:
            loss = F.mse_loss(y_hat, y).item()
            r2 = r2_score(y.cpu(), y_hat.cpu())
            loss_per_node = F.mse_loss(y_hat, y, reduction='none')
            loss_per_node = loss_per_node.view(1, self.model.n_nodes, self.model.n_output).mean(dim=0).mean(dim=1).cpu().detach().numpy()
            if test:
                preds = y_hat.view(1, self.model.n_nodes, self.model.n_target).cpu().detach().numpy()
                real = y.view(1, self.model.n_nodes, self.model.n_target).cpu().detach().numpy()
                return loss, r2, preds, real, loss_per_node
            else:
                return loss, r2
        

class TrainerA3TGCN(TrainerMPNNLSTM):
    def __init__(self, model, dataset,device, save_dir, dataloader_params, verbose=True):
        super().__init__(model, dataset,device, save_dir, dataloader_params, verbose=verbose)
      

class TrainerEvolveGCN(TrainerMPNNLSTM):
    def __init__(self, model, dataset,device, save_dir, dataloader_params, verbose=True):
        super().__init__(model, dataset,device, save_dir, dataloader_params, verbose=verbose)
    

class TrainerSTConv(TrainerMPNNLSTM):
    def __init__(self, model, dataset,device, save_dir, dataloader_params, verbose=True):
        super().__init__(model, dataset,device, save_dir, dataloader_params, verbose=verbose)

    def _train_snap(self, snapshot):
        x = snapshot.x.permute(1, 0)[None,:,:].unsqueeze(-1).to(self.device) 
        edge_index = snapshot.edge_index.permute(2, 1, 0).to(self.device) 
        edge_attr = snapshot.edge_attr.permute(1, 0, 2).to(self.device) 
        y = snapshot.y.to(self.device) 
        y_hat= self.model(x, edge_index[:,:,0],edge_attr)
        loss = self.__loss__(y_hat, y)
        return loss
    
    def _eval_snap(self, snapshot, test=True):
        x = snapshot.x.permute(1, 0)[None,:,:].unsqueeze(-1).to(self.device) 
        edge_index = snapshot.edge_index.permute(2, 1, 0).to(self.device)  # [2, num_edges, num_time_steps]-> [2, 30, 100]
        edge_attr = snapshot.edge_attr.permute(1, 0, 2).to(self.device)  # [num_edges, num_time_steps, num_edge_features] -> [30, 100, 2]
        y = snapshot.y.to(self.device)
        y_hat = self.model(x, edge_index[:,:,0],edge_attr)
        loss = F.mse_loss(y_hat, y).item()
        loss_per_node = F.mse_loss(y_hat, y, reduction='none')
        loss_per_node = loss_per_node.view(1, self.model.n_nodes, self.model.n_target).mean(dim=1).cpu().detach().numpy()
        r2 = r2_score(y.detach().cpu(), y_hat.detach().cpu())
        if test:
            preds = y_hat.view(1, self.model.n_nodes, self.model.n_target).cpu().detach().numpy()
            real = y.view(1, self.model.n_nodes, self.model.n_target).cpu().detach().numpy()
            return loss, r2, preds, real, loss_per_node
        else:
            return loss, r2
        

class TrainerMSTGCN(TrainerMPNNLSTM):
    def __init__(self, model, dataset,device, save_dir, dataloader_params, verbose=True):
        super().__init__(model, dataset,device, save_dir, dataloader_params, verbose=verbose)

    def _train_snap(self, snapshot):
        x = snapshot.x[None, None,:,:].permute(0,2,1,3).to(self.device)
        edge_index = snapshot.edge_index[0,:,:].permute(1,0).to(self.device)
        y = snapshot.y.to(self.device)
        y_hat= self.model(x, edge_index).squeeze(0)
        loss = self.__loss__(y_hat, y)
        return loss
    
    def _eval_snap(self, snapshot, test=True):
        x = snapshot.x[None, None,:,:].permute(0,2,1,3).to(self.device)
        edge_index = snapshot.edge_index[0,:,:].permute(1,0).to(self.device)
        y = snapshot.y.to(self.device)
        y_hat = self.model(x, edge_index).squeeze(0)
        loss = F.mse_loss(y_hat, y).item()
        loss_per_node = F.mse_loss(y_hat, y, reduction='none')
        loss_per_node = loss_per_node.view(1, self.model.n_nodes, self.model.n_target).mean(dim=1).cpu().detach().numpy()
        r2 = r2_score(y.detach().cpu(), y_hat.detach().cpu())
        if test:
            preds = y_hat.view(1, self.model.n_nodes, self.model.n_target).cpu().detach().numpy()
            real = y.view(1, self.model.n_nodes, self.model.n_target).cpu().detach().numpy()
            return loss, r2, preds, real, loss_per_node
        else:
            return loss, r2
        

from torch_geometric.utils import to_scipy_sparse_matrix



class TrainerMTGNN(TrainerMPNNLSTM):
    def __init__(self, model, dataset,device, save_dir, dataloader_params, verbose=True):
        super().__init__(model, dataset,device, save_dir, dataloader_params, verbose=verbose)

    def _train_snap(self, snapshot):
        x = snapshot.x[None,None,:,:].to(self.device)
        ei = snapshot.edge_index[0,:,:].permute(1,0).to(self.device)
        ea = snapshot.edge_attr[0,:,:].mean(1).to(self.device)
        matrix = to_scipy_sparse_matrix(ei, ea)
        y = snapshot.y.to(self.device)
        y_hat= self.model(x, matrix)
        loss = self.__loss__(y_hat, y)
        return loss
    
    def _eval_snap(self, snapshot, test=True):
        x = snapshot.x[None,None,:,:].to(self.device)
        ei = snapshot.edge_index[0,:,:].permute(1,0).to(self.device)
        ea = snapshot.edge_attr[0,:,:].mean(1).to(self.device)
        matrix = to_scipy_sparse_matrix(ei, ea)
        y = snapshot.y.to(self.device)
        y_hat = self.model(x, matrix)
        loss = F.mse_loss(y_hat, y).item()
        loss_per_node = F.mse_loss(y_hat, y, reduction='none')
        loss_per_node = loss_per_node.view(1, self.model.n_nodes, self.model.n_target).mean(dim=1).cpu().detach().numpy()
        r2 = r2_score(y.detach().cpu(), y_hat.detach().cpu())
        if test:
            preds = y_hat.view(1, self.model.n_nodes, self.model.n_target).cpu().detach().numpy()
            real = y.view(1, self.model.n_nodes, self.model.n_target).cpu().detach().numpy()
            return loss, r2, preds, real, loss_per_node
        else:
            return loss, r2