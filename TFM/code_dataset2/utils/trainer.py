
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


def train_test_val_split(dataset, **kwargs):
    
    
    # Extraer los parámetros del diccionario kwargs
    train_ratio = kwargs.get('train_ratio', 0.8)
    val_ratio = kwargs.get('val_ratio', 0.1)
    test_ratio = kwargs.get('test_ratio', 0.1)
    random_seed = kwargs.get('random_seed', 0)
    batch_size = kwargs.get('batch_size', 64)

    torch.manual_seed(random_seed)
    
    total_series = len(dataset.features)
    train_size = int(train_ratio * total_series)
    val_size = int(val_ratio * total_series)

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
    
    train_dataset = dataset[:train_index]
    val_dataset = dataset[train_index:val_index]
    test_dataset = dataset[val_index:]

    dataloaders = {}
    dataloaders['train'] = pygt_loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    dataloaders['val'] = pygt_loader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    dataloaders['test'] = pygt_loader.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    for split, dataloader in dataloaders.items():
        print(f"DataLoader for {split} set:")
        print(f"Number of batches: {len(dataloader)}")
        
    return dataloaders




class TrainerModel(object):
    def __init__(self, model, dataset,device, save_dir, **params):
        dataloader_params =params.get("dataloader_params")
        self.model = model
        self.dataset = dataset 
        self.loader = train_test_val_split(dataset, **dataloader_params)
        self.device = device
        self.optimizer = None
        self.name = model.name
        self.save_dir = save_dir

    def __loss__(self, logits, labels):
        return F.mse_loss(logits, labels)
    

    def train(self,train_params):
        num_epochs = train_params["num_epochs"]
        num_early_stop = train_params["num_early_stop"]
        self.optimizer = Adam(self.model.parameters())
        lr_schedule = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.1, patience=2, min_lr=0.001
        )
        self.model.to(self.device)
        best_eval_r2score = -100.0
        best_eval_loss = 0.0
        early_stop_counter = 0
        for epoch in range(num_epochs):
            is_best = False
            self.model.train()
            losses = []
            for batch in self.loader["train"]:
                batch = batch.to(self.device)
                loss = self._train_batch(batch, batch.y)
                losses.append(loss)
            train_loss = torch.FloatTensor(losses).mean().item()
            with torch.no_grad():
                eval_loss, eval_r2score = self.eval()
            print(
                    f"Epoch:{epoch}, Training_loss:{train_loss:.4f}, Eval_loss:{eval_loss:.4f}, Eval_r2:{eval_r2score:.4f}, lr:{self.optimizer.param_groups[0]['lr']}"
                )
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

            if best_eval_r2score < eval_r2score:
                is_best = True
                best_eval_r2score = eval_r2score
            recording = {"epoch": epoch, "is_best": str(is_best)}
            if self.save:
                self.save_model(is_best, recording=recording)

    def eval(self):
        self.model.to(self.device)
        self.model.eval()
        
        losses, r2scores = [], []
        for batch in self.loader["eval"]:
            batch = batch.to(self.device)
            loss, batch_preds = self._eval_batch(batch, batch.y)
            r2scores.append(r2_score(batch.y.cpu(), batch_preds.cpu()))
            losses.append(loss)
        eval_loss = torch.tensor(losses).mean().item()
        eval_r2score = np.mean(r2scores)
        print(
            f"eval loss: {eval_loss:.6f}, eval r2score {eval_r2score:.6f}"
        )
        return eval_loss, eval_r2score
    


    def test(self):
        state_dict = torch.load(
            os.path.join(self.save_dir, f"{self.name}_latest.pth")
        )["net"]

        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        losses, r2scores, preds = [], [], []
        for batch in self.loader["test"]:
            batch = batch.to(self.device)
            loss, batch_preds = self._eval_batch(batch, batch.y)
            preds.append(batch_preds)
            r2scores.append(r2_score(batch.y.detach().cpu(), batch_preds.detach().cpu()))
            losses.append(loss)
        test_loss = torch.tensor(losses).mean().item()
        test_r2score = np.mean(r2scores)
        preds = torch.cat(preds, dim=-1)
        print(
            f"test loss: {test_loss:.6f}, test r2score {test_r2score:.6f}"
        )
        scores = {
        "test_loss": test_loss,
        "test r2score": test_r2score,
        }
        self.save_scores(scores)
        return test_loss, test_r2score, preds
    
    
    
    def _train_batch(self, data, labels):
        logits = self.model(data.x, data.edge_index, data.batch)
        loss = self.__loss__(logits, labels)
        loss.backward()
        
        self.optimizer.zero_grad()
        self.optimizer.step()
        return loss.item()



    def _eval_batch(self, data, labels):
        self.model.eval()
        logits = self.model(data.x, data.edge_index, data.batch)
        loss = self.__loss__(logits, labels)
        loss = loss.item()
        preds = logits
        return loss, preds
    

    def save_scores(self, scores):
        with open(os.path.join(self.save_dir, f"{self.name}_scores.json"), "w") as f:
            json.dump(scores, f)


    def load_model(self):
        state_dict = torch.load(
            os.path.join(self.save_dir, f"{self.name}_best.pth")
        )["net"]
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def save_model(self, is_best=False, recording=None):
        self.model.to("cpu")
        state = {"net": self.model.state_dict()}
        for key, value in recording.items():
            state[key] = value
        latest_pth_name = f"{self.name}_latest.pth"
        best_pth_name = f"{self.name}_best.pth"
        ckpt_path = os.path.join(self.save_dir, latest_pth_name)
        torch.save(state, ckpt_path)
        if is_best:
            print("saving best...")
            shutil.copy(ckpt_path, os.path.join(self.save_dir, best_pth_name))
        self.model.to(self.device)
       