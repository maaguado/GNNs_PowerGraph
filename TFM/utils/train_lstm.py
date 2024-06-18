import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
import torch.nn.functional as F
import numpy as np
from utils.models import LSTMModel


def train_lstm_model(dataset_gt, params_model, dataloader, num_epochs, lr, problem, save_model = False, path_save = "./results", batch=True, steps = 30):

    n_nodes =dataset_gt.features[0].shape[0]
    n_target = dataset_gt.targets[0].shape[1]
    n_features = dataset_gt[0].x.shape[1]

    n_layers = params_model['n_layers']
    hidden_size = params_model['hidden_size']

    model = LSTMModel(name="LSTM", node_features=n_features, node_count=n_nodes, n_target=n_target, hidden_size=hidden_size, num_layers=n_layers)

    device = device =torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model.to(device)
    # Función de pérdida y optimizador
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    eval_losses, r2scores = [], []

    print("\n==================== TRAIN ===================\n")
    for epoch in range(num_epochs):
        
        model.train()
        losses_epoch = []
        if batch:

            for batch in dataloader['train']:
                x = batch.x.view(len(batch), model.n_nodes, model.n_features).to(device)
                y = batch.y.to(device)
                
                y_hat = model(x)
                loss = criterion(y_hat.view(-1, model.n_target), y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses_epoch.append(loss.item())
        else:
            counter = 0
            accumulated_loss = 0.0
            for time, snapshot in enumerate(dataloader['train']):
                x = snapshot.x.view( model.n_nodes, model.n_features)[None,:,:].to(device)
                y = snapshot.y[None,:,:].to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)


                loss.backward()
                accumulated_loss += loss.item()
                counter += 1

                if counter % steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    losses_epoch.append(accumulated_loss / steps)  
                    accumulated_loss = 0.0  
            if counter % steps != 0:
                optimizer.step()
                optimizer.zero_grad()
                losses_epoch.append(accumulated_loss / (counter % steps))
                accumulated_loss = 0.0

        train_loss = torch.FloatTensor(losses_epoch).mean().item()
        losses.append(train_loss)

        # Validación del modelo
        model.eval()

        with torch.no_grad():
            e_loss_epoch, r2_loss_epoch = [], []
            if batch:
                for batch in dataloader['val']:
                    batch = batch.to(device)
                    x = batch.x.view(len(batch), model.n_nodes, model.n_features)
                    y_hat = model(x)
                    
                    loss = F.mse_loss(y_hat.view(-1, model.n_target), batch.y).item()
                    r2_loss_epoch.append(r2_score(batch.y.cpu(), y_hat.view(-1, model.n_target).cpu()))
                    e_loss_epoch.append(loss)
            else:
                for snapshot in dataloader['val']:
                    x = snapshot.x.view(model.n_nodes, model.n_features)[None,:,:].to(device)
                    y = snapshot.y[None,:,:].to(device)
                    y_hat = model(x)
                    loss = F.mse_loss(y_hat, y).item()
                    r2_loss_epoch.append(r2_score(snapshot.y.cpu(), y_hat.view(-1, model.n_target).cpu()))
                    e_loss_epoch.append(loss)

                
        eval_loss = torch.FloatTensor(e_loss_epoch).mean().item()
        eval_r2score = torch.FloatTensor(r2_loss_epoch).mean().item()
        eval_losses.append(eval_loss)
        r2scores.append(eval_r2score)

        print(f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Eval Loss: {eval_loss:.4f} | "
            f"Eval R2: {eval_r2score:.4f} | ")
        
    if save_model:
    
        path_save_model = path_save+f"/{problem}/"+ "lstm_100_20_batch.pt" if batch else path_save+f"/{problem}/"+ "lstm_100_20_nobatch.pt"
        torch.save(model.state_dict(), path_save_model)
        print("Entrenamiento completado y modelo guardado.")
    else:
        print("Entrenamiento completado.")
    return model,losses, eval_losses, r2scores




def test_lstm(model, dataloader, batch = True):
    
    print("\n==================== TEST INFO ===================\n")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.eval()
    predictions = []
    real =[]
    losses_tst, r2score_tst, loss_nodes = [], [], []
    if batch:
        for batch in dataloader['test']:
            batch = batch.to(device)
            x = batch.x.view(len(batch), model.n_nodes, model.n_features)
            y_hat = model(x)
            loss = F.mse_loss(y_hat.view(-1, model.n_target), batch.y).item()

            loss_per_node = F.mse_loss(y_hat.view(-1, model.n_target), batch.y, reduction='none')
            loss_per_node = loss_per_node.view(len(batch), model.n_nodes, model.n_target).mean(dim=0).mean(dim=1).cpu().detach().numpy()
            loss_nodes.append(loss_per_node)


            preds = y_hat.view(len(batch), model.n_nodes, model.n_target)
            real.append(batch.y.view(len(batch), model.n_nodes, model.n_target).cpu().detach().numpy())
            predictions.append(preds.cpu().detach().numpy())
            r2score_tst.append(r2_score(batch.y.detach().cpu(), y_hat.view(-1, model.n_target).detach().cpu()))
            losses_tst.append(loss)
    else:
        for snapshot in dataloader['test']:
            x = snapshot.x.view(model.n_nodes, model.n_features)[None,:,:].to(device)
            y = snapshot.y[None,:,:].to(device)
            y_hat = model(x)
            loss = F.mse_loss(y_hat, y).item()

            loss_per_node = F.mse_loss(y_hat, y, reduction='none')
            loss_per_node = loss_per_node.view(1, model.n_nodes, model.n_target).mean(dim=0).mean(dim=1).cpu().detach().numpy()
            loss_nodes.append(loss_per_node)


            preds = y_hat.view(1, model.n_nodes, model.n_target)
            real.append(y.view(1, model.n_nodes, model.n_target).cpu().detach().numpy())
            predictions.append(preds.cpu().detach().numpy())
            r2_score_prelim = r2_score(y.squeeze(0).detach().cpu(), y_hat.squeeze(0).detach().cpu())
            r2score_tst.append(r2_score_prelim)
            losses_tst.append(loss)

    test_loss = torch.tensor(losses_tst).mean().item()
    test_r2score = np.mean(r2score_tst)

    print(f"Test_loss:{test_loss:.4f}, Test_r2:{test_r2score:.4f}")
    return r2score_tst, losses_tst, loss_nodes, predictions, real