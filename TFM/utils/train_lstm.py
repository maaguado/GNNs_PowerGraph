import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
import torch.nn.functional as F
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, name, node_features, node_count, n_target, hidden_size =50, num_layers=1):
        self.name  =name
        self.n_nodes = node_count
        self.n_target = n_target
        self.n_features = node_features
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(self.n_features, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.n_target)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out


def train_lstm_model(dataset_gt, params_model, dataloader, num_epochs, lr, problem, save_model = False, path_save = "./results" ):

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
        losses_batch = []
        for batch in dataloader['train']:
            x = batch.x.view(len(batch), model.n_nodes, model.n_features).to(device)
            y = batch.y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat.view(-1, model.n_target), y)

            loss.backward()
            optimizer.step()
            
            losses_batch.append(loss.item())
        train_loss = torch.FloatTensor(losses_batch).mean().item()
        losses.append(train_loss)

        # Validación del modelo
        model.eval()

        with torch.no_grad():
            e_loss_batch, r2_loss_batch = [], []
            for batch in dataloader['val']:
                batch = batch.to(device)
                x = batch.x.view(len(batch), model.n_nodes, model.n_features)
                loss = F.mse_loss(y_hat.view(-1, model.n_target), batch.y).item()
                r2_loss_batch.append(r2_score(batch.y.cpu(), y_hat.view(-1, model.n_target).cpu()))
                e_loss_batch.append(loss)
                
        eval_loss = torch.FloatTensor(e_loss_batch).mean().item()
        eval_r2score = torch.FloatTensor(r2_loss_batch).mean().item()
        eval_losses.append(eval_loss)
        r2scores.append(eval_r2score)

        print(f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Eval Loss: {eval_loss:.4f} | "
            f"Eval R2: {eval_r2score:.4f} | ")
        

    if save_model:
        path_save_model = path_save+f"/{problem}/"+ "lstm_100_20.pt"
        torch.save(model.state_dict(), path_save_model)


        print("Entrenamiento completado y modelo guardado.")
    else:
        print("Entrenamiento completado.")
    return model,losses, eval_losses, r2scores




def test(model, dataloader):
    
    print("\n==================== TEST INFO ===================\n")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.eval()
    predictions = []
    real =[]
    losses_tst, r2score_tst, loss_nodes = [], [], []

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
    test_loss = torch.tensor(losses_tst).mean().item()
    test_r2score = np.mean(r2score_tst)

    print(f"Test_loss:{test_loss:.4f}, Test_r2:{test_r2score:.4f}")
    return r2score_tst, losses_tst, loss_nodes, predictions, real