
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.cm as cm


def format_plot(ax):
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')

    ax.xaxis.label.set_color('dimgrey')
    ax.tick_params(axis='both', colors='dimgrey', size=20, pad=1)



def show_graph(data, loader, titulo=None):
    G = nx.Graph()
    G.add_edges_from(data.edge_index[0].tolist())  

    for i, x in enumerate(data.x[:,0].tolist()):
        G.nodes[i].update({'Voltage': x})

    for i, (u, v) in enumerate(data.edge_index[0].tolist()):
        G[u][v].update({'Active power': data.edge_attr[0,i].tolist()[0], 
                        'Reactive power': data.edge_attr[0,i].tolist()[1]})

    # Ajuste de posiciones
    pos_temp = {
        3011: (0, 8), 3001: (0, 7), 3002: (1, 8), 3003: (0, 6), 3004: (1, 6),
        3005: (0, 5), 3007: (1, 4), 3006: (1, 5), 3008: (0, 3), 3018: (0, 2), 
        101: (2, 8), 102: (3, 8), 151: (2.5, 7), 152: (2.5, 6), 153: (2.5, 5), 
        154: (2.5, 4), 201: (4, 7), 202: (4, 6), 203: (4, 5), 204: (4, 4), 
        205: (4, 3), 206: (4, 2.5), 211: (5, 7)
    }
    pos = {}
    for x in pos_temp.keys():
        pos[loader.transformation_dict[x]] = pos_temp[x]

    inverse_transformation_dict = {v: k for k, v in loader.transformation_dict.items()}
    
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_labels = [str(inverse_transformation_dict[node]) for node in G.nodes()]  # Etiquetas de los nodos
    text_colors = ['white' if  G.nodes()[node]['Voltage'] <= 1 or G.nodes()[node]['Voltage'] >= 1.05 else 'black' for node in G.nodes()]


    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_labels,  # Añadir etiquetas de nodos
        textposition='middle center',  # Posición de las etiquetas
        hoverinfo='text+name',
        marker=dict(
            symbol='line-ew-open',  # Definir el símbolo del nodo como una raya horizontal
            color=[G.nodes()[node]["Voltage"] for node in G.nodes()],
            showscale=True,
            colorscale='RdBu_r',
            colorbar={"title": "Voltaje"},
            size=30,  # Tamaño de los nodos
            line=dict(color='black', width=17)  # Borde negro para los nodos y ajuste del ancho de las rayas
        ),
        textfont = dict(color=text_colors)
    )

    edge_traces = []  # Lista para almacenar los trazos de los bordes
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        attr = G[edge[0]][edge[1]]["Active power"]
        # Convert the RGBA color to hexadecimal
        hex_color ="#ff0000" if  abs(attr) < 0.001 else "#ffffd9" if abs(attr) < 0.1 else "#96a3a3" 
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=1, color=hex_color),
            hoverinfo='text',
            mode='lines'
        )
        edge_traces.append(edge_trace)

    # Ajustar la posición y el tamaño del eje de la leyenda
    layout = go.Layout(
    title=titulo if titulo is not None else "Sistema de transmisión",
    titlefont_size=20,
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=50),
    plot_bgcolor='white',  # Fondo blanco
    paper_bgcolor='white',  # Fondo blanco
    font=dict(color='black'),  # Letras negras
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis2=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, 1],  # Ajustar la altura del eje de la leyenda
            visible=True,
            overlaying='y',  # Superponer el eje de la leyenda sobre el eje principal
            side='right',  # Posicionar el eje de la leyenda a la derecha
            ticks='',  # Ocultar las marcas del eje de la leyenda
            title='Voltaje',  # Título del eje de la leyenda
            titlefont=dict(
                size=14,  # Tamaño del título del eje de la leyenda
            ),
    )
)
    config = {
  'toImageButtonOptions': {
    'format': 'png', # one of png, svg, jpeg, webp
    'filename': 'plot',
    'height': 500,
    'width': 700,
    'scale':6 # Multiply title/legend/axis/canvas sizes by this factor
  }
}


    fig = go.Figure(data=[*edge_traces, node_trace], layout=layout)
    fig.show(config=config)



def plot_caso_concreto(i, loader):
    info_nodos = loader.reconstruir_voltages(i)


    n_plots = 23
    n_cols = 4
    n_rows = (n_plots + 1) // n_cols  # División redondeada hacia arriba

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 10), dpi=200)
    handles = []
    labels = []
    # Trazar los gráficos
    for i in range(n_plots):
        row = i % n_rows  # Calculamos el índice de la fila
        col = i // n_rows  # Calculamos el índice de la columna
        ax = axs[row, col] if n_plots > 1 else axs
        
        ax = axs[row, col] if n_plots > 1 else axs
        
        sns.lineplot(y=info_nodos[i],x=range(len(info_nodos[i])), ax=ax, legend=False)
            
        if not handles:
                handles, labels = ax.get_legend_handles_labels()
        ax.set_title(f'Nodo {i+1}')
        format_plot(ax)

    # Add legend to the last plot
    #axs[n_rows - 1, n_cols - 2].legend(loc='upper right', bbox_to_anchor=(1.5, 0.95), frameon=True)
    #fig.legend(handles, ['Real', 'Predicciones'], bbox_to_anchor=(0.95, 0.08),loc = 'lower right', fontsize=15)

    # Remove any unused subplots
    if n_plots < (n_rows * n_cols):
        for i in range(n_plots, (n_rows * n_cols)):
            fig.delaxes(axs.flatten()[i])
    plt.tight_layout()
    plt.show()



def plot_specific_problem(problem, loader):
    indices = np.where(np.array(loader.types)==problem)[0]
    info_nodos_full = [loader.reconstruir_voltages(i) for i in indices]

    n_plots = 4
    n_cols = 2
    n_rows = (n_plots + 1) // 2  # División redondeada hacia arriba
    n_check = min(50, len(info_nodos_full))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(30, 10),dpi=200)
    # Trazar los gráficos
    for i in range(n_plots):
        row = i % n_rows  # Calculamos el índice de la fila
        col = i // n_rows  # Calculamos el índice de la columna
        ax = axs[row, col] if n_plots > 1 else axs
        
        ax = axs[row, col] if n_plots > 1 else axs
        
        for j in range(n_check):
            sns.lineplot(y=info_nodos_full[j][i],x=range(len(info_nodos_full[j][i])), ax=ax, label='Real', legend=False)
            
        ax.set_title(f'Nodo {i+1}')
        format_plot(ax)


    if n_plots % n_cols != 0:
        fig.delaxes(axs.flatten()[-1])
    # Ajustes finales
    plt.suptitle('Voltajes de nodos en {} (max {} simulaciones)'.format(problem, n_check), fontsize=20)
    plt.tight_layout()
    plt.show()




def reconstruir_predictions(predictions,real, n_target, situacion, n_div, n_nodes=23):
    
    temp = np.array(predictions).reshape(-1, n_nodes, n_target)
    temp2 =np.array(real).reshape(-1, n_nodes, n_target)
    if n_div != None:
        id_situacion = situacion*n_div 
        return n_div, np.concatenate([np.array(temp[id_situacion+i]) for i in range(n_div)], axis=1), np.concatenate([np.array(temp2[id_situacion+i]) for i in range(n_div)], axis=1)
    m = temp.shape[0]
    return m, np.concatenate([np.array(temp[situacion+i]) for i in range(m)], axis=1), np.concatenate([np.array(temp2[situacion+i]) for i in range(m)], axis=1)



def plot_training_and_eval_losses(train_losses, eval_losses, num_epochs, format_plot):
    epochs = range(1, num_epochs + 1)


    plt.figure(figsize=(12, 5), dpi=200)
    ax = plt.gca()

    sns.lineplot(x=epochs, y=train_losses, label='Training Loss')
    sns.lineplot(x=epochs, y=eval_losses, label='Evaluation Loss', color="royalblue")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss by Epoch')
    ax.legend()
    format_plot(ax) 
    plt.tight_layout()
    plt.show()


    
def plot_predictions(predictions, real, n_target, n_situation, n_div, problem):
    # Reconstruct predictions and true values
    m, preds, y_true = reconstruir_predictions(predictions, real, n_target, n_situation, n_div=n_div)
    
    n_plots = 23
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols  # Calculating number of rows

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 10), dpi=200)
    handles = []
    labels = []
    for i in range(n_plots):
        row = i // n_cols  # Calculate the row index
        col = i % n_cols   # Calculate the column index
        ax = axs[row, col]

        sns.lineplot(y=y_true[i], x=range(n_target * m), ax=ax, label='Real', legend=False, color="royalblue")
        sns.lineplot(y=preds[i], x=range(n_target * m), ax=ax, label='Predicciones', legend=False)
        if not handles:
            handles, labels = ax.get_legend_handles_labels()
        ax.set_title(f'Nodo {i+1}')
        format_plot(ax)
    
    # Add legend to the last plot
    #axs[n_rows - 1, n_cols - 2].legend(loc='upper right', bbox_to_anchor=(1.5, 0.95), frameon=True)
    fig.legend(handles, ['Real', 'Predicciones'], bbox_to_anchor=(0.95, 0.08),loc = 'lower right', fontsize=15)

    # Remove any unused subplots
    if n_plots < (n_rows * n_cols):
        for i in range(n_plots, (n_rows * n_cols)):
            fig.delaxes(axs.flatten()[i])

    # Adjust layout and add super title
    plt.suptitle(f'Predicciones y valores reales en {problem}, caso {n_situation}', fontsize=20)
    plt.tight_layout(pad=2)
    plt.show()