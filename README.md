# Master Thesis - Forecasting and Classification of Incidents in Electrical Networks Using Temporal Neural Networks on Graphs

## Overview

This repository contains the code and documentation for my Master Thesis project, which focuses on conducting experiments with different temporal GNNs architectures, applied to two different problems: classification and forecasting of incidents in electrical networks. 


The data used in this project is a dataset obtained by simulating different kinds of incidents in a distribution and transmission power grid, originally presented by the paper "A Multi-scale Time-series Dataset with Benchmark for Machine Learning in Decarbonized Energy Grids" (https://arxiv.org/abs/2110.06324). 

## Summary and Background

The significant climate transformations we have observed and anticipated for the 21st century, including global warming, as a consequence of over a century of net greenhouse gas emissions, reflect profound changes that have manifested over the past 65 years.

Energy-related greenhouse gas emissions are the largest contributor to climate warming in recent years. For this reason, the role of advanced technologies becomes crucial in addressing and mitigating incidents in electrical networks, as these networks are utilized in most sectors and have the potential to generate a positive impact on them.

In this study, experiments have been conducted with a power grid transmission system through various network disturbance simulations, specifically focusing on five types of issues: generator trip, node trip, node fault, branch trip, and branch fault. Each simulation has been modeled as a sequence of graphs, and the experiments have been conducted using neural network technologies specifically designed for dynamic graphs.

The objective has been to solve two main problems: first, the identification of incident types using historical network data; second, the development of a predictive system that, for a specific type of incident, predicts the future state of the network, aiming to anticipate conditions of instability in the electrical grid to facilitate effective preventive actions.

The experimentation included graph neural network architectures based on recurrent networks (AGCRN, DyGrEncoder, MPNN-LSTM, EvolveGCN, and DCRNN), and architectures based on attention mechanisms (ASTGCN, MSTGCN, MTGNN, and STConv). For each of these, an adaptation process was necessary to incorporate the structural information of the graph over time. Additionally, in both the classification and regression problems, the results obtained with a simple LSTM architecture were included as a reference, and they were compared using appropriate metrics in each case.


## Project Description

This project is composed by the following stages:

- Data Preparation: software development for data preparation, in order to generate the powergrid dataset for the experiments, from the files obtained in [the dataset paper](https://arxiv.org/abs/2110.06324).
- Model Development: development of different models based on GNNs, based on the implementation of [Pytorch Geometric Temporal](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html), with a focus on the adaptation of each model to the modular software developed for the project and the specifics of the dataset, such as the inclusion of grid interactions.
- Generalized Training Class: we have developed a modular software that allows us to train the models implemented. The heart of the system is a training class which is responsible for managing the entire lifecycle of the models, from data loading, training, and validation to evaluation and result export. This class has been designed to be highly flexible and extensible, allowing new models to be incorporated with minimal modification to the existing code.
- Experiments: We have tested all the models in the dataset, fine-tuned their parameters, and we have compared the results with a simple LSTM model. The results are presented in the study, and they are compared using appropriate metrics in each problem.


## Repository Structure and Description of main classes


### Data Preparation


| File Name | Description | Stage and Type|
| --- | --- | --- |
| [Data Preprocessing and EDA Notebook](https://github.com/maaguado/GNNs_PowerGraph/blob/main/TFM/preprocesamiento/preprocess.ipynb) | Initial experiment with dataloader class and EDA |![NOTEBOOK](https://img.shields.io/badge/Notebook-FAA31B) ![DONE](https://img.shields.io/badge/-DONE-green)|
| [DataLoader Class](https://github.com/maaguado/GNNs_PowerGraph/blob/main/TFM/utils/powergrid.py) | Includes preliminary experiments for different types of time series and missing data distribution | ![CLASS](https://img.shields.io/badge/Class-02AFB8) ![DONE](https://img.shields.io/badge/-DONE-green)|


#### PowerGridDatasetLoader Class

The PowerGridDatasetLoader class is designed to handle the loading and preprocessing of power grid datasets to use in machine learning tasks, specifically regression and classification, ensuring that the data is correctly formatted and preprocessed before it is fed into the model. Below is an outline of its attributes:


- **_natural_folder**: A string representing the path to the directory containing the dataset files.
- **problem**: A string that indicates the type of problem being addressed, either "regression" or "classification".
- **voltages**: A dictionary that stores voltage data for each node in the power grid.
- **buses**: A list of bus numbers representing different nodes in the power grid.
- **types**: A list that holds the type of each incident.
- **edge_attr**: A list of attributes associated with the edges in the graph representation of the power grid.
- **edge_index**: A list of indices that represent the connections (edges) between nodes (buses) in the grid.
- **processed**: A boolean flag indicating whether the data has been processed.
- **transformation_dict**: A dictionary used for transforming node identifiers.


### Model Development


| File Name | Description | Stage and Type|
| --- | --- | --- |
| [Models Implementations](https://github.com/maaguado/GNNs_PowerGraph/blob/main/TFM/utils/models.py) | Includes the implementation in PyTorch for each one of the models included in the study| ![CLASS](https://img.shields.io/badge/Class-02AFB8) ![DONE](https://img.shields.io/badge/-DONE-green)|

In particular, the models included are:

**Recurrent Graph Convolutions**

* **[DCRNN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.dcrnn.DCRNN)** from Li *et al.*: [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://arxiv.org/abs/1707.01926) (ICLR 2018)

* **[DyGrEncoder](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.dygrae.DyGrEncoder)** from Taheri *et al.*: [Learning to Represent the Evolution of Dynamic Graphs with Recurrent Models](https://dl.acm.org/doi/10.1145/3308560.3316581)

* **[EvolveGCNO](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.evolvegcno.EvolveGCNO)** from Pareja *et al.*: [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/abs/1902.10191)

* **[AGCRN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.agcrn.AGCRN)** from Bai *et al.*: [Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting](https://arxiv.org/abs/2007.02842) (NeurIPS 2020)

* **[MPNN LSTM](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.mpnn_lstm.MPNNLSTM)** from Panagopoulos *et al.*: [Transfer Graph Neural Networks for Pandemic Forecasting](https://arxiv.org/abs/2009.08388) (AAAI 2021)
  
**Attention Aggregated Temporal Graph Convolutions**

* **[STGCN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.attention.stgcn.STConv)** from Yu *et al.*: [Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://arxiv.org/abs/1709.04875) (IJCAI 2018)

* **[ASTGCN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.attention.astgcn.ASTGCN)** from Guo *et al.*: [Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/3881) (AAAI 2019)

* **[MSTGCN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.attention.mstgcn.MSTGCN)** from Guo *et al.*: [Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/3881) (AAAI 2019)

* **[MTGNN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.attention.mtgnn.MTGNN)** from Wu *et al.*: [Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks](https://arxiv.org/abs/2005.11650) (KDD 2020)


### Generalized Training Class

| File Name | Description | Stage and Type|
| --- | --- | --- |
| [Trainer Class](https://github.com/maaguado/GNNs_PowerGraph/blob/main/TFM/utils/trainer.py) | includes the implementation of the trainer class, which manages the entire lifecycle of the models, from data loading, training, and validation to evaluation and result export| ![CLASS](https://img.shields.io/badge/Class-02AFB8) ![DONE](https://img.shields.io/badge/-DONE-green)|


The `trainer.py` file serves as the core of a flexible and extensible training framework designed to handle the training, validation, and evaluation processes of various machine learning models. At the heart of this framework is the `TrainerModel` class, a generalized trainer class that provides the foundational structure and methods required for managing the training lifecycle. This includes tasks such as data loading, model initialization, optimization, and performance tracking.

The `TrainerModel` class is engineered to be highly adaptable, allowing it to serve as a base class from which more specialized trainer classes are derived. Each of these derived classes is tailored to accommodate the unique training requirements of different model architectures, such as LSTM networks, Graph Convolutional Networks (GCN), and other advanced models that integrate temporal and graph-based learning. By leveraging the `TrainerModel` as a common framework, the software reduces redundancy and ensures consistency across the training processes of diverse models. This approach not only streamlines the integration of new models but also enhances the maintainability of the training pipeline, allowing each model-specific trainer class to focus on the intricacies of its respective architecture while relying on the generalized `TrainerModel` class for the overall training logic.


### Experiments 

As we mentioned earlier, we have conducted experiments in both regression and classification problems, using the models implemented. For the regression problem, however, we have separated each one of the simulations by incident type (gen trip, bus trip, bus fault, branch trip and branch fault), since the variability of the data is very high and the nature of the incidents is very different.

Also, due to computational limitations, for some of the models, we have only been able to tune the hyperparameters for a subset the parameter space. This information is included in the table below.

#### Regression

| File Name | Type of Parameter Tuning | Type | 
| --- | --- | --- |
| [DCRNN Notebook](https://github.com/maaguado/GNNs_PowerGraph/blob/main/TFM/regresion/dcrnn.ipynb) | ![Complete](https://img.shields.io/badge/-Complete-green)|![NOTEBOOK](https://img.shields.io/badge/Notebook-FAA31B) |
| [A3TGCN Notebook](https://github.com/maaguado/GNNs_PowerGraph/blob/main/TFM/regresion/a3tgcn.ipynb) | ![Complete](https://img.shields.io/badge/-Complete-green)|![NOTEBOOK](https://img.shields.io/badge/Notebook-FAA31B) |
| [AGCRN Notebook](https://github.com/maaguado/GNNs_PowerGraph/blob/main/TFM/regresion/agcrn.ipynb) | ![Complete](https://img.shields.io/badge/-Complete-green)|![NOTEBOOK](https://img.shields.io/badge/Notebook-FAA31B) |
| [MSTGCN Notebook](https://github.com/maaguado/GNNs_PowerGraph/blob/main/TFM/regresion/mstgcn.ipynb) | ![Complete](https://img.shields.io/badge/-Complete-green)|![NOTEBOOK](https://img.shields.io/badge/Notebook-FAA31B) |
| [ASTGCN Notebook](https://github.com/maaguado/GNNs_PowerGraph/blob/main/TFM/regresion/astgcn.ipynb) | ![Subset](https://img.shields.io/badge/-Subset-red)|![NOTEBOOK](https://img.shields.io/badge/Notebook-FAA31B) |
| [DyGrEncoder Notebook](https://github.com/maaguado/GNNs_PowerGraph/blob/main/TFM/regresion/DryGrEncoder.ipynb) | ![Complete](https://img.shields.io/badge/-Complete-green)|![NOTEBOOK](https://img.shields.io/badge/Notebook-FAA31B) |
| [EvolveGCN Notebook](https://github.com/maaguado/GNNs_PowerGraph/blob/main/TFM/regresion/EvolveGCN.ipynb) | ![Complete](https://img.shields.io/badge/-Complete-green)|![NOTEBOOK](https://img.shields.io/badge/Notebook-FAA31B) |
| [MPNN LSTM Notebook](https://github.com/maaguado/GNNs_PowerGraph/blob/main/TFM/regresion/mpnnlstm.ipynb) | ![Subset](https://img.shields.io/badge/-Subset-red)|![NOTEBOOK](https://img.shields.io/badge/Notebook-FAA31B) |
| [STConv Notebook](https://github.com/maaguado/GNNs_PowerGraph/blob/main/TFM/regresion/STConv.ipynb) | ![Subset](https://img.shields.io/badge/-Subset-red)|![NOTEBOOK](https://img.shields.io/badge/Notebook-FAA31B) |
