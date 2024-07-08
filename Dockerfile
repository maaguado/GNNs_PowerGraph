FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN apt-get update -y && \
    apt-get install -y libhdf5-dev git

RUN git clone https://github.com/maaguado/GNNs_PowerGraph.git

RUN pip install --no-cache-dir torch==2.2.2

RUN pip install --no-cache-dir torch_cluster==1.6.3 torch_scatter==2.1.2 torch_sparse==0.6.18 torch_geometric==2.3.0 torch_geometric_temporal==0.54.0 torchaudio==2.3.1 torchdata==0.7.1 torchvision==0.18.1

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir jupyter

EXPOSE 9000
