import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt

etfs = ['USMV', 'SPMV', 'EFAV', 'SPLV', 'XMLV', 'PXMV', 'MTUM', 'VFMO', 
        'PDP', 'VFMF', 'VTV', 'SCHV', 'RPV', 'IUSV', 'IVOV', 'QUAL', 
        'SPHQ', 'JHML', 'DGRW', 'QUS', 'SMLF', 'VIOO', 'IJS', 'IJR', 'VB']

# We will use past 6 months data
start_date = pd.Timestamp.now() - pd.DateOffset(days=7)
end_date = pd.Timestamp.now()

# Initialize a DataFrame to hold all the data
data_df = pd.DataFrame()

for etf in etfs:
    # Get the historical data
    data = yf.download(etf, start=start_date, end=end_date, interval='1m')

    # We'll use Close price for this example, add more features if needed
    data = data[['Close']]

    # Rename the columns to avoid conflict while concatenating
    data.columns = [f'{etf}']

    # Concatenate the data to the main DataFrame
    data_df = pd.concat([data_df, data], axis=1)

# Fill the missing values
data_df.fillna(method='ffill', inplace=True)
data_df.dropna(inplace=True)

# Calculate price ratios to use as edge weights
price_ratios = pd.DataFrame(index=data_df.index)
for i in range(len(etfs)):
    for j in range(i + 1, len(etfs)):
        ratio = data_df[etfs[i]] / data_df[etfs[j]]
        price_ratios[f'{etfs[i]}_{etfs[j]}'] = ratio

# Calculate the mean price ratio for each pair to use as the edge weight in the graph
mean_price_ratios = price_ratios.mean()

# Create directed edge index and edge weights
edge_index = []
edge_attr = []
for i, column in enumerate(mean_price_ratios.index):
    source, target = column.split('_')
    edge_index.append((etfs.index(source), etfs.index(target)))
    edge_attr.append(mean_price_ratios[i])
edge_index = torch.tensor(edge_index).t().contiguous()
edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)

# Node features are the scaled Close price for each ETF
scaler = MinMaxScaler()
x = torch.tensor(scaler.fit_transform(data_df.values), dtype=torch.float)

# Create a PyG data object
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Create a NetworkX directed graph for visualization
G = nx.DiGraph()
for i, (source, target) in enumerate(edge_index.t().tolist()):
    G.add_edge(source, target, weight=edge_attr[i].item())

# Visualize the graph
plt.figure(figsize=(12,12))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, connectionstyle='arc3, rad = 0.1')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()
