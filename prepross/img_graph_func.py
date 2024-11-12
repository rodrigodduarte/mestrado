import torch
from torch_geometric.data import Data
import numpy as np
from numba import jit

@jit(nopython=True)
def distancia_euclideana(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

@jit(nopython=True)
def build_graph(image, radius, L):
    height, width, channels = image.shape
    radius2 = radius * radius
    # number_of_pixels = height * width * channels

    valid_nodes = []
    node_mapping = {}
    edge_index = []
    edge_attr = []

    current_node_id = 0

    for y in range(height):
        for x in range(width):
            if np.mean(image[y, x]) == 255.0:
                continue

            for channel_node in range(3):
                node_id = y * width * 3 + x * 3 + channel_node
                valid_nodes.append(image[y, x, channel_node])
                node_mapping[node_id] = current_node_id
                current_node_id += 1

            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    neighbor_x, neighbor_y = x + dx, y + dy
                    if 0 <= neighbor_x < width and 0 <= neighbor_y < height and (dx != 0 or dy != 0):

                        if np.mean(image[neighbor_y, neighbor_x]) == 255.0:
                            continue
                        
                        dist = distancia_euclideana(x, y, neighbor_x, neighbor_y)
                        if (dist - radius + 1) > 0 and dist <= radius:
                            for channel_node in range(3):
                                node_id = y * width * 3 + x * 3 + channel_node
                                for channel_neighbor in range(3):
                                    neighbor_id = neighbor_y * width * 3 + neighbor_x * 3 + channel_neighbor
                                    if node_id in node_mapping and neighbor_id in node_mapping:
                                        diff = abs(image[y, x, channel_node] - image[neighbor_y, neighbor_x, channel_neighbor])
                                        edge_weight = ((diff + 1) * (dist + 1) - 1) / ((L + 1) * radius2 - 1)
                                        if image[y, x, channel_node] < image[neighbor_y, neighbor_x, channel_neighbor]:
                                            edge_index.append([node_mapping[node_id], node_mapping[neighbor_id]])
                                            edge_attr.append(edge_weight)
    
    return np.array(valid_nodes), np.array(edge_index).T, np.array(edge_attr)

def SSN_to_graph(image, radius, L=255):
    valid_nodes, edge_index, edge_attr = build_graph(image, radius, L)

    # Converte para tensores do PyTorch
    x = torch.tensor(valid_nodes, dtype=torch.float).view(-1, 1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Cria a estrutura de dados
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data
