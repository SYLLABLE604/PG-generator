#coding:utf-8

import numpy as np
from math import *
import networkx as nx
import pandas as pd
from grakel.kernels import WeisfeilerLehman, VertexHistogram, ShortestPath, RandomWalk
from grakel.utils import graph_from_networkx
from scipy.spatial.distance import cosine
from pathlib import Path

def save_to_file(file_name, contents):
  # fh = open(file_name, 'w')
  # fh.write(contents)
  # fh.close()
  file_path = Path(__file__).parent / file_name
  with file_path.open('w') as fh:
      fh.write(contents)

def read_file(file_name):
  # fh = open(file_name, 'r')
  # content = fh.read()
  # return content
  file_path = Path(__file__).parent / file_name
  with file_path.open('r') as fh:
      content = fh.read()
  return content

def random_generator(start, end):
  
    if start == end:
        result = start
    else:
        result = np.random.randint(start, end)
    return result

def output_check(input_code):
  code = ''
  index = input_code.find('python')
  if index == -1:
      code = input_code
  else:
      code = input_code[10:-3]
  return code

def sub_similarity_check(edge_index,node_list):  # 结构图和真实图相似度检查
    # if len(node_list) < 10:
    #     edge_data = pd.DataFrame(pd.read_excel(
    #         'reference grid/9 bus edge.xlsx')).values.tolist()
    #     node_data = pd.DataFrame(pd.read_excel(
    #         'reference grid/9 bus node.xlsx')).values.tolist()
    # elif len(node_list) < 20:
    #     edge_data = pd.DataFrame(pd.read_excel(
    #         'reference grid/18 bus edge.xlsx')).values.tolist()
    #     node_data = pd.DataFrame(pd.read_excel(
    #         'reference grid/18 bus node.xlsx')).values.tolist()
    # elif len(node_list) < 30:
    #     edge_data = pd.DataFrame(pd.read_excel(
    #         'reference grid/30 bus edge.xlsx')).values.tolist()
    #     node_data = pd.DataFrame(pd.read_excel(
    #         'reference grid/30 bus node.xlsx')).values.tolist()
    # else:
    #     edge_data = pd.DataFrame(pd.read_excel(
    #         'reference grid/39 bus edge.xlsx')).values.tolist()
    #     node_data = pd.DataFrame(pd.read_excel(
    #         'reference grid/39 bus node.xlsx')).values.tolist()

    if len(node_list) < 10:
        edge_path = Path(__file__).parent / 'reference grid/9 bus edge.xlsx'
        node_path = Path(__file__).parent / 'reference grid/9 bus node.xlsx'
    elif len(node_list) < 20:
        edge_path = Path(__file__).parent / 'reference grid/18 bus edge.xlsx'
        node_path = Path(__file__).parent / 'reference grid/18 bus node.xlsx'
    elif len(node_list) < 30:
        edge_path = Path(__file__).parent / 'reference grid/30 bus edge.xlsx'
        node_path = Path(__file__).parent / 'reference grid/30 bus node.xlsx'
    else:
        edge_path = Path(__file__).parent / 'reference grid/39 bus edge.xlsx'
        node_path = Path(__file__).parent / 'reference grid/39 bus node.xlsx'

        # Read the reference data using the paths
    edge_data = pd.read_excel(edge_path).values.tolist()
    node_data = pd.read_excel(node_path).values.tolist()

    G_input = nx.from_edgelist(edge_index)
    G_example = nx.from_edgelist(edge_data)

    # G_ShortestPath_similarity = 1 - (abs((nx.average_shortest_path_length(G_input) - nx.average_shortest_path_length(G_example)) / nx.average_shortest_path_length(G_example)))
    # similarity_clustering = 1 - (abs(nx.average_clustering(G_input) - nx.average_clustering(G_example)) / nx.average_clustering(G_example) if (nx.average_clustering(G_example)!=0 and nx.average_clustering(G_input)!=0) else 0)
    # similarity_degree =  1 - (abs(((sum(dict(nx.degree(G_input)).values()) / len(G_input.nodes)) - sum(dict(nx.degree(G_example)).values()) / len(G_example.nodes)) / (sum(dict(nx.degree(G_example)).values()) / len(G_example.nodes))))
    # similarity_cloness = 1 - (abs(((sum(dict(nx.closeness_centrality(G_input)).values())/ len(G_input.nodes)) - sum(dict(nx.closeness_centrality(G_example)).values()) / len(G_example.nodes)) / (sum(dict(nx.closeness_centrality(G_example)).values()) / len(G_example.nodes))))
    # similarity_betweenness = (nx.betweenness_centrality(G_input) - nx.betweenness_centrality(G_example))/nx.betweenness_centrality(G_example)

    A = np.array([[nx.average_shortest_path_length(G_input), nx.average_clustering(G_input), sum(dict(nx.degree(G_input)).values()) / len(G_input.nodes), sum(dict(nx.closeness_centrality(G_input)).values())/ len(G_input.nodes)]])
    B = np.array([[nx.average_shortest_path_length(G_example), nx.average_clustering(G_example), sum(dict(nx.degree(G_example)).values()) / len(G_example.nodes), sum(dict(nx.closeness_centrality(G_example)).values()) / len(G_example.nodes)]])
    cosine_similarity = 1 - cosine(A.flatten(), B.flatten())
    # print(G_ShortestPath_similarity,similarity_degree,similarity_cloness,similarity_clustering)
    # similarity = np.mean([G_ShortestPath_similarity, similarity_degree , similarity_cloness , similarity_clustering])
    return cosine_similarity

def final_similarity_check(edge_index,node_list):
    input_edge = np.array(edge_index)[:,[0,1]]
    if(len(node_list[0]) == 3):
        node_list_data = pd.DataFrame(node_list,columns=['ID','Type','Subgrid'])
        input_edge_data = pd.DataFrame(edge_index,columns=['Start','End','Edge type','Subgrid'])
    else:
        node_list_data = pd.DataFrame(node_list,columns=['ID','Type'])
        input_edge_data = pd.DataFrame(edge_index,columns=['Start','End','Edge type'])
    # edge_data = pd.DataFrame(pd.read_excel(
    #         'reference grid/real grid edge.xlsx'))
    # node_data = pd.DataFrame(pd.read_excel(
    #     'reference grid/real grid node.xlsx'))

    edge_data_path = Path(__file__).parent / 'reference grid' / 'real grid edge.xlsx'
    node_data_path = Path(__file__).parent / 'reference grid' / 'real grid node.xlsx'
    edge_data = pd.read_excel(edge_data_path)
    node_data = pd.read_excel(node_data_path)

    G_input = nx.from_edgelist(input_edge)
    G_example = nx.from_edgelist(edge_data.values.tolist())

    #节点占比
    PV_percentage = len(node_data[node_data['Type'] != 0])/len(node_data)
    PQ_percentage = 1-PV_percentage
    input_PV_percentage = len(node_list_data[node_list_data['Type'] != 0])/len(node_list)
    input_PQ_percentage = 1-input_PV_percentage

    #平均节点度
    degree = pd.DataFrame(list(nx.degree(G_example)),columns=['ID','degree'])
    input_degree = pd.DataFrame(list(nx.degree(G_input)),columns=['ID','degree'])
    average_degree = degree['degree'].sum() / len(node_data)
    input_average_degree = input_degree['degree'].sum() / len(node_list)
    PV_degree = 0
    PV_num = 0
    PQ_degree = 0
    PQ_num = 0
    for i,row in node_data.iterrows():
        if(row['Type'] == 0):
            PV_degree += degree[degree['ID'] == row['Node']]['degree'].values.tolist()[0]
            PV_num += 1
        else:
            PQ_degree += degree[degree['ID'] == row['Node']]['degree'].values.tolist()[0]
            PQ_num += 1
    referencr_PV_degree = PV_degree/PV_num
    referencr_PQ_degree = PQ_degree/PQ_num

    PV_degree = 0
    PV_num = 0
    PQ_degree = 0
    PQ_num = 0
    
    for i,row in node_list_data.iterrows():
        if(row['Type'] == 0):
            PV_degree += input_degree[input_degree['ID'] == row['ID']]['degree'].values.tolist()[0]
            PV_num += 1
        else:
            PQ_degree += input_degree[input_degree['ID'] == row['ID']]['degree'].values.tolist()[0]
            PQ_num += 1
    input_PV_degree = PV_degree/PV_num
    input_PQ_degree = PQ_degree/PQ_num

    #边类型占比
    edge_type = []
    for i,row in edge_data.iterrows():
        start_node = node_data[node_data['Node'] == row['Start node']].iloc[0]
        end_node = node_data[node_data['Node'] == row['End node']].iloc[0]
        if(start_node['Type'] == 0 and end_node['Type'] == 0 ):
            edge_type.append(3)
        elif(start_node['Type'] in [1,2] and end_node['Type'] in [1,2]):
            edge_type.append(1)
        else:
            edge_type.append(2)
    edge_data.loc[:,'Edge type'] = edge_type
    PVPV_percentage = len(edge_data[edge_data['Edge type'] == 1])/len(edge_data)
    PVPQ_percentage = len(edge_data[edge_data['Edge type'] == 2])/len(edge_data)
    PQPQ_percentage = len(edge_data[edge_data['Edge type'] == 3])/len(edge_data)

    
    input_PVPV_percentage = len(input_edge_data[input_edge_data['Edge type'] == 1])/len(input_edge)
    input_PVPQ_percentage = len(input_edge_data[input_edge_data['Edge type'] == 2])/len(input_edge)
    input_PQPQ_percentage = len(input_edge_data[input_edge_data['Edge type'] == 3])/len(input_edge)

    print(PV_percentage,input_PV_percentage)
    print(PQ_percentage,input_PQ_percentage)
    print(average_degree,input_average_degree)
    print(referencr_PV_degree,input_PV_degree)
    print(referencr_PQ_degree,input_PQ_degree)
    print(PVPV_percentage,input_PVPV_percentage)
    print(PVPQ_percentage,input_PVPQ_percentage)
    print(PQPQ_percentage,input_PQPQ_percentage)

    return 


def connection_type_check(start_node, end_node, node_sum):
    # PV--PV:1, PV--PQ:2, PQ--PQ:3
    start_type = node_sum[start_node - 1][1]
    end_type = node_sum[end_node - 1][1]
    if (start_type in [1,2] and end_type in [1,2]):
        return 1
    elif ((start_type in [1,2] and end_type == 0) or (start_type == 0 and end_type in [1,2])):
        return 2
    else:
        return 3
    


    