import math
import torch
import numpy as np
from math import *
import networkx as nx
import pandas as pd
from datetime import datetime
from Tool import save_to_file, read_file, random_generator, output_check, sub_similarity_check, final_similarity_check, \
    connection_type_check
from Image_gen import distribution_code_generation
import os
import matplotlib.pyplot as plt


def main(scale_type, graph_type):
    index = 1
    node_sum = []
    edge_sum = []
    sub_similarity = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 子网描述生成
    start = index
    net_feature = []  # subnet_bus_number, generator_num, start_num, end_num

    if scale_type == 1:
        net_feature.append(random_generator(10, 25))
        net_feature.append(random_generator(1, ceil(net_feature[0] / 6)))
    elif scale_type == 2:
        net_feature.append(random_generator(26, 50))
        net_feature.append(random_generator(1, ceil(net_feature[0] / 6)))
    elif scale_type == 3:
        net_feature.append(random_generator(51, 75))
        net_feature.append(random_generator(1, ceil(net_feature[0] / 6)))
    elif scale_type == 4:
        net_feature.append(random_generator(76, 100))
        net_feature.append(random_generator(1, ceil(net_feature[0] / 6)))

    bus_number = net_feature[0]
    net_feature.append(start)
    end = start + net_feature[0] - 1
    net_feature.append(end)
    start = end + 1

    print(net_feature)

    # 子网节点生成
    for k in range(net_feature[0]):
        node_feature = []
        node_feature.append(k + 1)
        if (net_feature[1] > 0) and (k == 0):
            node_feature.append(2)
        elif (k <= net_feature[1] - 1):
            node_feature.append(1)
        elif (net_feature[1] <= k < net_feature[0]):
            node_feature.append(0)
        node_sum.append(node_feature)

    print(node_sum)
    # 子网边生成
    start = 0
    end = 0

    # print(subnet_node)

    # 随机连边
    z = 0
    while z < len(node_sum):
        edge_pair = []
        start_node = node_sum[z][0]
        rest_graph = list(node_sum)
        rest_graph.remove(rest_graph[z])

        a = random_generator(0, len(rest_graph))
        end_node = rest_graph[a][0]

        exist_edge = []
        for i in range(len(edge_sum)):
            group = []
            group.append(edge_sum[i][0])
            group.append(edge_sum[i][1])
            exist_edge.append(group)

        iterr = 0
        state = True
        while (([start_node, end_node] in exist_edge) or ([end_node, start_node] in exist_edge)):
            # print(1)
            end_node = rest_graph[random_generator(0, len(rest_graph))][0]
            if iterr > 10:
                state = False
                break
        if state == True:
            edge_pair.append(start_node)
            edge_pair.append(end_node)
            edge_pair.append(connection_type_check(start_node, end_node, node_sum))
            edge_sum.append(edge_pair)
            z += 1

    # 检查连通性
    G = nx.from_edgelist(np.array(edge_sum)[:, [0, 1]])
    if nx.is_connected(G) == False:
        largest_net = max(nx.connected_components(G), key=len)
        for b in nx.connected_components(G):
            edge_pair = []
            if b != largest_net:
                index1 = random_generator(net_feature[2], net_feature[3])
                index2 = random_generator(net_feature[2], net_feature[3])
                while (index1 not in b or index2 not in largest_net):
                    index1 = random_generator(net_feature[2], net_feature[3])
                    index2 = random_generator(net_feature[2], net_feature[3])
                edge_pair.append(index1)
                edge_pair.append(index2)
                edge_pair.append(connection_type_check(index1, index2, node_sum))
                edge_sum.append(edge_pair)

    # 检查子网相似性
    # print(subnet_edge)
    # 

    while (len(edge_sum) / len(node_sum) <= 1.22):
        edge_pair = []
        state = True
        count = 0  # 若多次生成任不满足，则该子网边已达上限
        while state == True:
            edge_pair = []
            start_node = random_generator(1, len(node_sum))
            end_node = random_generator(1, len(node_sum))
            while (start_node == end_node):
                end_node = random_generator(1, len(node_sum))
            edge_pair = [start_node, end_node]
            # print(edge_sum)
            if (edge_pair not in np.array(edge_sum)[:, [0, 1]].tolist() and edge_pair not in np.array(edge_sum)[:,
                                                                                             [1, 0]].tolist()):
                edge_pair.append(connection_type_check(start_node, end_node, node_sum))
                edge_sum.append(edge_pair)
                break
            else:
                count += 1
                if count > 10:
                    state = False

    sub_similarity = sub_similarity_check(np.array(edge_sum)[:, [0, 1]], node_sum)
    print("subnet: " + str(sub_similarity))

    print("------------------------------------------------------------------------------")
    print("node sum:", node_sum)
    print("edge sum:", edge_sum)
    print("feature:", net_feature)

    # print('sub_similarity',sub_similarity)
    final_similarity_check(edge_sum, node_sum)
    # print('total_similarity',total_similarity)

    # 结构图和真实图相似度检查
    result = 0
    while True:
        code, description = distribution_code_generation(edge_sum, node_sum, graph_type)

        # 运行代码保存图片
        if graph_type == 1:
            # 运行代码,保存图片
            filename = 'bus_system_' + str(bus_number) + "_" + str(timestamp)
            head = '''
    from graphviz import Graph
    gz=Graph("Bus system ''' + str(len(node_sum)) + '''",'comment',None,None,'png',None,"UTF-8",
    {'bgcolor':'white','rankdir':'TB','splines':'ortho'},
    {'color':'black','fontcolor':'black','fontsize':'12','shape':'box','height':'0.05','length':'2','style':'filled'})
    '''
            end = '''
    print(gz.source)
    gz.render(filename = "''' + filename + '''",directory="Output/Distribution Grid/Image/")
    '''
            code_summary = head
            code_summary = code_summary + code + '\n' + end
            code_name = 'structure_image_' + str(bus_number) + "_" + str(timestamp)
            save_to_file('Output/Distribution Grid/Bus code/' + code_name + '.py', code_summary)

            result = os.system(f"""python "Output/Distribution Grid/Bus code/{code_name}.py" """)

            # plt.figure(figsize=(8, 6))
            # G = nx.from_edgelist(np.array(edge_sum)[:,[0,1]])
            # position = nx.kamada_kawai_layout(G)
            # nx.draw_networkx(G, node_size=2, width=0.1, pos = position)
            # # Construct file name
            # file_name = 'spring_image_'+str(bus_number)+"_"+str(timestamp)+'.png'
            # plt.savefig('Output/Distribution Grid/Image/' + file_name, dpi=800)
            # plt.close()

        elif graph_type == 2:
            plt.figure(figsize=(8, 6))
            G = nx.from_edgelist(np.array(edge_sum)[:, [0, 1]])
            position = nx.kamada_kawai_layout(G)
            nx.draw_networkx(G, node_size=2, width=0.1, pos=position)
            # Construct file name
            code_name = 'spring_image_' + str(bus_number) + "_" + str(timestamp) + '.png'
            plt.savefig('Output/Distribution Grid/Image/' + code_name, dpi=800)
            plt.close()

        if result == 0:
            break

    describe_name = 'description ' + str(bus_number) + "_" + str(timestamp) + '.txt'
    save_to_file('Output/Distribution Grid/Bus description/' + describe_name, description)

    edge_list = pd.DataFrame(edge_sum)
    edge_list.columns = ['Start', 'End', 'Edge type']
    edge_name = 'edge_index ' + str(bus_number) + "_" + str(timestamp) + '.csv'
    edge_list.to_csv('Output/Distribution Grid/Edge index/' + edge_name, index=False)

    node_list = pd.DataFrame(node_sum)
    node_list.columns = ['ID', 'Type']
    node_name = 'node_list ' + str(bus_number) + "_" + str(timestamp) + '.csv'
    node_list.to_csv('Output/Distribution Grid/Node feature/' + node_name, index=False)

    return 'Output/Distribution Grid/Edge index/' + edge_name, 'Output/Distribution Grid/Node feature/' + node_name, 'Output/Distribution Grid/Bus code/' + code_name + '.py'


if __name__ == "__main__":
    scale_type = 2  # 电网尺寸（1:25个bus左右； 2：26-50； 3：51-75； 4：76-100，）
    graph_type = 1  # 1:graphviz电网图， 2：networkx网络图
    include_component = 0  # 仅在graphviz电网图中触发
    main(scale_type, graph_type)
