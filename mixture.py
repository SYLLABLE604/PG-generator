import math
import torch
import numpy as np
from math import *
import networkx as nx
import pandas as pd
from datetime import datetime
from Tool import save_to_file,read_file,random_generator,output_check,sub_similarity_check,final_similarity_check,connection_type_check
from Image_gen import gemini_prompt,net_description,mix_image_code_generate
import os
import matplotlib.pyplot as plt
from pathlib import Path

def main(scale_type, subnet_num, graph_type):
    index = 1
    node_sum = []
    edge_sum = []
    mainnet_feature_sum = []
    main_node = []
    main_edge = []
    sub_node_sum = []
    sub_edge_sum = []
    subnet_feature_sum = []  # subnet_bus_number, generator_num, start_num, end_num
    interface_edge = []
    random_edge = []
    sub_similarity = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    # transmission net generation
    generator_num = 1
    main_num = subnet_num + generator_num
    mainnet_feature_sum = [main_num, generator_num, index, index + main_num - 1]

    for i in range(main_num):
        main_node_feature = []
        main_node_feature.append(index)
        if index == 1:
            main_node_feature.append(2) # slack bus
        elif (index <= generator_num):
            main_node_feature.append(1)
        elif (generator_num < index <= main_num):
            main_node_feature.append(0)
        main_node_feature.append(0)
        index += 1
        node_sum.append(main_node_feature)
        main_node.append(main_node_feature)

    # main_net edge
    for i in range(1, main_num + 1):
        edge = []
        if i == main_num:
            edge.append(i)
            edge.append(1)
            edge.append(connection_type_check(i, 1, node_sum))
            edge.append(0)
        else:
            edge.append(i)
            edge.append(i + 1)
            edge.append(connection_type_check(i, i + 1, node_sum))
            edge.append(0)
        edge_sum.append(edge)
        main_edge.append(edge)

    # 子网描述生成
    start = index
    for i in range(subnet_num):
        subnet_feature = []
        if scale_type == 1:
            subnet_feature.append(random_generator(5, ceil(25 / subnet_num)))
            subnet_feature.append(ceil(subnet_feature[0] / 5))
        elif scale_type == 2:
            subnet_feature.append(random_generator(ceil(26 / subnet_num), ceil(50 / subnet_num)))
            subnet_feature.append(ceil(subnet_feature[0] / 5))
        elif scale_type == 3:
            subnet_feature.append(random_generator(ceil(51 / subnet_num), ceil(75 / subnet_num)))
            subnet_feature.append(ceil(subnet_feature[0] / 5))
        elif scale_type == 4:
            subnet_feature.append(random_generator(ceil(76 / subnet_num), ceil(100 / subnet_num)))
            subnet_feature.append(ceil(subnet_feature[0] / 5))
        else:
            break
        subnet_feature.append(start)
        end = start + subnet_feature[0] - 1
        subnet_feature.append(end)
        start = end + 1
        subnet_feature_sum.append(subnet_feature)

    # 子网节点生成
    for i in range(len(subnet_feature_sum)):
        for k in range(subnet_feature_sum[i][0]):
            node_feature = []
            node_feature.append(index)
            if k == 0:
                node_feature.append(2)
            elif (k <= subnet_feature_sum[i][1] - 1):
                node_feature.append(1)
            elif (subnet_feature_sum[i][1] - 1 < k < subnet_feature_sum[i][0]):
                node_feature.append(0)
            node_feature.append(i+1)
            index += 1
            sub_node_sum.append(node_feature)
            node_sum.append(node_feature)

    # 子网边生成
    start = 0
    end = subnet_feature_sum[0][0]
    p = 0
    while p < len(subnet_feature_sum):
        subnet_edge = []
        subnet_node = sub_node_sum[start:end]
        print(start)
        print(end)
        print(subnet_feature_sum[p][0])
        # print(subnet_node)

        # 随机连边
        for z in range(len(subnet_node)):
            edge_pair = []
            start_node = subnet_node[z][0]
            rest_graph = list(subnet_node)
            rest_graph.remove(rest_graph[z])

            a = random_generator(0, len(rest_graph))
            end_node = rest_graph[a][0]

            exist_edge = []
            for i in range(len(subnet_edge)):
                group = []
                group.append(subnet_edge[i][0])
                group.append(subnet_edge[i][1])
                exist_edge.append(group)

            while (([start_node, end_node] in exist_edge) or ([end_node, start_node] in exist_edge)):
                # print(f"edge exist {start_node},{end_node}")
                end_node = rest_graph[random_generator(0, len(rest_graph))][0]

            edge_pair.append(start_node)
            edge_pair.append(end_node)
            edge_pair.append(connection_type_check(start_node, end_node, node_sum))
            edge_pair.append(p+1)
            subnet_edge.append(edge_pair)

        # 检查连通性
        G = nx.from_edgelist(np.array(subnet_edge)[:,[0,1]])
        if nx.is_connected(G) == False:
            largest_net = max(nx.connected_components(G), key=len)
            for b in nx.connected_components(G):
                edge_pair = []
                if b != largest_net:
                    index1 = random_generator(subnet_feature_sum[p][2], subnet_feature_sum[p][3])
                    index2 = random_generator(subnet_feature_sum[p][2], subnet_feature_sum[p][3])
                    while (index1 not in b or index2 not in largest_net):
                        index1 = random_generator(subnet_feature_sum[p][2], subnet_feature_sum[p][3])
                        index2 = random_generator(subnet_feature_sum[p][2], subnet_feature_sum[p][3])
                    edge_pair.append(index1)
                    edge_pair.append(index2)
                    edge_pair.append(connection_type_check(index1, index2, node_sum))
                    edge_pair.append(p+1)
                    subnet_edge.append(edge_pair)

        # 检查子网相似性
        # print(subnet_edge)
        similarity = sub_similarity_check(np.array(subnet_edge)[:,[0,1]],subnet_node)
        print("subnet " + str(p+1) +": "+ str(similarity))

        if(similarity > 0.5):
            for q in range(len(subnet_edge)):
                edge_pair = []
                edge_pair.append(subnet_edge[q][0])
                edge_pair.append(subnet_edge[q][1])
                edge_pair.append(subnet_edge[q][2])
                edge_pair.append(subnet_edge[q][3])
                sub_edge_sum.append(edge_pair)
                edge_sum.append(edge_pair)
            # 连接主网,子网的发电机和slackbus不能接入
            edge_pair = []
            connect_to = p + mainnet_feature_sum[1] + 1
            connect_from = random_generator(subnet_feature_sum[p][2] + subnet_feature_sum[p][1], subnet_feature_sum[p][3])
            edge_pair.append(connect_to)
            edge_pair.append(connect_from)
            edge_pair.append(connection_type_check(connect_to, connect_from, node_sum))
            edge_pair.append(-1)
            edge_sum.append(edge_pair)
            interface_edge.append(edge_pair)
            # sub_edge_sum.append(edge_pair)#需要把networkx处理好的数据取出来
            p += 1 
            start = end
            if( p != len(subnet_feature_sum)):
                end = end + subnet_feature_sum[p][0]
            # nx.draw(G,node_size = 2,width = 0.1)



    while (len(edge_sum) / len(node_sum) <= 1.5):
        edge_pair = []
        append_grid = random_generator(1,len(subnet_feature_sum))
        state = True
        count = 0 #若多次生成任不满足，则该子网边已达上限
        while append_grid == 0 and mainnet_feature_sum[0] <= 3:
            append_grid = random_generator(1,len(subnet_feature_sum))
        if append_grid == 0 and mainnet_feature_sum[0] > 3: #输电网加边
            while state == True:
                start_node = random_generator(mainnet_feature_sum[2], mainnet_feature_sum[3])
                end_node = random_generator(mainnet_feature_sum[2], mainnet_feature_sum[3])
                num = 0
                while (start_node == end_node):
                    num += 1
                    end_node = random_generator(mainnet_feature_sum[2], mainnet_feature_sum[3])
                    if num > 10:
                        state = False
                        break
                edge_pair = [start_node,end_node]
                # print(np.array(main_edge)[:,[0,1]].tolist() )
                if(edge_pair not in np.array(main_edge)[:,[0,1]].tolist() ):
                    edge_pair.append(connection_type_check(start_node,end_node,node_sum))
                    edge_pair.append(0)
                    main_edge.append(edge_pair)
                    edge_sum.append(edge_pair)
                    break
                else:
                    count += 1
                    if count > 10:
                        state = False
        else:
            while state == True:
                edge_pair = []
                start_node = random_generator(subnet_feature_sum[append_grid-1][2], subnet_feature_sum[append_grid-1][3])
                end_node = random_generator(subnet_feature_sum[append_grid-1][2], subnet_feature_sum[append_grid-1][3])
                while (start_node == end_node and state == True):
                    end_node = random_generator(subnet_feature_sum[append_grid-1][2], subnet_feature_sum[append_grid-1][3])
                edge_pair = [start_node,end_node]
                # print(edge_sum)
                if(edge_pair not in np.array(edge_sum)[:,[0,1]].tolist() and edge_pair not in np.array(edge_sum)[:,[1,0]].tolist()):
                    edge_pair.append(connection_type_check(start_node,end_node,node_sum))
                    edge_pair.append(append_grid)
                    edge_sum.append(edge_pair)
                    break
                else:
                    count += 1
                    if count > 10:
                        state = False

        

    # G = nx.from_edgelist(e[:2] for e in subnet_edge)
    # for i in range(len(sub_node_sum)):
    #     nx.set_node_attributes(G, {i+main_num+1: sub_node_sum[i][1]}, 'label')
    # nx.draw(G, node_size=2, width=0.1)
    total_similarity = final_similarity_check(edge_sum,node_sum)

    
    bus_number = len(node_sum)
    print("main feature:", mainnet_feature_sum)
    print("main node:", main_node)
    print("main edge:", main_edge)
    print("------------------------------------------------------------------------------")
    print("subnet feature", subnet_feature_sum)
    print("subnet node", sub_node_sum)
    print("subnet edge", sub_edge_sum)
    print("------------------------------------------------------------------------------")
    print("node sum:", node_sum)
    print("edge sum:", edge_sum)
    print("interface edge:", interface_edge)
    # print('sub_similarity',sub_similarity)
    # print('total_similarity',total_similarity)

    result = 0
    while True:
        # 结构图和真实图相似度检查
        code, description = mix_image_code_generate(mainnet_feature_sum, main_edge, subnet_feature_sum, sub_edge_sum,
                                                    node_sum, interface_edge, graph_type)
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
        gz.render(filename = "''' + filename + '''",directory="Output/Mixture Grid/Image/")
        '''
            code_summary = head
            code_summary = code_summary + code + '\n' + end
            code.replace('    ', '')
            code_name = 'bus_system_image_' + str(bus_number) + "_" + str(timestamp)
            save_to_file('Output/Mixture Grid/Bus code/' + code_name + '.py', code_summary)

            result = os.system(f"""python "Output/Mixture Grid/Bus code/{code_name}.py" """)

        elif graph_type == 2:
            plt.figure(figsize=(8, 6))
            G = nx.from_edgelist(np.array(edge_sum)[:, [0, 1]])
            position = nx.kamada_kawai_layout(G)
            nx.draw_networkx(G, node_size=6, width=0.3, pos=position, font_size=9)
            # Construct file name
            code_name = 'bus_system_image_' + str(bus_number) + "_" + str(timestamp) + '.png'
            plt.savefig('Output/Mixture Grid/Image/' + code_name, dpi=800)
            plt.close()

        if result == 0:
            break

    describe_name = 'descriptiom_bus_system '+str(bus_number)+"_"+str(timestamp)+'.txt'
    save_to_file('Output/Mixture Grid/Bus description/'+describe_name, description)

    # edge_list = pd.DataFrame(edge_sum)
    # edge_list.columns=['Start','End','Edge type','Subgrid']
    # edge_name = 'bus_system_edge_index '+str(bus_number)+"_"+str(timestamp)+'.csv'
    # edge_list.to_csv('Output/Mixture Grid/Edge index/'+edge_name,index=False)
    #
    #
    # node_list = pd.DataFrame(node_sum)
    # node_list.columns=['ID','Type','Subgrid']
    # node_name = 'bus_system_node_list  '+str(bus_number)+"_"+str(timestamp)+'.csv'
    # node_list.to_csv('Output/Mixture Grid/Node feature/'+node_name,index=False)
    edge_list = pd.DataFrame(edge_sum)
    edge_list.columns = ['Start', 'End', 'Edge type', 'Subgrid']

    # Define the output path using Path objects
    edge_output_dir = Path(__file__).parent / 'Output/Mixture Grid/Edge index'
    edge_output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    # Create the file name and save as CSV
    edge_name = f'bus_system_edge_index_{bus_number}_{timestamp}.csv'
    edge_list.to_csv(edge_output_dir / edge_name, index=False)

    # Convert node_sum to DataFrame
    node_list = pd.DataFrame(node_sum)
    node_list.columns = ['ID', 'Type', 'Subgrid']

    # Define the output path for nodes using Path objects
    node_output_dir = Path(__file__).parent / 'Output/Mixture Grid/Node feature'
    node_output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    # Create the file name and save as CSV
    node_name = f'bus_system_node_list_{bus_number}_{timestamp}.csv'
    node_list.to_csv(node_output_dir / node_name, index=False)

    return 'Output/Mixture Grid/Edge index/' + edge_name, 'Output/Mixture Grid/Node feature/' + node_name, 'Output/Mixture Grid/Bus code/' + code_name + '.py'


if __name__ == "__main__":
    grid_type = 1 #电网类型（1，包含传输网和配电网；2，只包含配电网）
    scale_type = 1 #电网尺寸（1:25个bus左右； 2：26-50； 3：51-75； 4：76-100，）
    subnet_num = 1 #子网数量（1：<4; 2:< 6; 3<7; 4<9；只包含配电网只为1）
    graph_type = 2 # 1:graphviz电网图， 2：networkx网络图
    main(grid_type, scale_type, subnet_num, graph_type)




