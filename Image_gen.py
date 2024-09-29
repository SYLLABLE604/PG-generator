#coding:utf-8

from math import *
import networkx as nx
from datetime import datetime
from Tool import read_file,output_check
import google.generativeai as genai
import os

os.environ['http_proxy']="http://10.60.2.25:3128"
os.environ['https_proxy']="http://10.60.2.25:3128"

def gemini_prompt(description,type):  #后面还可以加参数，设置不同的prompt模板,1是没有分批，2是分批的头部，3是分批的中部，4是分批的尾部
    genai.configure(api_key='AIzaSyDvDoaTUaCQpRYyEE-wGsyeEi3vaVO8xUo')
    model = genai.GenerativeModel(model_name="gemini-pro")

    if type == 1:
      prompt = read_file("Prompt/mix_graphviz_normal.txt")+description
    elif type == 2:
      prompt = read_file("Prompt/mix_graphviz_split_1.txt")+description
    elif type == 3:
      prompt = read_file("Prompt/mix_graphviz_split_2.txt")+description
    elif type == 4:
      prompt = read_file("Prompt/mix_graphviz_split_3.txt")+description
    elif type == 5:
      prompt = read_file("Prompt/dis_graphviz_normal.txt")+description
    elif type == 6:
      prompt = read_file("Prompt/dis_graphviz_split_1.txt")+description
    else:
      prompt = ""
    print(prompt)
    response = model.generate_content(prompt)
    code = output_check(response.text)
    return code

def net_description(feature_sum, node_sum, edge_sum, type, index):
    statement = ""

    grid_description = ""
    node_description = ""
    edge_description = ""
    if (type == 1):
        grid_description = "transmission grid include " + str(feature_sum[0]) + " buses, ranged from " + str(
            feature_sum[2]) + " to " + str(feature_sum[3]) + ".\n"
    else:
        grid_description = "distribution grid " + str(index) + " include " + str(
            feature_sum[0]) + " buses, ranged from " + str(
            feature_sum[2]) + " to " + str(feature_sum[3]) + ".\n"
    statement = grid_description

    node_list = node_sum[feature_sum[2] - 1:feature_sum[3]]
    for node in node_list:
        node_type = ""
        if (node[1] == 0):
            node_type = "PQ bus"
        elif (node[1] == 1):
            node_type = "PV bus"
        else:
            node_type = "slack bus"
        description = "bus " + str(node[0]) + " is " + node_type + ".\n"
        statement += description

    for edge in edge_sum:
        description = "bus " + str(edge[0]) + " is connected to bus " + str(edge[1]) + ".\n"
        statement += description
    return statement



def mix_image_code_generate(mainnet_feature_sum, main_edge, subnet_feature_sum, sub_edge_sum, node_sum,
                   interface_edge, graph_type):
    whole_grid_description = ""
    main_grid_description = ""
    sub_grid_description = []
    code = ""
    bus_number = len(node_sum)

    # whole grid description
    whole_grid_description = "This is a power grid include " + str(len(node_sum)) + " buses, which have 1 transmission grid and " + str(len(subnet_feature_sum)) + " distribution grids.\n"
    # print(whole_grid_description)

    # main grid description
    main_grid_description = net_description(mainnet_feature_sum, node_sum, main_edge, 1, 0)
    # print(main_grid_description)

    interface_description = ""
    for i in range(len(interface_edge)):
        text = "This edge is tansformer line, which is from bus " + str(interface_edge[i][0]) + " to bus " + str(
            interface_edge[i][1]) + ".\n"
        interface_description += text
    # print(interface_description)

    # sub grid description
    for i in range(len(subnet_feature_sum)):
      node_start_number = subnet_feature_sum[i][2]
      node_end_number = subnet_feature_sum[i][3]
      edge_list = []
      for edge_pair in sub_edge_sum:
        if (edge_pair[0] in range(node_start_number, node_end_number + 1) and edge_pair[1] in range(
              node_start_number, node_end_number + 1)):
          edge_list.append(edge_pair)
      sub_grid_description.append(net_description(subnet_feature_sum[i], node_sum, edge_list, 2, i + 1))

    description = whole_grid_description + main_grid_description
    for i in range(len(sub_grid_description)):
      description += sub_grid_description[i]
    description +=  interface_description

    # print(description)
    # print(bus_number)
    if graph_type == 1:
        # 分组生成,需要修改，30个bus以下直接生成，30-60个以上分两批
        group_bus_num = 0
        group_list = []
        group = []
        if bus_number <= 35:
          for i in range(len(subnet_feature_sum)):
            group.append(i)
          group_list.append(group)
        elif bus_number > 35 and bus_number <= 70:
          mid_number = int(len(subnet_feature_sum)/2)
          for i in range(mid_number):
            group.append(i)
          group_list.append(group)
          group = []
          for i in range(mid_number,len(subnet_feature_sum)):
            group.append(i)
          group_list.append(group)
        else:
          number_1 = int(len(subnet_feature_sum)/3)
          number_2 = int(len(subnet_feature_sum)/3*2)
          for i in range(number_1):
            group.append(i)
          group_list.append(group)
          group = []
          for i in range(number_1,number_2):
            group.append(i)
          group_list.append(group)
          group = []
          for i in range(number_2,len(subnet_feature_sum)):
            group.append(i)
          group_list.append(group)
        # print(group_list)


        if len(group_list) == 1:
          code = gemini_prompt(description,1)
          # print(description)
        else:
          for i in range(len(group_list)):
            # print(f"---------------------{i}----------------------")
            description_split = ""
            if i == 0:
              description_split =  main_grid_description
              for g in range(len(group_list[i])):
                description_split += sub_grid_description[group_list[i][g]]
                # print(description_split)
              code_split = gemini_prompt(description_split,2)
            elif i == len(group_list)-1:
              for g in range(len(group_list[i])):
                description_split += sub_grid_description[group_list[i][g]]
              description_split +=  interface_description
              # print(description_split)
              code_split = gemini_prompt(description_split,4)
            else:
              for g in range(len(group_list[i])):
                description_split += sub_grid_description[group_list[i][g]]
                # print(description_split)
              code_split = gemini_prompt(description_split,3)
            code += '\n' + code_split
            code = code.replace('    ', '')
            # print(code_split)

    elif graph_type == 2:
        code = ""

    return code, description

def distribution_code_generation(edge_sum,node_sum,graph_type):
  bus_number = len(node_sum)
  discription = ''
  node_description = "This is a distribution power grid include " + str(len(node_sum)) + " buses\n"
  edge_discription_list = []
  for node in node_sum:
    if (node[1] == 0):
        node_type = "PQ bus"
    elif (node[1] == 1):
        node_type = "PV bus"
    else:
        node_type = "slack bus"
    node_description += "bus " + str(node[0]) + " is " + node_type + ".\n"
  
  total_description = node_description

  for edge in edge_sum:
    dis = "bus " + str(edge[0]) + " is connected to bus " + str(edge[1]) + ".\n"
    edge_discription_list.append(dis)
    total_description += dis
  # print(edge_discription_list)

  if graph_type == 1:
    # 分组生成,需要修改，30个bus以下直接生成，30-60个以上分两批
    group = ''
    if bus_number <= 35:
      for i in range(len(edge_discription_list)):
        group += edge_discription_list[i]
      discription = node_description + group
      code = gemini_prompt(discription,5)

    elif bus_number > 35 and bus_number <= 70:
      mid_number = int(len(edge_discription_list)/2)
      for i in range(mid_number):
        group += edge_discription_list[i]
      discription = node_description + group
      code1 = gemini_prompt(discription,5)
      print(discription)
      print(code1)

      group = ''
      for i in range(mid_number,len(edge_discription_list)):
        group += edge_discription_list[i]
      discription = group
      code2 = gemini_prompt(discription,6)
      code = code1 + '\n' + code2
      print(discription)
      print(code2)
      
    else:
      number_1 = int(len(edge_discription_list)/3)
      number_2 = int(len(edge_discription_list)/3*2)
      for i in range(number_1):
        group += edge_discription_list[i]
      discription = node_description + group
      code1 = gemini_prompt(discription,5)

      group = ''
      for i in range(number_1,number_2):
        group += edge_discription_list[i]
      discription = group
      code2 = gemini_prompt(discription,6)

      group = ''
      for i in range(number_2,len(edge_discription_list)):
        group += edge_discription_list[i]
      discription = group
      code3 = gemini_prompt(discription,6)

      code = code1 + '\n' + code2 + '\n' + code3

    code = code.replace('    ', '')
    code = code.replace("'''", '')

  elif graph_type == 2:
      code = ""

  return code,total_description



