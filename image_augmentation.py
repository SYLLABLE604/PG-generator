import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.output_parsers import CommaSeparatedListOutputParser, PydanticOutputParser
import os
from langchain.prompts import PromptTemplate
from difflib import SequenceMatcher
from pydantic import BaseModel, Field, validator
import logging
import re
from pathlib import Path

def clean_code_block(text):
    text = text.strip()

    if text.startswith("```"):
        text = text[3:].strip()

    first_line_end = text.find('\n')
    if first_line_end != -1:
        first_line = text[:first_line_end].strip()
        if first_line.isalpha():
            text = text[first_line_end + 1:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text

def type_translation(number, type):
    if type == 1:
        array = ['PV', 'PQ', 'slack']
        return array[number]
    elif type == 2:
        array = ['PV to PV', 'PV to PQ', 'PQ to PQ']
        return array[number - 1]


def query_generator(edge_feature, node_feature):
    description = "This power grid includes "
    # node
    if 0 in node_feature['Subgrid']:
        description += "transmission grid and distribution grids. For node types, in transmission grid include "
        for i, row in node_feature.iterrows():
            if row['Subgrid'] == 0:
                description += (type_translation(row['Type'], 1) + ' bus, ')
    else:
        description += "distribution grids. For node types, "

    node_feature = node_feature[node_feature['Subgrid'] != 0].drop_duplicates(subset=['Type'])
    description += "In distribution grids include "
    for i, row in node_feature.iterrows():
        if row['Subgrid'] != 0 and i == len(node_feature) - 1:
            description += (type_translation(row['Type'], 1) + ' bus. ')
        elif row['Subgrid'] != 0:
            description += (type_translation(row['Type'], 1) + ' bus, ')
    # edge
    if 0 in edge_feature['Subgrid']:
        description += "For edge types, in transmission grid include "
        for i, row in edge_feature.iterrows():
            if row['Subgrid'] == 0:
                description += (type_translation(row['Edge type'], 2) + ' edge, ')
    else:
        description += "For edge types, "

    edge_feature = edge_feature[edge_feature['Subgrid'] != 0].drop_duplicates(subset=['Edge type'])
    description += "In distribution grids include "
    for i, row in edge_feature.iterrows():
        if row['Subgrid'] != 0 and i == len(edge_feature) - 1:
            description += (type_translation(row['Edge type'], 2) + ' edge. ')
        elif row['Subgrid'] != 0:
            description += (type_translation(row['Edge type'], 2) + ' edge, ')
    return description

def load_txt_files(directory):
    txt_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            txt_loader = TextLoader(filepath)
            txt_data.extend(txt_loader.load())
    return txt_data

def main(edge_index_path, node_index_path, python_path):
    # 设置API密钥和代理
    
    api_key = "replace with your key"

    txt_data_dir = Path(__file__).parent / "reference grid" / "txt"
    TXT_data = load_txt_files(txt_data_dir)
    # total_data = TXT_data + csv_data_node + csv_data_edge
    # total_data = TXT_data + pdf_data
    total_data = TXT_data

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=5)
    all_splits = text_splitter.split_documents(total_data)

    # embedding = AzureOpenAIEmbeddings(
    #     azure_deployment="<GPT4o>",
    #     openai_api_version="2024-04-01-preview",
    # )
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    persist_directory = 'Output/Augmented image/Intermediate output/database'
    vectordb = Chroma(embedding_function=embedding, persist_directory=persist_directory)
    vectordb.delete_collection()
    vectordb = Chroma.from_documents(documents=all_splits, embedding=embedding)

    # 读取表格，计算特征
    edge_feature = []
    node_feature = []
    edge_data = pd.read_csv(edge_index_path)
    node_data = pd.read_csv(node_index_path)

    edge_feature = edge_data.drop_duplicates(subset=['Edge type', 'Subgrid'])
    node_feature = node_data.drop_duplicates(subset=['Type', 'Subgrid'])
    print(node_feature)

    description = query_generator(edge_feature, node_feature)
    print("1. ")
    print(description)
    # question: this power grid is made up of transmission grid and distribution grid; In transmission grid we have PQ, PV and slack bus; In distribution grid we have PQ buses
    question = description + ' I want to know the detailed structure of each component.'

    # 输出拆分后的query
    # 这个 prompt 的输出不是很稳定
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template="""You are an AI language model assistant. Your task is to dicompose the given question for retrieving relevant documents from a vector database. 
    By decompose the user question, your goal is to help the user overcome some of the limitations of similarity-based search. These questions should be only focus on only one type of bus (PQ, PV or slack bus in transmission grid or distribution grid)
    Provide sub questions in the following example formate, return questions split by comma.
        Example:
            Question: This power grid include transmission grid and distribution grids. For node types, in transmission grid include slack bus, PV bus, PQ bus, In distribution grids include slack bus, PQ bus, For edge types, in transmission grid include PV to PQ edge, PQ to PQ edge, In distribution grids include PV to PQ edge, PQ to PQ edge, PV to PV edge,
            Answer: "What is the structure of PQ bus in transmission grid? " ,"What is the structure of PV bus in transmission grid", "What is the structure of slack bus in transmission grid","What is the structure of PQ bus in distribution grid","What is the structure of slack bus in distribution grid",
    "What is the structure of PV to PQ edge in transmission grid", "What is the structure of PQ to PQ edge in transmission grid", "What is the structure of PV to PQ edge in distribution grid", "What is the structure of PQ to PQ edge in distribution grid", "What is the structure of PV to PV edge in distribution grid"
        Original question: {question} \n {format_instructions}""",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )

    _input = prompt.format(question=question)
    output = llm(_input)
    query_list = output_parser.parse(output)
    print("2. ")
    print(query_list)

    # 可以考虑增强query
    # retriever_from_llm = MultiQueryRetriever.from_llm(
    #     retriever=vectordb.as_retriever(), llm=llm, prompt=QUERY_PROMPT
    # )

    # 输出检索出来的内容
    # parser = PydanticOutputParser(pydantic_object=Answer_percentage)
    prompt = PromptTemplate(
        template="You are an AI energy system and power grid design assistant. Your task is verify to exactly which answers I provide in the following can answer the question. Your must choose which of the answers in the results exactly and answer me only the serial number of the answer and the exact text of this question, please don't give me anything else. question:{question}\n answer:{answer}\n ",
        input_variables=["question", "answer"],
        # partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Generate colored description
    file_path = node_index_path
    data = pd.read_csv(file_path)
    type_mapping = {0: "PQ bus", 1: "PV bus", 2: "slack bus"}
    unique_subgrids = data['Subgrid'].unique()
    subgrid_mapping = {0: "transmission grid"}
    for i, subgrid in enumerate(unique_subgrids):
        if subgrid != 0:
            subgrid_mapping[subgrid] = f"distribution grid {i}"

    pq_transmission = "Only change the node's color and keep its format unchanged. PQ transmission grid includes the following buses: \n"
    pv_transmission = "Only change the node's color and keep its format unchanged. PV transmission grid includes the following buses:\n"
    slack_transmission = "Slack transmission grid includes the following buses:\n"
    pq_distribution = "Only change the node's color and keep its format unchanged. PQ distribution grids include the following buses:\n"
    pv_distribution = "Only change the node's color and keep its format unchanged. PV distribution grids include the following buses:\n"
    slack_distribution = "Slack distribution grids include the following buses:\n"

    has_pq_transmission = False
    has_pv_transmission = False
    has_slack_transmission = False
    has_pq_distribution = False
    has_pv_distribution = False
    has_slack_distribution = False

    unique_subgrids = data['Subgrid'].unique()
    print(unique_subgrids)

    for subgrid in unique_subgrids:
        subgrid_name = subgrid_mapping[subgrid]
        subgrid_data = data[data['Subgrid'] == subgrid]

        for _, row in subgrid_data.iterrows():
            bus_id = row['ID']
            bus_type = type_mapping[row['Type']]
            if row['Type'] == 0:
                if subgrid == 0:
                    pq_transmission += f"bus {bus_id} is {bus_type}. this node's color pq_transmission bus must be purple. \n"
                    has_pq_transmission = True
                else:
                    pq_distribution += f"bus {bus_id} in {subgrid_name}. this node's color pq_distribution bus must be red. \n"
                    has_pq_distribution = True
            elif row['Type'] == 1:
                if subgrid == 0:
                    pv_transmission += f"bus {bus_id} is {bus_type}. this node's color pv_transmission bus must be green. \n"
                    has_pv_transmission = True
                else:
                    pv_distribution += f"bus {bus_id} in {subgrid_name}. this node's color pv_distribution bus must be blue. \n"
                    has_pv_distribution = True
            elif row['Type'] == 2:
                if subgrid == 0:
                    slack_transmission += f"bus {bus_id} is {bus_type}. this node's color pv_transmission bus must be green.\n"
                    has_slack_transmission = True
                else:
                    slack_distribution += f"bus {bus_id} in {subgrid_name}. this node's color pv_distribution bus must be blue.\n"
                    has_slack_distribution = True

    output_dir = 'reference grid/colored_node'
    os.makedirs(output_dir, exist_ok=True)

    if has_pq_transmission:
        with open(os.path.join(output_dir, 'pq_transmission.txt'), 'w') as file:
            file.write(pq_transmission)
    if has_pv_transmission:
        with open(os.path.join(output_dir, 'pv_transmission.txt'), 'w') as file:
            file.write(pv_transmission)
    if has_slack_transmission:
        with open(os.path.join(output_dir, 'slack_transmission.txt'), 'w') as file:
            file.write(slack_transmission)
    if has_pq_distribution:
        with open(os.path.join(output_dir, 'pq_distribution.txt'), 'w') as file:
            file.write(pq_distribution)
    if has_pv_distribution:
        with open(os.path.join(output_dir, 'pv_distribution.txt'), 'w') as file:
            file.write(pv_distribution)
    if has_slack_distribution:
        with open(os.path.join(output_dir, 'slack_distribution.txt'), 'w') as file:
            file.write(slack_distribution)
    print("Descriptions have been saved to the 'colored_node' directory.")

    PV_transmission_detail = False
    PV_distribution_detail = False
    PQ_distribution_detail = False
    PQ_transmission_detail = False
    slack_distribution_detail = False
    slack_transmission_detail = False

    output_map = {}
    # 将检索结果输入llm检验是否有效
    for i in range(len(query_list)):
        print(query_list[i])

        all_results = ""
        if not query_list[i].strip():
            continue
        search_results = vectordb.similarity_search(query_list[i])
        for j, result in enumerate(search_results):
            all_results += f"{j + 1}. {result.page_content}\n\n"
        question = query_list[i] + "I want to know the detailed structure of each component."
        _input = prompt.format_prompt(question=question, answer=all_results)
        output = llm(_input.to_string())
        print(output)

        text = "1. This flowchart describes the process of integrating PV (photovoltaic) and wind power into the main grid. Initially, the wind or hydropower generator converts mechanical energy into electrical energy, with a portion of the electricity stored in batteries. The remaining electricity is passed through a step-up transformer to increase the voltage, meeting the requirements of long-distance transmission. To enhance economic efficiency and transmission effectiveness, the AC power is converted to DC power via a rectifier and transmitted over long distances through a DC transmission line. During transmission, reactive power compensators such as synchronous condensers or STATCOMs are used to compensate for reactive power, ensuring the stability and efficiency of power transmission." \
               "2. This flowchart describes the process of integrating PV (or thermal power) into the main grid. Initially, the generator converts mechanical energy into electrical energy, and a step-up transformer increases the voltage to meet the requirements of long-distance transmission. To enhance economic efficiency and transmission effectiveness, the AC power is converted to DC power via a rectifier and transmitted over long distances through a DC transmission line. During transmission, reactive power compensators such as synchronous condensers or STATCOMs are used to compensate for reactive power, ensuring the stability and efficiency of power transmission. Finally, circuit breakers and switches control the flow of electricity, ensuring the safety and stability of the circuit." \
               "3. This flowchart describes the process of PQ integration into a load. Initially, the electrical power passes through a 10kV switch at the substation into the medium voltage ring main unit. Next, the power is further distributed through the switching station, and finally, it enters the distribution substation or overhead transformer for use by the load." \
               "4. This flowchart illustrates the process of PQ integration into the main grid, focusing on the operation of a substation that serves only heavy industrial users and acts as the source of the distribution network. Initially, the high voltage power is protected by a lightning arrester, followed by a high voltage circuit breaker to ensure circuit safety. Subsequently, the power passes through an isolating switch for further isolation and maintenance convenience. The power then goes through a reactor for inductance adjustment and is converted by a transformer from high voltage to low voltage suitable for distribution. Finally, the low voltage power is compensated for reactive power by a capacitor, ensuring the stability and efficiency of distribution."
        # 如果可以，给llm描述以及代码范例，生成单个节点的连接关系代码，运行生成图
        match = re.search(r'of\s+(\S+)\s+bus', query_list[i])
        if match:
            extracted_word = match.group(1)
            print(f'1st word: {extracted_word}')
        else:
            extracted_word = 'null'
            print('No match found')

        match2 = re.search(r'in\s+(.+?)\s+grid', query_list[i])
        if match2:
            extracted_word2 = match2.group(1).replace(' ', '')
            print(f'2nd word: {extracted_word2}')
        else:
            extracted_word2 = 'null'
            print('No match found 2nd')
        bus_key = extracted_word + ' ' + extracted_word2
        print(bus_key)

        # no match found case

        if (extracted_word == 'PV' and extracted_word2 == 'transmission'):
            print("!!")
            prompt_path = Path(
                __file__).parent / "reference grid" / "rag_test" / "prompt_pv_power_integration_main_1.txt"
            with prompt_path.open("r") as f:
                txt_content = f.read()
            final_prompt = f"Based on answer: '{output}', '{txt_content}' The title of the whole diagram must be green and use Mermaid"
            final_output = "" + llm(final_prompt)
            PV_transmission_detail = True

        elif (extracted_word == 'PV' and extracted_word2 == 'distribution'):
            print("2!")
            prompt_path = Path(__file__).parent / "reference grid" / "rag_test" / "prompt_PV_Power_Integration_2.txt"
            with prompt_path.open("r") as f:
                txt_content = f.read()
            final_prompt = f"Based on answer: '{output}', '{txt_content}'  The title of the whole diagram must be blue and use Mermaid"
            final_output = '' + llm(final_prompt)
            PV_distribution_detail = True

        elif (extracted_word == 'PQ' and extracted_word2 == 'distribution'):
            print("3!")
            prompt_path = Path(__file__).parent / "reference grid" / "rag_test" / "prompt_pq_integration_load_3.txt"
            with prompt_path.open("r") as f:
                txt_content = f.read()
            final_prompt = f"Based on answer: '{output}', '{txt_content}'  The title of the whole diagram must be red and use Mermaid"
            final_output = "" + llm(final_prompt)
            PQ_distribution_detail = True

        elif (extracted_word == 'PQ' and extracted_word2 == 'transmission'):
            print("4!")
            prompt_path = Path(__file__).parent / "reference grid" / "rag_test" / "prompt_pq_integration_main_4.txt"
            with prompt_path.open("r") as f:
                txt_content = f.read()
            final_prompt = f"Based on answer: '{output}', '{txt_content}'  The title of the whole diagram must be purple and use Mermaid"
            final_output = "" + llm(final_prompt)
            PQ_transmission_detail = True

        elif (extracted_word == 'slack' and extracted_word2 == 'transmission'):
            print("5!")
            prompt_path = Path(
                __file__).parent / "reference grid" / "rag_test" / "prompt_pv_power_integration_main_1.txt"
            with prompt_path.open("r") as f:
                txt_content = f.read()
            final_prompt = f"Based on answer: '{output}', '{txt_content}' The title of the whole diagram must be green and use Mermaid"
            final_output = "" + llm(final_prompt)
            slack_transmission_detail = True

        elif (extracted_word == 'slack' and extracted_word2 == 'distribution'):
            print("6!")
            prompt_path = Path(__file__).parent / "reference grid" / "rag_test" / "prompt_PV_Power_Integration_2.txt"
            with prompt_path.open("r") as f:
                txt_content = f.read()
            final_prompt = f"Based on answer: '{output}', '{txt_content}'  The title of the whole diagram must be blue and use Mermaid"
            final_output = '' + llm(final_prompt)
            slack_distribution_detail = True

        else:
            print("error type!")
            final_output = "error type!"

        print(final_output)

        output_map[bus_key] = final_output

        print("---------------------------")

    print("Validation Output and Final Output Mapping:")
    for key, value in output_map.items():
        print(f"Validation Output: {key}, Final Output: {value}")

    # Define the output directory
    output_dir2 = Path(__file__).parent / 'Output' / 'Augmented image' / 'Intermediate output'
    output_dir2.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    # Initialize description and detailed code
    final_color_description = ""
    total_detailed_code = ""

    # Read color descriptions based on condition
    if has_pq_transmission and PQ_transmission_detail:
        key_text = 'PQ transmission'
        pq_transmission_code = output_map[key_text]
        total_detailed_code += pq_transmission_code
        pq_transmission_path = Path(__file__).parent / "reference grid" / "colored_node" / "pq_transmission.txt"
        pq_transmission_content = pq_transmission_path.read_text()
        final_color_description += pq_transmission_content

    if has_pv_transmission and PV_transmission_detail:
        key_text = 'PV transmission'
        pv_transmission_code = output_map[key_text]
        total_detailed_code += pv_transmission_code
        pv_transmission_path = Path(__file__).parent / "reference grid" / "colored_node" / "pv_transmission.txt"
        pv_transmission_content = pv_transmission_path.read_text()
        final_color_description += pv_transmission_content

    if has_pq_distribution and PQ_distribution_detail:
        key_text = 'PQ distribution'
        pq_distribution_code = output_map[key_text]
        total_detailed_code += pq_distribution_code
        pq_distribution_path = Path(__file__).parent / "reference grid" / "colored_node" / "pq_distribution.txt"
        pq_distribution_content = pq_distribution_path.read_text()
        final_color_description += pq_distribution_content

    if has_pv_distribution and PV_distribution_detail:
        key_text = 'PV distribution'
        pv_distribution_code = output_map[key_text]
        total_detailed_code += pv_distribution_code
        pv_distribution_path = Path(__file__).parent / "reference grid" / "colored_node" / "pv_distribution.txt"
        pv_distribution_content = pv_distribution_path.read_text()
        final_color_description += pv_distribution_content

    if has_slack_distribution and slack_distribution_detail:
        key_text = 'slack distribution'
        slack_distribution_code = output_map[key_text]
        total_detailed_code += slack_distribution_code
        slack_distribution_path = Path(__file__).parent / "reference grid" / "colored_node" / "slack_distribution.txt"
        slack_distribution_content = slack_distribution_path.read_text()
        final_color_description += slack_distribution_content

    if has_slack_transmission and slack_transmission_detail:
        key_text = 'slack transmission'
        slack_transmission_code = output_map[key_text]
        total_detailed_code += slack_transmission_code
        slack_transmission_path = Path(__file__).parent / "reference grid" / "colored_node" / "slack_transmission.txt"
        slack_transmission_content = slack_transmission_path.read_text()
        final_color_description += slack_transmission_content

    # Save the detailed code and color description to files
    (output_dir2 / 'total_detailed_code.txt').write_text(total_detailed_code)
    (output_dir2 / 'final_color_description.txt').write_text(final_color_description)

    # Load combined example
    combined_eg_path = Path(__file__).parent / "reference grid" / "rag_test" / "combined_eg.txt"
    combined_eg = combined_eg_path.read_text()

    # Create final code prompt
    final_code_prompt = f"Based on the examples:'{combined_eg}'Based on the code in parts: '{total_detailed_code}' Just plot several legends separately in one graph together as the code is now in separate codes. The code should be in Mermaid. The original title of the legends and its color should be remained unchanged. (eg. if subgraph cluster0 [<font color='green'>PV Main Integration</font>], then the title of that legend should be green!) Just give me the code and don't give me anything else (Output should be one graph with the several legends) If there are repetition legends, remove them."
    legend_output = llm(final_code_prompt)
    cleaned_legend_output = clean_code_block(legend_output)

    # Save the detailed code output
    (output_dir2 / 'detailed_code.txt').write_text(cleaned_legend_output)

    # Load the Python Graphviz code from the input path
    input_file_path = Path(python_path)
    content = input_file_path.read_text()

    # Save the example Python code
    (output_dir2 / 'example_code.txt').write_text(content)

    # Generate modified color output prompt
    modified_color_output_prompt = f"Given the Python Graphviz code and the instructions on which nodes need to be changed into a specific color. Please change the color of specific nodes in the given Python Graphviz code (mere code) according to the instructions. Only change the node's color and keep its format unchanged. Original code: '{content}' Node color change instruction: '{final_color_description}'"
    print("____________")
    print(modified_color_output_prompt)

    # Get the final prompt output and save it
    finite_prompt = llm(modified_color_output_prompt)
    cleaned_finite_prompt = clean_code_block(finite_prompt)
    (output_dir2 / 'finite.txt').write_text(cleaned_finite_prompt)



if __name__ == "__main__":
    edge_index_path = "/Users/felixyan/Desktop/proj/correct/bus_system_edge_index 45_20240820_051950.csv"
    node_index_path = "/Users/felixyan/Desktop/proj/correct/bus_system_node_list  45_20240820_051950.csv"
    python_path = '/Users/felixyan/Desktop/proj/correct/bus_system_image_45_20240820_051950.py'

    main(edge_index_path, node_index_path, python_path)
    # main()
