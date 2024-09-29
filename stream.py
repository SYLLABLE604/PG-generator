import zipfile
import distribution as dis
import mixture as mix
import os
import webpage_example
from datetime import datetime
import shutil
from pathlib import Path
import streamlit as st
import pyperclip
from streamlit_mermaid import st_mermaid
import image_augmentation as aug


def set_page_config():
    st.set_page_config(
        page_title='Power Grid Structure Agent',
        page_icon=' ',
        layout='wide'
    )


set_page_config()


def zip_files(zip_name, folder_paths):
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder_path in folder_paths:
            folder_base = os.path.basename(os.path.normpath(folder_path))
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join(folder_base, os.path.relpath(file_path, start=os.path.dirname(folder_path)))
                    zipf.write(file_path, arcname=arcname)


def update_progress(progress_bar, progress):
    progress_bar.progress(progress)


with st.sidebar:
    st.title('Welcome to the Power Grid Structure Generation Agent')
    st.markdown('---')
    st.markdown('Basic settings：')
    grid_type_value = 2
    grid_type = st.radio(
        "Grid Type:", ["Distribution Grid Only", "Both Transmission Grid and Distribution Grid"], key="grid_type_key"
    )
    grid_type_value = 1 if grid_type == "Both Transmission Grid and Distribution Grid" else 2
    # st.text(grid_type_value)

    size_option = st.selectbox(
        "Choose a grid scale:",
        ("Select Size", "Small —— Around 25 buses", "Medium —— 26-50 buses", "Large —— 51-75 buses",
         "Enormous —— 76-100 buses")
    )
    grid_scale_value = None

    if grid_type_value == 2:
        subnet_number = 1
        st.text('Subnet number should be 1 as the grid type is distribution grid only.')

    if size_option == "Small —— Around 25 buses":
        grid_scale_value = 1
        if grid_type_value == 1:
            subnet_number = st.slider("Select Subnet Number", 0, 3, 0)
    elif size_option == "Medium —— 26-50 buses":
        grid_scale_value = 2
        if grid_type_value == 1:
            subnet_number = st.slider("Select Subnet Number", 1, 5, 1)
    elif size_option == "Large —— 51-75 buses":
        grid_scale_value = 3
        if grid_type_value == 1:
            subnet_number = st.slider("Select Subnet Number", 1, 6, 1)
    elif size_option == "Enormous —— 76-100 buses":
        grid_scale_value = 4
        if grid_type_value == 1:
            subnet_number = st.slider("Select Subnet Number", 1, 8, 1)

    graph_type = st.radio("Graph Type:", ["IEEE Style Graph", "Force-directed Graph"])
    graph_type_value = 2 if graph_type == "Networkx Graph" else 1
    # st.text(graph_type_value)
    # if graph_type_value == 1:
    #     include_component = st.text_input(
    #         'Include Component: ', key="include_component_key"
    #     )


def page1():
    st.title('Generation')

    st.info('Example Generated Results: (in folder)')
    expander_1 = st.expander('Example Generated code and description')
    expander_2 = st.expander('Example Generated Image')
    c3, c4 = expander_1.columns(2)
    with c3:
        st.markdown('Generated Code：')
        st.code(webpage_example.example_grid_code, language='python')

    with c4:
        st.markdown('Generated Descriptions：')
        st.text(webpage_example.example_description)

    image_path_IEEE = Path(__file__).parent / "reference grid/IEEE.png"
    expander_2.image(str(image_path_IEEE), caption='IEEE test case style')
    image_path_Sparse = Path(__file__).parent / "reference grid/Sparse.png"
    expander_2.image(str(image_path_Sparse), caption='Sparse style of grid')

    if 'ready_to_download' not in st.session_state:
        st.session_state.ready_to_download = False

    st.title("Progress Bar")
    progress_bar = st.progress(0)

    # Initialize the variables with default values
    edge_temp_path = ""
    node_temp_path = ""
    python_temp_path = ""

    if st.button('Generate'):
        if grid_type_value == 2:
            # Distribution Grid logic
            path_bus_code = Path(__file__).parent / "Output/Distribution Grid/Bus code"
            shutil.rmtree(path_bus_code)
            os.mkdir(path_bus_code)

            path_bus_description = Path(__file__).parent / "Output/Distribution Grid/Bus description"
            shutil.rmtree(path_bus_description)
            os.mkdir(path_bus_description)

            path_edge_index = Path(__file__).parent / "Output/Distribution Grid/Edge index"
            shutil.rmtree(path_edge_index)
            os.mkdir(path_edge_index)

            path_image = Path(__file__).parent / "Output/Distribution Grid/Image"
            shutil.rmtree(path_image)
            os.mkdir(path_image)

            path_node_feature = Path(__file__).parent / "Output/Distribution Grid/Node feature"
            shutil.rmtree(path_node_feature)
            os.mkdir(path_node_feature)

            edge_temp_path, node_temp_path, python_temp_path = dis.main(scale_type=grid_scale_value,                                                            
                                                                        graph_type=graph_type_value)

        elif grid_type_value == 1:
            # Mixture Grid logic
            path_bus_code = Path(__file__).parent / "Output/Mixture Grid/Bus code"
            shutil.rmtree(path_bus_code)
            os.mkdir(path_bus_code)

            path_bus_description = Path(__file__).parent / "Output/Mixture Grid/Bus description"
            shutil.rmtree(path_bus_description)
            os.mkdir(path_bus_description)

            path_edge_index = Path(__file__).parent / "Output/Mixture Grid/Edge index"
            shutil.rmtree(path_edge_index)
            os.mkdir(path_edge_index)

            path_image = Path(__file__).parent / "Output/Mixture Grid/Image"
            shutil.rmtree(path_image)
            os.mkdir(path_image)

            path_node_feature = Path(__file__).parent / "Output/Mixture Grid/Node feature"
            shutil.rmtree(path_node_feature)
            os.mkdir(path_node_feature)

            edge_temp_path, node_temp_path, python_temp_path = mix.main(scale_type=grid_scale_value,
                                                                        subnet_num=subnet_number,
                                                                        graph_type=graph_type_value)

        st.session_state.edge_temp_path = edge_temp_path
        st.session_state.node_temp_path = node_temp_path
        st.session_state.python_temp_path = python_temp_path

        st.session_state.ready_to_download = True

    if st.session_state.ready_to_download:
        if grid_type_value == 2:
            folder_path = [
                Path(__file__).parent / 'Output' / 'Distribution Grid'
            ]
        elif grid_type_value == 1:
            folder_path = [
                Path(__file__).parent / 'Output' / 'Mixture Grid'
            ]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f'generated_files_{timestamp}.zip'

        zip_name = Path(__file__).parent / f'generated_files_{timestamp}.zip'
        zip_files(zip_name, folder_path)
        with open(zip_name, "rb") as fp:
            btn = st.download_button(
                label="Download Packaged Files",
                data=fp,
                file_name=zip_name.name,
                mime="application/zip"
            )
    if st.button("Jump to make augmentation grid"):
        st.session_state.page = "page2"


def page2():
    st.title("Detailed structure Power Grid Visualization")

    col1, spacer, col2 = st.columns([10, 2, 20])

    with col1:
        if st.button("Generate (only generate IEEE style image in the first step can use this)"):
            edge_temp_path = st.session_state.get('edge_temp_path', '')
            node_temp_path = st.session_state.get('node_temp_path', '')
            python_temp_path = st.session_state.get('python_temp_path', '')
            print(edge_temp_path)

            aug.main(edge_temp_path, node_temp_path, python_temp_path)
            st.success("Generation procedure is done. The result has been saved.")

        output_dir2 = Path(__file__).parent / 'Output' / 'Augmented image' / 'Intermediate output'
        finite_file_path = output_dir2 / 'finite.txt'
        detailed_file_path = output_dir2 / 'detailed_code.txt'

        # if st.button("Copy finite.txt to Clipboard"):
        #     with open(finite_file_path, 'r') as file:
        #         pyperclip.copy(file.read())
        #     st.success("text has been copied to clipboard！")

        # if st.button("Copy detailed_code.txt to Clipboard"):
        #     with open(detailed_file_path, 'r') as file:
        #         pyperclip.copy(file.read())
        #     st.success("detailed code has been copied to clipboard！")

        if st.button("Jump to original page"):
            st.session_state.page = "page1"

    with col2:
        st.subheader("Mermaid Diagram")
        mermaid_code = """graph TD;
            A-->B;
            A-->C;
            B-->D;
            C-->D;
        """
        mermaid_code = st.text_area("Mermaid Code", mermaid_code, height=250)
        st_mermaid(mermaid_code, height=1000)

def main():
    if 'page' not in st.session_state:
        st.session_state.page = "page1"
    if st.session_state.page == "page1":
        page1()
    elif st.session_state.page == "page2":
        page2()


if __name__ == "__main__":
    main()
