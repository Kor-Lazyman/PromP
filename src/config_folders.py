
import os
import shutil
from config_SimPy import *
from config_RL import *
from multiprocessing import Process, current_process
def DEFINE_FOLDER(folder_name):
    if os.path.exists(folder_name):
        file_list = os.listdir(folder_name)
        folder_name = os.path.join(folder_name, f"Train_{len(file_list)+1}")
        os.makedirs(folder_name)
    else:
        folder_name = os.path.join(folder_name, "Train_1")
        os.makedirs(folder_name)
    return folder_name


def save_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    # Create a new folder
    os.makedirs(path)
    return path

if "MainProcess" == current_process().name:
    # Define parent dir's path
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)

    if EXPERIMENT_MAML:
        PAR_FOLDER = os.path.join(
            parent_dir, "Tensorboard_logs_Experiment_MAML")
    elif EXPERIMENT_ADAPTATION:
        PAR_FOLDER = os.path.join(
            parent_dir, "Tensorboard_logs_Experiment_ADAPT")
    elif DRL_TENSORBOARD:
        PAR_FOLDER = os.path.join(
            parent_dir, "Tensorboard_logs_Experiment_DRL")
    else:
        PAR_FOLDER = os.path.join(
            parent_dir, "Tensorboard_logs_MAML")

    result_csv_folder = os.path.join(parent_dir, "result_CSV")
    STATE_folder = os.path.join(result_csv_folder, "state")
    daily_report_folder = os.path.join(result_csv_folder, "daily_report")

    # Define dir's path
    TENSORFLOW_LOGS = DEFINE_FOLDER(PAR_FOLDER)

    STATE = save_path(STATE_folder)
    REPORT_LOGS = save_path(daily_report_folder)

    graph_log = os.path.join(parent_dir, "result_graph_folder")
    GRAPH_LOG = DEFINE_FOLDER(graph_log)
    # Saved Model
    SAVED_MODEL_PATH = os.path.join(parent_dir, "Saved_Model")
    SAVE_MODEL = False
    SAVED_MODEL_NAME = "PPO_AP3_100"
    # SAVED_MODEL_NAME = "MAML_PPO_AP3_E5_O1000"

    # Load Model
    LOAD_MODEL = False
    LOAD_MODEL_NAME = "MAML_PPO_AP3_E5_O1000"
