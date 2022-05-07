from eICU_preprocessing.split_train_test import create_folder
from models.initialise_arguments import initialise_lstm_arguments
from models.run_lstm import BaselineLSTM
from models.final_experiment_scripts.best_hyperparameters import best_cw_lstm
import torch
import os


def get_log_folder_path(experiment_name: str):
    log_folder_path = create_folder(
        # parent_path='./experiment_results/train',
        parent_path='models/experiments/final/eICU/LoS',
        folder=experiment_name)
    return log_folder_path


def get_last_experiment(log_folder_path: str):
    last_experiment_folder = sorted([d for d in os.listdir(log_folder_path) if '20' in str(d)], reverse=True)[0]
    last_experiment = f'{log_folder_path}/' + last_experiment_folder
    return last_experiment


def reload_best_cw_lstm_and_test():
    torch.multiprocessing.set_start_method('spawn')

    c = initialise_lstm_arguments()
    c['exp_name'] = 'ChannelwiseLSTM'
    c['dataset'] = 'eICU'
    c = best_cw_lstm(c)

    log_folder_path = get_log_folder_path(experiment_name=c.exp_name)
    last_experiment_folder = get_last_experiment(log_folder_path=log_folder_path)
    baseline_lstm = BaselineLSTM(
        config=c,
        n_epochs=c.n_epochs,
        name=c.exp_name,
        base_dir=log_folder_path,
        explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'},
        resume=last_experiment_folder)
    baseline_lstm.run_test()


if __name__ == '__main__':
    # Note: modified models.initialise_arguments.py gen_config(parser):
    # args = parser.parse_args() to args = parser.parse_args(args=[])
    # https://stackoverflow.com/questions/48796169/how-to-fix-ipykernel-launcher-py-error-unrecognized-arguments-in-jupyter/48798075#48798075

    # run_lstm()
    reload_best_cw_lstm_and_test()
