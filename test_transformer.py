from eICU_preprocessing.split_train_test import create_folder
from models.initialise_arguments import initialise_transformer_arguments
from models.run_transformer import BaselineTransformer
from models.final_experiment_scripts.best_hyperparameters import best_transformer
import torch
import os


def get_last_experiment(log_folder_path: str):
    last_experiment_folder = sorted([d for d in os.listdir(log_folder_path) if '20' in str(d)], reverse=True)[0]
    last_experiment = f'{log_folder_path}/' + last_experiment_folder
    return last_experiment


def get_log_folder_path(experiment_name: str):
    log_folder_path = create_folder(
        # parent_path='./experiment_results/train',
        parent_path='models/experiments/final/eICU/LoS',
        folder=experiment_name)
    return log_folder_path


def reload_best_transformer_and_test():
    torch.multiprocessing.set_start_method('spawn')

    c = initialise_transformer_arguments()
    c['exp_name'] = 'Transformer'
    c['dataset'] = 'eICU'
    c = best_transformer(c)

    log_folder_path = get_log_folder_path(experiment_name=c.exp_name)
    last_experiment_folder = get_last_experiment(log_folder_path=log_folder_path)
    transformer = BaselineTransformer(
        config=c,
        n_epochs=c.n_epochs,
        name=c.exp_name,
        base_dir=log_folder_path,
        explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'},
        resume=last_experiment_folder)
    transformer.run_test()


if __name__ == '__main__':
    # Note: modified models.initialise_arguments.py gen_config(parser):
    # args = parser.parse_args() to args = parser.parse_args(args=[])
    # https://stackoverflow.com/questions/48796169/how-to-fix-ipykernel-launcher-py-error-unrecognized-arguments-in-jupyter/48798075#48798075

    # run_transformer()
    reload_best_transformer_and_test()
