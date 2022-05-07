from models.initialise_arguments import initialise_tpc_arguments
from models.run_tpc import TPC
from eICU_preprocessing.split_train_test import create_folder
from models.final_experiment_scripts.best_hyperparameters import best_tpc
import torch
import argparse
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


def reload_best_tpc_and_test():
    torch.multiprocessing.set_start_method('spawn')

    c = initialise_tpc_arguments()
    c['exp_name'] = 'TPC'
    c['dataset'] = 'eICU'
    c = best_tpc(c)
    c['batch_size_test'] = 24

    log_folder_path = get_log_folder_path(experiment_name=c.exp_name)
    last_experiment_folder = get_last_experiment(log_folder_path=log_folder_path)
    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'},
              resume=last_experiment_folder)
    tpc.run_test()


def reload_best_pointwise_only_and_test():
    torch.multiprocessing.set_start_method('spawn')

    c = initialise_tpc_arguments()
    c['exp_name'] = 'PointwiseOnly'
    c['dataset'] = 'eICU'
    c = best_tpc(c)
    c["model_type"] = "pointwise_only"
    c['batch_size'] = 512
    c['batch_size_test'] = 512

    log_folder_path = get_log_folder_path(folder=c.exp_name)
    last_experiment_folder = get_last_experiment(log_folder_path=log_folder_path)
    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'},
              resume=last_experiment_folder)
    tpc.run_test()


def reload_best_temp_only_and_test():
    torch.multiprocessing.set_start_method('spawn')

    c = initialise_tpc_arguments()
    c['exp_name'] = 'TempOnly'
    c['dataset'] = 'eICU'
    c = best_tpc(c)
    c['model_type'] = 'temp_only'

    log_folder_path = get_log_folder_path(folder=c.exp_name)
    last_experiment_folder = get_last_experiment(log_folder_path=log_folder_path)
    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'},
              resume=last_experiment_folder)
    tpc.run_test()


def reload_best_tpc_no_skip_and_test():
    torch.multiprocessing.set_start_method('spawn')

    c = initialise_tpc_arguments()
    c['exp_name'] = 'TPCNoSkip'
    c['dataset'] = 'eICU'
    c = best_tpc(c)
    c['no_skip_connections'] = True

    log_folder_path = get_log_folder_path(folder=c.exp_name)
    last_experiment_folder = get_last_experiment(log_folder_path=log_folder_path)
    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'},
              resume=last_experiment_folder)
    tpc.run_test()


def reload_best_tpc_multitask_and_test():
    torch.multiprocessing.set_start_method('spawn')

    c = initialise_tpc_arguments()
    c['exp_name'] = 'TPC'
    c['dataset'] = 'eICU'
    c = best_tpc(c)
    c["task"] = "multitask"
    c['batch_size'] = 24
    c["batch_size_test"] = 24

    log_folder_path = get_log_folder_path(folder=c.exp_name)
    last_experiment_folder = get_last_experiment(log_folder_path=log_folder_path)
    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'},
              resume=last_experiment_folder)
    tpc.run_test()


def reload_best_tpc_mse_and_test():
    torch.multiprocessing.set_start_method('spawn')

    c = initialise_tpc_arguments()
    c['exp_name'] = 'TPC'
    c['dataset'] = 'eICU'
    c = best_tpc(c)
    c["loss"] = "mse"
    c['batch_size'] = 24
    c["batch_size_test"] = 24

    log_folder_path = get_log_folder_path(folder=c.exp_name)
    last_experiment_folder = get_last_experiment(log_folder_path=log_folder_path)
    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'},
              resume=last_experiment_folder)
    tpc.run_test()


def reload_best_tpc_mask_skip_and_test():
    torch.multiprocessing.set_start_method('spawn')

    c = initialise_tpc_arguments()
    c['exp_name'] = 'TPCMaskSkip'
    c['dataset'] = 'eICU'
    c = best_tpc(c)
    c['batch_size_test'] = 24

    log_folder_path = get_log_folder_path(experiment_name=c.exp_name)
    last_experiment_folder = get_last_experiment(log_folder_path=log_folder_path)
    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'},
              resume=last_experiment_folder)
    tpc.run_test()


def run():
    parser = argparse.ArgumentParser()
    models = ["tpc", "tpc-multitask", "tpc-mse", "pointwise-only", "temp-only", "tpc-no-skip", "tpc-mask-skip"]
    # parser.add_argument('--model', default='tpc', type=str, choices=models)
    parser.add_argument('--model', type=str, required=True, choices=models)
    args = parser.parse_args()
    function_mappings = {
        "tpc": reload_best_tpc_and_test,
        "tpc-multitask": reload_best_tpc_multitask_and_test,
        "tpc-mse": reload_best_tpc_mse_and_test,
        "pointwise-only": reload_best_pointwise_only_and_test,
        "temp-only": reload_best_temp_only_and_test,
        "tpc-no-skip": reload_best_tpc_no_skip_and_test,
        "tpc-mask-skip": reload_best_tpc_mask_skip_and_test
    }
    function_mappings[args.model]()


def run_manually():
    reload_best_tpc_and_test()
    # reload_best_tpc_multitask_and_test()
    # reload_best_tpc_mse_and_test()
    # reload_best_pointwise_only_and_test()
    # reload_best_temp_only_and_test()
    # reload_best_tpc_no_skip_and_test()


if __name__ == '__main__':
    # Note: modified models.initialise_arguments.py gen_config(parser):
    # args = parser.parse_args() to args = parser.parse_args(args=[])
    # https://stackoverflow.com/questions/48796169/how-to-fix-ipykernel-launcher-py-error-unrecognized-arguments-in-jupyter/48798075#48798075
    run()
