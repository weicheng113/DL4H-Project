from eICU_preprocessing.eicu_dataset import EICUDataModule
from models.experiment_module import ExperimentModule
from models.initialise_arguments import initialise_tpc_arguments
from models.run_tpc import TPC
from models.run_tpc2 import TPC2
from eICU_preprocessing.split_train_test import create_folder
from models.final_experiment_scripts.best_hyperparameters import best_tpc
import torch
from eICU_preprocessing.run_all_preprocessing import eICU_path
from models.tpc_model import TempPointConv
import pytorch_lightning as pl
import argparse


def run_tpc():
    c = initialise_tpc_arguments()
    c['exp_name'] = 'TPC'
    c['model_type'] = 'tpc'
    # c['n_layers'] = 4
    # c['kernel_size'] = 3
    # c['no_temp_kernels'] = 10
    # c['point_size'] = 10
    # c['last_linear_size'] = 20
    # c['diagnosis_size'] = 20
    # c['batch_size'] = 32
    # c['learning_rate'] = 0.001
    # c['main_dropout_rate'] = 0.3
    # c['temp_dropout_rate'] = 0.1

    log_folder_path = create_folder('models/experiments/{}/{}'.format(c.dataset, c.task), c.exp_name)

    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    tpc.run()


def get_log_folder_path(experiment_name: str):
    log_folder_path = create_folder(
        # parent_path='./experiment_results/train',
        parent_path='models/experiments/final/eICU/LoS',
        folder=experiment_name)
    return log_folder_path


def run_best_tpc():
    torch.multiprocessing.set_start_method('spawn')

    c = initialise_tpc_arguments()
    c['exp_name'] = 'TPC'
    c['dataset'] = 'eICU'
    # c['n_layers'] = 4
    # c['kernel_size'] = 3
    # c['no_temp_kernels'] = 10
    # c['point_size'] = 10
    # c['last_linear_size'] = 20
    # c['diagnosis_size'] = 20
    # c['batch_size'] = 32
    # c['learning_rate'] = 0.001
    # c['main_dropout_rate'] = 0.3
    # c['temp_dropout_rate'] = 0.1
    c = best_tpc(c)
    c['mode'] = 'train'
    # c['n_epochs'] = 2
    c["shuffle_train"] = True
    c['batch_size'] = 24
    c['batch_size_test'] = 24

    log_folder_path = get_log_folder_path(experiment_name=c.exp_name)
    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    tpc.run()


def run_pl_best_tpc():
    # torch.multiprocessing.set_start_method('spawn')

    c = initialise_tpc_arguments()
    c['exp_name'] = 'TPC'
    c['dataset'] = 'eICU'
    c = best_tpc(c)
    c['n_epochs'] = 2
    # c["shuffle_train"] = True

    data_module = EICUDataModule(
        data_path=eICU_path,
        train_batch_size=c.batch_size,
        val_batch_size=c.batch_size_test,
        test_batch_size=c.batch_size_test)
    model = TempPointConv(
        config=c,
        F=data_module.F,
        D=data_module.D,
        no_flat_features=data_module.no_flat_features)
    log_folder_path = create_folder('models/experiments/final/eICU/LoS', c.exp_name)
    model_module = ExperimentModule(
        model=model, config=c, log_folder_path=log_folder_path)

    pl.seed_everything(seed=42)
    # trainer = pl.Trainer(fast_dev_run=True, max_epochs=2)
    trainer = pl.Trainer(
        max_epochs=c.n_epochs,
        # progress_bar_refresh_rate=20,
        accelerator="gpu",
        devices=1,
        # profiler="advanced",
        # profiler="simple",
        # strategy="ddp"
    )
    trainer.fit(model=model_module, datamodule=data_module)


def run_best_pointwise_only():
    torch.multiprocessing.set_start_method('spawn')

    c = initialise_tpc_arguments()
    c['exp_name'] = 'PointwiseOnly'
    c['dataset'] = 'eICU'
    c = best_tpc(c)
    c['mode'] = 'train'
    c["shuffle_train"] = True
    c["model_type"] = "pointwise_only"
    c['batch_size'] = 512
    c['batch_size_test'] = 512

    log_folder_path = get_log_folder_path(experiment_name=c.exp_name)
    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    tpc.run()


def run_best_temp_only():
    torch.multiprocessing.set_start_method('spawn')

    c = initialise_tpc_arguments()
    c['exp_name'] = 'TempOnly'
    c['dataset'] = 'eICU'
    c = best_tpc(c)
    c['mode'] = 'train'
    c["shuffle_train"] = True
    c['model_type'] = 'temp_only'

    log_folder_path = get_log_folder_path(experiment_name=c.exp_name)
    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    tpc.run()


def run_best_tpc_no_skip():
    torch.multiprocessing.set_start_method('spawn')

    c = initialise_tpc_arguments()
    c['exp_name'] = 'TPCNoSkip'
    c['dataset'] = 'eICU'
    c = best_tpc(c)
    c['mode'] = 'train'
    c["shuffle_train"] = True
    c['no_skip_connections'] = True

    log_folder_path = get_log_folder_path(experiment_name=c.exp_name)
    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    tpc.run()


def run_best_tpc_multitask():
    torch.multiprocessing.set_start_method('spawn')

    c = initialise_tpc_arguments()
    c['exp_name'] = 'TPC'
    c['dataset'] = 'eICU'
    c = best_tpc(c)
    c['mode'] = 'train'
    # c['n_epochs'] = 2
    c["shuffle_train"] = True
    c["task"] = "multitask"
    c['batch_size'] = 24
    c["batch_size_test"] = 24

    log_folder_path = get_log_folder_path(experiment_name=c.exp_name)
    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    tpc.run()


def run_best_tpc_mse():
    torch.multiprocessing.set_start_method('spawn')

    c = initialise_tpc_arguments()
    c['exp_name'] = 'TPCmse'
    c['dataset'] = 'eICU'
    c = best_tpc(c)
    c['mode'] = 'train'
    # c['n_epochs'] = 2
    c["shuffle_train"] = True
    c["loss"] = "mse"
    c['batch_size'] = 24
    c["batch_size_test"] = 24

    log_folder_path = get_log_folder_path(experiment_name=c.exp_name)
    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    tpc.run()


def run_best_tpc_mask_skip():
    torch.multiprocessing.set_start_method('spawn')

    c = initialise_tpc_arguments()
    c['exp_name'] = 'TPCMaskSkip'
    c['dataset'] = 'eICU'
    c = best_tpc(c)
    c['mode'] = 'train'
    c["shuffle_train"] = True
    c['batch_size'] = 24
    c['batch_size_test'] = 24

    log_folder_path = get_log_folder_path(experiment_name=c.exp_name)
    tpc = TPC2(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    tpc.run()


def run():
    parser = argparse.ArgumentParser()
    models = ["tpc", "tpc-multitask", "tpc-mse", "pointwise-only", "temp-only", "tpc-no-skip", "tpc-mask-skip"]
    # parser.add_argument('--model', default='tpc', type=str, choices=models)
    parser.add_argument('--model', type=str, required=True, choices=models)
    args = parser.parse_args()
    function_mappings = {
        "tpc": run_best_tpc,
        "tpc-multitask": run_best_tpc_multitask,
        "tpc-mse": run_best_tpc_mse,
        "pointwise-only": run_best_pointwise_only,
        "temp-only": run_best_temp_only,
        "tpc-no-skip": run_best_tpc_no_skip,
        "tpc-mask-skip": run_best_tpc_mask_skip
    }

    function_mappings[args.model]()


def run_manually():
    # run_tpc()
    # python -W ignore:semaphore_tracker:UserWarning train_tpc.py
    # run_best_tpc()
    # run_best_tpc_multitask()
    # run_best_tpc_mse()
    # run_pl_best_tpc()
    # run_best_pointwise_only()
    # run_best_temp_only()
    # run_best_tpc_no_skip()
    run_best_tpc_mask_skip()


if __name__ == '__main__':
    # Note: modified models.initialise_arguments.py gen_config(parser):
    # args = parser.parse_args() to args = parser.parse_args(args=[])
    # https://stackoverflow.com/questions/48796169/how-to-fix-ipykernel-launcher-py-error-unrecognized-arguments-in-jupyter/48798075#48798075
    run()
    # run_manually()



