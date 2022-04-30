from eICU_preprocessing.split_train_test import create_folder
from models.initialise_arguments import initialise_transformer_arguments
from models.run_transformer import BaselineTransformer
from models.final_experiment_scripts.best_hyperparameters import best_transformer
from models.transformer_model import Transformer
import pytorch_lightning as pl
from eICU_preprocessing.run_all_preprocessing import eICU_path
from eICU_preprocessing.eicu_dataset import EICUDataModule
from models.experiment_module import ExperimentModule


def run_transformer():
    c = initialise_transformer_arguments()
    c['exp_name'] = 'Transformer'

    log_folder_path = create_folder('models/experiments/{}/{}'.format(c.dataset, c.task), c.exp_name)
    baseline_transformer = BaselineTransformer(config=c,
                                               n_epochs=c.n_epochs,
                                               name=c.exp_name,
                                               base_dir=log_folder_path,
                                               explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    baseline_transformer.run()


def run_best_transformer():
    c = initialise_transformer_arguments()
    c['exp_name'] = 'Transformer'
    c['dataset'] = 'eICU'
    c = best_transformer(c)

    log_folder_path = create_folder('models/experiments/final/eICU/LoS', c.exp_name)
    transformer = BaselineTransformer(config=c,
                                      n_epochs=c.n_epochs,
                                      name=c.exp_name,
                                      base_dir=log_folder_path,
                                      explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    transformer.run()


def run_pl_best_transformer():
    # torch.multiprocessing.set_start_method('spawn')

    c = initialise_transformer_arguments()
    c['exp_name'] = 'Transformer'
    c['dataset'] = 'eICU'
    c = best_transformer(c)

    data_module = EICUDataModule(
        data_path=eICU_path,
        train_batch_size=c.batch_size,
        val_batch_size=c.batch_size_test,
        test_batch_size=c.batch_size_test)
    model = Transformer(
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
        # strategy="ddp"
    )
    trainer.fit(model=model_module, datamodule=data_module)


if __name__ == '__main__':
    # Note: modified models.initialise_arguments.py gen_config(parser):
    # args = parser.parse_args() to args = parser.parse_args(args=[])
    # https://stackoverflow.com/questions/48796169/how-to-fix-ipykernel-launcher-py-error-unrecognized-arguments-in-jupyter/48798075#48798075

    # run_transformer()
    # run_best_transformer()
    run_pl_best_transformer()
