from eICU_preprocessing.split_train_test import create_folder
from models.initialise_arguments import initialise_lstm_arguments
from models.run_lstm import BaselineLSTM
from models.final_experiment_scripts.best_hyperparameters import best_lstm


def run_lstm():
    c = initialise_lstm_arguments()
    c['exp_name'] = 'LSTM'

    log_folder_path = create_folder('models/experiments/{}/{}'.format(c.dataset, c.task), c.exp_name)
    baseline_lstm = BaselineLSTM(config=c,
                                 n_epochs=c.n_epochs,
                                 name=c.exp_name,
                                 base_dir=log_folder_path,
                                 explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    baseline_lstm.run()


def run_best_lstm():
    c = initialise_lstm_arguments()
    c['exp_name'] = 'StandardLSTM'
    c['dataset'] = 'eICU'
    c = best_lstm(c)

    log_folder_path = create_folder('models/experiments/final/eICU/LoS', c.exp_name)
    baseline_lstm = BaselineLSTM(config=c,
                                 n_epochs=c.n_epochs,
                                 name=c.exp_name,
                                 base_dir=log_folder_path,
                                 explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    baseline_lstm.run()


if __name__ == '__main__':
    # Note: modified models.initialise_arguments.py gen_config(parser):
    # args = parser.parse_args() to args = parser.parse_args(args=[])
    # https://stackoverflow.com/questions/48796169/how-to-fix-ipykernel-launcher-py-error-unrecognized-arguments-in-jupyter/48798075#48798075

    # run_lstm()
    run_best_lstm()
