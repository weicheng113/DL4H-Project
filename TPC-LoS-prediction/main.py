import joblib
from models.initialise_arguments import initialise_tpc_arguments
from models.run_tpc import TPC
from eICU_preprocessing.split_train_test import create_folder
from models.initialise_arguments import initialise_transformer_arguments
from models.run_transformer import BaselineTransformer
from models.initialise_arguments import initialise_lstm_arguments
from models.run_lstm import BaselineLSTM


def joblib_version():
    print(joblib.__version__)


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


if __name__ == '__main__':
    # Note: modified models.initialise_arguments.py gen_config(parser):
    # args = parser.parse_args() to args = parser.parse_args(args=[])
    # https://stackoverflow.com/questions/48796169/how-to-fix-ipykernel-launcher-py-error-unrecognized-arguments-in-jupyter/48798075#48798075

    # joblib_version()
    # run_tpc()
    # run_transformer()
    run_lstm()

