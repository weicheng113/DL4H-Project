import joblib
from eICU_preprocessing.split_train_test import create_folder
from models.initialise_arguments import initialise_tpc_arguments
from models.run_tpc import TPC


def joblib_version():
    print(joblib.__version__)


def run_tpc():
    c = initialise_tpc_arguments()
    c['exp_name'] = 'TPC'
    c['model_type'] = 'tpc'
    c['n_layers'] = 4
    c['kernel_size'] = 3
    c['no_temp_kernels'] = 10
    c['point_size'] = 10
    c['last_linear_size'] = 20
    c['diagnosis_size'] = 20
    c['batch_size'] = 32
    c['learning_rate'] = 0.001
    c['main_dropout_rate'] = 0.3
    c['temp_dropout_rate'] = 0.1

    log_folder_path = create_folder('models/experiments/{}/{}'.format(c.dataset, c.task), c.exp_name)

    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    tpc.run()


if __name__ == '__main__':
    # joblib_version()
    run_tpc()
