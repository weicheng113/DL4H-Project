from trixi.experiment.pytorchexperiment import PytorchExperiment
from eICU_preprocessing.split_train_test import create_folder
import os
import numpy as np
import torch
import time
from timeit import default_timer as timer
from datetime import timedelta
from models.metrics import print_metrics_regression, print_metrics_mortality
from enum import Enum


class Stage(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class Stats:
    def __init__(self):
        self.y_los = []
        self.y_hat_los = []
        self.y_mort = []
        self.y_hat_mort = []
        self.losses = []

    def add_los(self, y_true: np.ndarray, predictions: np.ndarray):
        assert len(y_true) == len(predictions)
        self.y_los.append(y_true)
        self.y_hat_los.append(predictions)

    def add_mort(self, y_true: np.ndarray, predictions: np.ndarray):
        self.y_mort.append(y_true)
        self.y_hat_mort.append(predictions)

    def add_loss(self, loss: float):
        self.losses.append(loss)

    def clear(self):
        self.y_los = []
        self.y_hat_los = []
        self.y_mort = []
        self.y_hat_mort = []
        self.losses = []

    def avg_loss(self):
        return sum(self.losses) / len(self.losses)

    def y_los_as_numpy(self):
        return np.concatenate(self.y_los)

    def y_mort_as_numpy(self):
        return np.concatenate(self.y_mort)

    def y_hat_los_as_numpy(self):
        return np.concatenate(self.y_hat_los)

    def y_hat_mort_as_numpy(self):
        return np.concatenate(self.y_hat_mort)


class TrixiExperimentAdapter(PytorchExperiment):
    def setup(self):
        super(TrixiExperimentAdapter, self).setup()
        self.elog.print("Config:")
        self.elog.print(self.config)
        # add a new function to elog (will save to csv, rather than as a numpy array like elog.save_numpy_data)
        self.elog.save_to_csv = self.save_to_csv
        self.bool_type = torch.BoolTensor

        self.train_epoch_start_time = None
        self.train_stats: Stats = Stats()

        self.val_stats: Stats = Stats()

        self.test_stats: Stats = Stats()

        self.elog.print(f"Experiment started at {self.current_time_as_str()}.")
        self.time_start = time.time()
        self._epoch_idx = -1

    def save_to_csv(self, data, path, header=None):
        """
            Saves a numpy array to csv in the experiment save dir

            Args:
                data: The array to be stored as a save file
                path: sub path in the save folder (or simply filename)
        """

        folder_path = create_folder(self.save_dir, os.path.dirname(path))
        file_path = folder_path + '/' + os.path.basename(path)
        if not file_path.endswith('.csv'):
            file_path += '.csv'
        np.savetxt(file_path, data, delimiter=',', header=header, comments='')
        return

    @classmethod
    def remove_padding(cls, y, mask):
        """
            Filters out padding from tensor of predictions or labels

            Args:
                y: tensor of los predictions or labels
                mask (bool_type): tensor showing which values are padding (0) and which are data (1)
        """
        # note it's fine to call .cpu() on a tensor already on the cpu
#         print(f"""
#             y.device: {y.device},
#             mask.device: {mask.device}, 
#             torch.tensor(float('nan'), device=mask.device).device: {torch.tensor(float('nan'), device=mask.device).device}
#             """)
        y = y.where(mask, torch.tensor(float('nan'), device=mask.device)).flatten().detach().cpu().numpy()
        y = y[~np.isnan(y)]
        return y

    @classmethod
    def current_time_as_str(cls):
        return cls.format_time(time.time())

    @classmethod
    def format_time(cls, t: float):
        return time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(t))

    def train_step_end(self, batch, batch_idx, loss: torch.Tensor, y_hat_los: torch.Tensor, y_hat_mort: torch.Tensor):
        self.record_step_stats(
            stats=self.train_stats,
            batch=batch,
            loss=loss,
            y_hat_los=y_hat_los,
            y_hat_mort=y_hat_mort
        )

    def record_step_stats(
            self,
            stats: Stats,
            batch,
            loss: torch.Tensor,
            y_hat_los: torch.Tensor,
            y_hat_mort: torch.Tensor):
        mort_pred_time = 24
        # unpack batch
        if self.config.dataset == 'MIMIC':
            padded, mask, flat, los_labels, mort_labels, seq_lengths = batch
            diagnoses = None
        else:
            padded, mask, diagnoses, flat, los_labels, mort_labels, seq_lengths = batch

        stats.add_loss(loss=loss.item())
        if self.config.task in ('LoS', 'multitask'):
            stats.add_los(
                y_true=self.remove_padding(los_labels, mask.bool()),
                predictions=self.remove_padding(y_hat_los.detach(), mask.bool()))

        if self.config.task in ('mortality', 'multitask') and mort_labels.shape[1] >= mort_pred_time:
            stats.add_mort(
                y_true=self.remove_padding(mort_labels[:, mort_pred_time],
                                           mask.bool()[:, mort_pred_time]),
                predictions=self.remove_padding(y_hat_mort.detach()[:, mort_pred_time],
                                                mask.bool()[:, mort_pred_time]))

    def train_epoch_start(self):
#         self.elog.print(f"train_epoch_start")
        self._epoch_idx += 1
        self.train_epoch_start_time = timer()
        self.train_stats.clear()

    def train_epoch_end(self):
#         self.elog.print(f"train_epoch_end")
        end = timer()
        self.record_epoch_stats(stats=self.train_stats, stage=Stage.TRAIN)
        self.elog.print(f"Done epoch {self._epoch_idx}, spending {timedelta(seconds=end-self.train_epoch_start_time)}.")
        self.elog.print(f"-----------------------------------")

    def record_epoch_stats(self, stats: Stats, stage: Stage):
        if stage == Stage.TRAIN:
            metric_prefix = "train"
            stage_name = "Train"
        elif stage == Stage.VAL:
            metric_prefix = "val"
            stage_name = "Validation"
        else:
            metric_prefix = "test"
            stage_name = "Test"
        if (self.epoch < 0) and (stage == Stage.VAL):
            mean_loss = stats.avg_loss()
            self.elog.print(f"Sanity Checking | {stage_name} Loss: {mean_loss:3.4f}")
            return

        if self.config.task in ('LoS', 'multitask'):
            los_metrics_list = print_metrics_regression(
                y_true=stats.y_los_as_numpy(),
                predictions=stats.y_hat_los_as_numpy(),
                elog=self.elog)  # order: mad, mse, mape, msle, r2, kappa
            for metric_name, metric in zip(['mad', 'mse', 'mape', 'msle', 'r2', 'kappa'], los_metrics_list):
                self.add_result(value=metric, name=f"{metric_prefix}_{metric_name}", counter=self.epoch)
        if self.config.task in ('mortality', 'multitask'):
            mort_metrics_list = print_metrics_mortality(
                y_true=stats.y_mort_as_numpy(),
                prediction_probs=stats.y_hat_mort_as_numpy(),
                elog=self.elog)
            for metric_name, metric in zip(['acc', 'prec0', 'prec1', 'rec0', 'rec1', 'auroc', 'auprc', 'f1macro'], mort_metrics_list):
                self.add_result(value=metric, name=f"{metric_prefix}_{metric_name}", counter=self.epoch)
        mean_loss = stats.avg_loss()
        self.elog.print(f"Epoch: {self.epoch} | {stage_name} Loss: {mean_loss:3.4f}")

    def validation_step_end(self, batch, batch_idx, loss: torch.Tensor, y_hat_los: torch.Tensor, y_hat_mort: torch.Tensor):
        self.record_step_stats(
            stats=self.val_stats,
            batch=batch,
            loss=loss,
            y_hat_los=y_hat_los,
            y_hat_mort=y_hat_mort
        )

    def validation_epoch_start(self):
#         self.elog.print(f"validation_epoch_start")
        self.val_stats.clear()

    def validation_epoch_end(self):
#         self.elog.print(f"validation_epoch_end")
        self.record_epoch_stats(stats=self.val_stats, stage=Stage.VAL)

    def test_step_end(self, batch, batch_idx, loss: torch.Tensor, y_hat_los: torch.Tensor, y_hat_mort: torch.Tensor):
        self.record_step_stats(
            stats=self.test_stats,
            batch=batch,
            loss=loss,
            y_hat_los=y_hat_los,
            y_hat_mort=y_hat_mort
        )

    def test_epoch_start(self):
        self.test_stats.clear()

    def test_epoch_end(self):
        self.record_epoch_stats(stats=self.test_stats, stage=Stage.VAL)

    def end(self):
        super(TrixiExperimentAdapter, self).end()
        time_end = time.time()
        self.elog.print(
            f"Experiment ended at {self.format_time(time_end)}, spending {timedelta(seconds=time_end-self.time_start)}."
        )
