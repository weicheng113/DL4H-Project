from pytorch_lightning import LightningModule
from torch import nn
import torch
from trixi.util.config import Config
from models.trixi_experiment_adapter import TrixiExperimentAdapter
from typing import Tuple, Optional


class ExperimentModule(LightningModule):
    def __init__(self, model: nn.Module, config: Config, log_folder_path: str):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = model
        self.config = config
        self.trixi_experiment = TrixiExperimentAdapter(
            config=config,
            n_epochs=config.n_epochs,
            name=config.exp_name,
            base_dir=log_folder_path,
            explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'}
        )

    def setup(self, stage: Optional[str] = None) -> None:
        self.trixi_experiment.elog.print(self.model)
        self.trixi_experiment.setup()

    def forward(self, batch, batch_idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.dataset == 'MIMIC':
            padded, mask, flat, los_labels, mort_labels, seq_lengths = batch
            diagnoses = None
        else:
            padded, mask, diagnoses, flat, los_labels, mort_labels, seq_lengths = batch
        
#         print(f"""
#             padded.device: {padded.device}, 
#             mask.device: {mask.device}, 
#             diagnoses.device: {diagnoses.device}
#             """)
        y_hat_los, y_hat_mort = self.model(padded, diagnoses, flat)
        loss = self.model.loss(y_hat_los, y_hat_mort, los_labels, mort_labels, mask, seq_lengths,
                               self.config.sum_losses, self.config.loss)
        return loss, y_hat_los, y_hat_mort

    def training_step(self, batch, batch_idx):
        loss, y_hat_los, y_hat_mort = self.forward(batch=batch, batch_idx=batch_idx)

        self.trixi_experiment.train_step_end(batch=batch, batch_idx=batch_idx, loss=loss,
                                             y_hat_los=y_hat_los, y_hat_mort=y_hat_mort)

        return {"loss": loss}

    def training_step_end(self, training_step_outputs):
        return {'loss': training_step_outputs['loss'].sum()}

    def on_train_epoch_start(self):
        self.trixi_experiment.train_epoch_start()

    def training_epoch_end(self, outputs):
        self.trixi_experiment.train_epoch_end()

    def validation_step(self, batch, batch_idx):
        loss, y_hat_los, y_hat_mort = self.forward(batch=batch, batch_idx=batch_idx)

        self.trixi_experiment.validation_step_end(batch=batch, batch_idx=batch_idx, loss=loss,
                                                  y_hat_los=y_hat_los, y_hat_mort=y_hat_mort)
        return

    def on_validation_epoch_start(self):
        self.trixi_experiment.validation_epoch_start()

    def validation_epoch_end(self, outputs):
        self.trixi_experiment.validation_epoch_end()

    def test_step(self, batch, batch_idx):
        loss, y_hat_los, y_hat_mort = self.forward(batch=batch, batch_idx=batch_idx)

        self.trixi_experiment.test_step_end(batch=batch, batch_idx=batch_idx, loss=loss,
                                            y_hat_los=y_hat_los, y_hat_mort=y_hat_mort)
        return

    def on_test_epoch_start(self):
        self.trixi_experiment.test_epoch_start()

    def test_epoch_end(self, outputs):
        self.trixi_experiment.test_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.L2_regularisation)
        return optimizer

    def teardown(self, stage: Optional[str] = None) -> None:
        self.trixi_experiment.end()
