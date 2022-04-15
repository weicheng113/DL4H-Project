import torch
import pandas as pd
from itertools import groupby
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from timeit import default_timer as timer
from datetime import timedelta

# bit hacky but passes checks and I don't have time to implement a neater solution
lab_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 15, 16, 18, 21, 22, 23, 24, 29, 32, 33, 34, 39, 40, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 62, 63, 67, 68, 69, 70, 71, 72, 75, 83, 84, 86]
labs_to_keep = [0] + [(i + 1) for i in lab_indices] + [(i + 88) for i in lab_indices] + [-1]
no_lab_indices = list(range(87))
no_lab_indices = [x for x in no_lab_indices if x not in lab_indices]
no_labs_to_keep = [0] + [(i + 1) for i in no_lab_indices] + [(i + 88) for i in no_lab_indices] + [-1]


class EICUDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_path, device=None, labs_only=False, no_labs=False):
        self._diagnoses_path = data_path + '/diagnoses.csv'
        self._labels_path = data_path + '/labels.csv'
        self._flat_path = data_path + '/flat.csv'
        self._timeseries_path = data_path + '/timeseries.csv'
        self._device = device
        self.labs_only = labs_only
        self.no_labs = no_labs
        self._dtype = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor

        self.labels = pd.read_csv(self._labels_path, index_col='patient')
        self.flat = pd.read_csv(self._flat_path, index_col='patient')
        self.diagnoses = pd.read_csv(self._diagnoses_path, index_col='patient')

        # we minus 2 to calculate F because hour and time are not features for convolution
        self.F = (pd.read_csv(self._timeseries_path, index_col='patient', nrows=1).shape[1] - 2)//2
        self.D = self.diagnoses.shape[1]
        self.no_flat_features = self.flat.shape[1]

        self.patients = list(self.labels.index)
        self.no_patients = len(self.patients)

    def line_split(self, line):
        return [float(x) for x in line.split(',')]

    def get_los_labels(self, labels: torch.Tensor, offset_times: torch.Tensor):
        # labels(batch_size) is actualICULOS in days, which is the total actual ICU length of stay for a patient.
        # times(batch_size, # time series of a patient). Times store offset time from admit time.
        # mask(batch_size, # time series of a patient)

        # We use actualICULOS - offset time from admit time = the length of stay left from that measurement,
        # which is the value we would like to predict given the measurement at the point of time.
        times = labels.unsqueeze(dim=-1).repeat(offset_times.shape[0]) - offset_times
        # clamp any labels that are less than 30 mins otherwise it becomes too small when the log is taken
        # make sure where there is no data the label is 0
        return times.clamp(min=1/48)

    def get_mort_labels(self, labels, length):
        repeated_labels = labels.unsqueeze(-1).repeat(length)
        return repeated_labels

    def __iter__(self):
        # note that once the generator is finished, the file will be closed automatically
        with open(self._timeseries_path, 'r') as timeseries_file:
            # the first line is the feature names; we have to skip over this
            self.timeseries_header = next(timeseries_file).strip().split(',')
            # create a generator to capture a single patient timeseries
            ts_patient = groupby(map(self.line_split, timeseries_file), key=lambda line: line[0])  #  dict(key=patientid, value=record)
            # we loop through these batches, tracking the index because we need it to index the pandas dataframes
            # for i in range(self.no_patients):
            for i in range(64):
                ts = torch.tensor([measure[1:] for measure in next(ts_patient)[1]]).type(self._dtype)  # ts_batch(batch_size, # time series of a patient, a time series features)
                ts[:, 0] /= 24  # scale the time into days instead of hours
                los_labels = self.get_los_labels(torch.tensor(self.labels.iloc[i, 7]).type(self._dtype), ts[:, 0])
                mort_labels = self.get_mort_labels(torch.tensor(self.labels.iloc[i, 5]).type(self._dtype), length=los_labels.shape[0])
                # padded(batch_size, a time series features, # time series of a patient)
                # we must avoid taking data before time_before_pred hours to avoid diagnoses and apache variable from the future
                yield {
                    "time_series": ts,
                    "diagnoses": torch.tensor(self.diagnoses.iloc[i].values).type(self._dtype),  # B * D; diagnoses(batch_size, diagnoses feature dim)
                    "flat": torch.tensor(self.flat.iloc[i].values.astype(float)).type(self._dtype),  # B * no_flat_features; flat(batch_size, patient feature dim)
                    "los_labels": los_labels,
                    "mort_labels": mort_labels
                }

    def collate(self, batch):
        time_before_pred = 5

        batch_padded, ts_lengths = self.pad_batch(batch)
        batch_collated = torch.utils.data.dataloader.default_collate(batch_padded)
        my_batch = {
            "time_series": batch_collated["time_series"].permute(0, 2, 1),
            "diagnoses": batch_collated["diagnoses"],
            "flat": batch_collated["flat"],
            "los_labels": batch_collated["los_labels"][:, time_before_pred:],
            "mort_labels": batch_collated["mort_labels"][:, time_before_pred:],
            "mask": batch_collated["mask"][:, time_before_pred:],
            "ts_lengths": torch.tensor(ts_lengths).type(self._dtype) - time_before_pred
        }
        return (
            my_batch["time_series"].to(self._device),
            my_batch["mask"].to(self._device),
            my_batch["diagnoses"].to(self._device),
            my_batch["flat"].to(self._device),
            my_batch["los_labels"].to(self._device),
            my_batch["mort_labels"].to(self._device),
            my_batch["ts_lengths"].to(self._device)
        )

    def pad_batch(self, batch):
        ts_lengths = [i["time_series"].shape[0] for i in batch]
        max_length = max(ts_lengths)
        updated_batch = []
        for i in batch:
            n_measures, measure_size = i["time_series"].shape
            n_paddings = max_length - n_measures
            ts = torch.concat([i["time_series"], torch.tensor([[0] * measure_size] * n_paddings).type(self._dtype)])

            mask = torch.zeros(max_length).type(self._dtype)
            mask[:n_measures] = 1

            los_labels = torch.concat([i["los_labels"], torch.tensor([0] * n_paddings).type(self._dtype)])
            mort_labels = torch.concat([i["mort_labels"], torch.tensor([0] * n_paddings).type(self._dtype)])

            updated_batch.append({
                "time_series": ts,
                "diagnoses": i["diagnoses"],
                "flat": i["flat"],
                "los_labels": los_labels,
                "mort_labels": mort_labels,
                "mask": mask
            })
        return updated_batch, ts_lengths

    # def pad_sequences(self, ts_batch):
    #     seq_lengths = [len(x) for x in ts_batch]
    #     max_len = max(seq_lengths)
    #     padded = [patient + [[0] * (self.F * 2 + 2)] * (max_len - len(patient)) for patient in ts_batch]
    #     if self.labs_only:
    #         padded = np.array(padded)
    #         padded = padded[:, :, labs_to_keep]
    #     if self.no_labs:
    #         padded = np.array(padded)
    #         padded = padded[:, :, no_labs_to_keep]
    #     padded = torch.tensor(padded, device=self._device).type(self._dtype).permute(0, 2, 1)  # B * (2F + 2) * T
    #     padded[:, 0, :] /= 24  # scale the time into days instead of hours
    #     mask = torch.zeros(padded[:, 0, :].shape, device=self._device).type(self._dtype)
    #     for p, l in enumerate(seq_lengths):
    #         mask[p, :l] = 1
    #     return padded, mask, torch.tensor(seq_lengths).type(self._dtype)


class EICUReaderAdapter:
    def __init__(self, data_path, batch_size: int, device=None, labs_only=False, no_labs=False):
        dataset = EICUDataset(data_path=data_path, device=device, labs_only=labs_only, no_labs=no_labs)
        self.data_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate, num_workers=4)
        self.F = dataset.F
        self.D = dataset.D
        self.no_flat_features = dataset.no_flat_features
        self.patients = dataset.patients

    def __iter__(self):
        for batch in self.data_loader:
            yield batch


def test():
    dataset = EICUDataset('../eicu-data/train', device=torch.device('cpu'))
    # it_dataset = iter(dataset)
    # print(next(it_dataset))

    data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False, collate_fn=dataset.collate, num_workers=4)
    start = timer()
    for batch in tqdm(data_loader, "loading data"):
        pass
    end = timer()
    print(timedelta(seconds=end-start))


if __name__ == '__main__':
    test()
