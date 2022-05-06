import lmdb
from itertools import groupby
from tqdm.auto import tqdm
from timeit import default_timer as timer
from datetime import timedelta
from eICU_preprocessing.run_all_preprocessing import eICU_path


def read_patients(db_dir, patient_ids):
    env = lmdb.open(path=db_dir)
    with env.begin(write=False) as txn:
        for patient_id in patient_ids:
            start = timer()

            value = txn.get(key=patient_id.encode())
            print(f"patient_id: {patient_id}")
            records = value.decode().splitlines()
            print(f"{len(records)} records")

            end = timer()
            print(f"read time for patient '{patient_id}': {timedelta(seconds=end-start)}.")
    env.close()



def write_db(db_dir, timeseries_file_path):
    env = lmdb.open(path=db_dir, map_size=int(2e10), map_async=True, writemap=True)  # 20GB
    with env.begin(write=True) as txn:
        with open(timeseries_file_path, 'r') as timeseries_file:
            # the first line is the feature names; we have to skip over this
            print(f"header: " + str(next(timeseries_file).strip().split(',')))
            # create a generator to capture a single patient timeseries
            # dict(key=patientid, value=records)
            ts_patients = groupby(timeseries_file, key=lambda line: line.split(',')[0])
            for patient_id, records in tqdm(iterable=ts_patients, desc="Write"):
                measures = "\n".join([m.strip("\n") for m in records])
                txn.put(patient_id.encode(), measures.encode())
    env.close()


def test():
    # db_dir = "./eICU_data/train/timeseries.db"
    # timeseries_file_path = "./eICU_data/train/timeseries.csv"
    # db_dir = "./eICU_data/val/timeseries.db"
    # timeseries_file_path = "./eICU_data/val/timeseries.csv"
    db_dir = "./eICU_data/test/timeseries.db"
    timeseries_file_path = "./eICU_data/test/timeseries.csv"
    write_db(db_dir=db_dir, timeseries_file_path=timeseries_file_path)

    patient_ids = ["2000228", "1836991", "1729377", "1585228", "3241336", "2787233", "1051213", "3157500", "2887689"]
    # read_patients(db_dir=db_dir, patient_ids=patient_ids)


def create_lmdbs():
    for subset in ["train", "val", "test"]:
        timeseries_file_path = f"{eICU_path}{subset}/timeseries.csv"
        db_dir = f"{eICU_path}{subset}/timeseries.db"
        write_db(db_dir=db_dir, timeseries_file_path=timeseries_file_path)


if __name__ == "__main__":
    # test()
    create_lmdbs()
