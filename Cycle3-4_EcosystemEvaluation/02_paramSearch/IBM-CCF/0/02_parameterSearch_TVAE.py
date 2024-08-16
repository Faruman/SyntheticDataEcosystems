import numpy as np
import pandas as pd
import os
from pathlib import Path

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from modified_sitepackages.sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GMMSynthesizer
from modified_sitepackages.sdv.sequential import TIMEGANSynthesizer
from modified_sitepackages.sdv.metadata import SingleTableMetadata

from modified_sitepackages.sdv.evaluation.single_table import run_diagnostic, evaluate_quality

import wandb

if os.path.exists("./working"):
    os.makedirs("./working")

name = "IBM-CCF"
path = "../../../data/IBM-CCF/processed/"
bank_id = 0

sepMajor_synthesizer = []

X = pd.read_pickle(path + "IBM-CCF_encoded.pkl")
X = X.sort_values(by=["Timestamp"])
X = X.drop(columns=["Timestamp"])

banks = sorted([name for name, _ in X.groupby("Bank")])
bank = banks[bank_id]

train_X, test_X = train_test_split(X, shuffle=False, test_size=0.3, random_state=42)
train_X = train_X.reset_index(drop=False)
test_X = test_X.reset_index(drop=False)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_X)

# update metadata
metadata.update_column(column_name='User', sdtype='id')
metadata.update_column(column_name='Card', sdtype='numerical')
metadata.update_column(column_name='Use Chip', sdtype='categorical')
metadata.update_column(column_name='Merchant Name', sdtype='id')
metadata.update_column(column_name='Merchant State', sdtype='categorical')
metadata.update_column(column_name='Zip', sdtype='postcode')
metadata.update_column(column_name='MCC', sdtype='categorical')
metadata.update_column(column_name='Errors?', sdtype='categorical')
metadata.update_column(column_name='Card ID', sdtype='id')
metadata.update_column(column_name='in Home State?', sdtype='boolean')
metadata.update_column(column_name='target', sdtype='boolean')
if os.path.isfile(path + "IBM-CCF_metadata.json"):
    os.remove(path + "IBM-CCF_metadata.json")
metadata.save_to_json(path + "IBM-CCF_metadata.json")

# split the datasets into banks
train_X = train_X.loc[train_X["Bank"] == bank]
test_X = test_X.loc[test_X["Bank"] == bank]
if train_X.shape[0] > 100000:
    train_X = train_X.sample(100000, random_state=42)

sweep_config = {
    "name": "TVAE_{}".format(bank),
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "Quality Score"},
    "parameters": {
        "embedding_dim": {"values": [32, 64, 256]},
        "compress_dims": {"values": [(128, 128), (256, 256), (512, 512)]},
        "decompress_dims": {"values": [(128, 128), (256, 256), (512, 512)]},
        "l2scale": {"min": 0.00001, "max": 0.001},
        "loss_factor": {"min": 1, "max": 5},
        "learning_rate": {"min": 0.00001, "max": 0.001},
        "epochs": {"min": 100, "max": 1000},
        "batch_size": {"values": [5000]}
    },
}

sweep_id = wandb.sweep(sweep=sweep_config, project="{}_paramSearch".format(name), entity="SyntheticFinancialDataEcosystem")

def main():
    wandb.init(project="{}_paramSearch".format(name), entity="SyntheticFinancialDataEcosystem")
    synthesizer = TVAESynthesizer(metadata,
                                  embedding_dim=wandb.config["embedding_dim"],
                                  compress_dims=wandb.config["compress_dims"],
                                  decompress_dims=wandb.config["decompress_dims"],
                                  l2scale=wandb.config["l2scale"], loss_factor=wandb.config["loss_factor"],
                                  learning_rate=wandb.config["learning_rate"],
                                  epochs=wandb.config["epochs"], batch_size=wandb.config["batch_size"],
                                  verbose=True, use_wandb=True, cuda= "cuda:0")
    synthesizer.fit(data=train_X)
    synthetic_data = synthesizer.sample(num_rows=30000)
    diagnostic_report = run_diagnostic(real_data= train_X, synthetic_data=synthetic_data, metadata=metadata)
    quality_report = evaluate_quality(real_data= train_X, synthetic_data=synthetic_data, metadata=metadata)
    quality_score = np.mean(list(quality_report.get_properties().set_index("Property")["Score"].to_dict().values()))
    results_dict = {**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(), **quality_report.get_properties().set_index("Property")["Score"].to_dict(), **{"Quality Score": quality_score}}
    wandb.log(results_dict)
    wandb.finish()

wandb.agent(sweep_id, function=main, count=20)
        
        