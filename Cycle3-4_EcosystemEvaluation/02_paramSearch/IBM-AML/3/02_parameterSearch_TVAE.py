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

name = "IBM-AML"
path = "../../../data/IBM-AML/processed/"
bank_id = 3

sepMajor_synthesizer = []

X = pd.read_pickle(path + "IBM-AML_encoded.pkl")
X = X.sort_values(by=["Timestamp"])
X = X.drop(columns=["Timestamp"])

banks = sorted([name for name, _ in X.groupby("To Bank")])
bank = banks[bank_id]

X = X.loc[~((X["From Bank"] == "Unknown") | (X["To Bank"] == "Unknown"))]

train_X, test_X = train_test_split(X, shuffle=False, test_size=0.3, random_state=42)
train_X = train_X.reset_index(drop=False)
test_X = test_X.reset_index(drop=False)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_X)

# update metadata
metadata.update_column(column_name='Receiving Currency', sdtype='categorical')
metadata.update_column(column_name='Payment Currency', sdtype='categorical')
metadata.update_column(column_name='Sender', sdtype='id')
metadata.update_column(column_name='Receiver', sdtype='id')
metadata.update_column(column_name='To Branch', sdtype='id')
metadata.update_column(column_name='From Branch', sdtype='id')
metadata.update_column(column_name='To Bank', sdtype='categorical')
metadata.update_column(column_name='From Bank', sdtype='categorical')
metadata.update_column(column_name='Currency Conversion', sdtype='boolean')
metadata.update_column(column_name='Intrabank Transfer', sdtype='boolean')
metadata.update_column(column_name='target', sdtype='boolean')
if os.path.isfile(path + "IBM-AML_metadata.json"):
    os.remove(path + "IBM-AML_metadata.json")
metadata.save_to_json(path + "IBM-AML_metadata.json")

# split the datasets into banks
train_X_1 = train_X.loc[train_X["To Bank"] == bank]
train_X_2 = train_X.loc[train_X["From Bank"] == bank]
train_X = pd.concat([train_X_1, train_X_2]).drop_duplicates()
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
                                  verbose=True, use_wandb=True, cuda= "cuda:1")
    synthesizer.fit(data=train_X)
    synthetic_data = synthesizer.sample(num_rows=30000)
    diagnostic_report = run_diagnostic(real_data= train_X, synthetic_data=synthetic_data, metadata=metadata)
    quality_report = evaluate_quality(real_data= train_X, synthetic_data=synthetic_data, metadata=metadata)
    quality_score = np.mean(list(quality_report.get_properties().set_index("Property")["Score"].to_dict().values()))
    results_dict = {**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(),
                    **quality_report.get_properties().set_index("Property")["Score"].to_dict(),
                    **{"Quality Score": quality_score}}
    wandb.log(results_dict)
    wandb.finish()

wandb.agent(sweep_id, function=main, count=20)
        
        