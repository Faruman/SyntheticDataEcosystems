import os

import pandas as pd
import numpy as np

from modified_sitepackages.sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GMMSynthesizer
from modified_sitepackages.sdv.sequential import TIMEGANSynthesizer
from modified_sitepackages.sdv.metadata import SingleTableMetadata

from modified_sitepackages.sdv.evaluation.single_table import run_diagnostic, evaluate_quality

import wandb


operator = "mastercard"

#load data
real_data = pd.read_pickle('../../data/IEEE-CIS/processed/data_reduced.pkl')
real_data = real_data.drop(columns=['IdentityID', 'TransactionID'])
real_data = real_data.loc[real_data["operator"] == operator].drop(columns= ["operator"])
if real_data.shape[0] > 100000:
    real_data = real_data.sample(100000)
if real_data.shape[0] > 50000:
    num_synth_samples = 50000
else:
    num_synth_samples = real_data.shape[0]
real_data = real_data.drop(columns= ["purchaseremaildomain", "recipientemaildomain"])

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
metadata.update_column("isFraud", sdtype= "boolean")
for column in ["productcategory", "cardtype", "cardcountry", "billingcountry"]:
    metadata.update_column(column, sdtype="categorical")

## ParamSearch CTGAN
sweep_config = {
    "name": "CTGAN_{}".format(operator),
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "Quality Score"},
    "parameters": {
        "embedding_dim": {"values": [32, 64, 256]},
        "generator_dim": {"values": [(128, 128), (256, 256), (512, 512)]},
        "discriminator_dim": {"values": [(128, 128), (256, 256), (512, 512)]},
        "generator_lr": {"min": 0.00001, "max": 0.001},
        "generator_decay": {"min": 0.0, "max": 0.05},
        "discriminator_lr": {"min": 0.00001, "max": 0.001},
        "discriminator_decay": {"min": 0.0, "max": 0.05},
        "discriminator_steps": {"min": 1, "max": 15},
        "epochs": {"min": 100, "max": 1000},
        "batch_size": {"values": [5000]}
    },
}

#sweep_id = wandb.sweep(sweep=sweep_config, project="IEEE-CIS_paramSearch", entity="SyntheticFinancialDataEcosystem")
sweep_id = "syntheticFinancialDataEcosystem/IEEE-CIS_paramSearch/bjxq64dc"

def main():
    wandb.init(project="IEEE-CIS_paramSearch", entity="SyntheticFinancialDataEcosystem")
    synthesizer = CTGANSynthesizer(metadata,
                                   embedding_dim=wandb.config["embedding_dim"],
                                   generator_dim=wandb.config["generator_dim"],
                                   discriminator_dim=wandb.config["discriminator_dim"],
                                   generator_lr=wandb.config["generator_lr"],
                                   generator_decay=wandb.config["generator_decay"],
                                   discriminator_lr=wandb.config["discriminator_lr"],
                                   discriminator_decay=wandb.config["discriminator_decay"],
                                   batch_size=wandb.config["batch_size"],
                                   epochs=wandb.config["epochs"],
                                   discriminator_steps=wandb.config["discriminator_steps"],
                                   verbose=True, use_wandb=True, cuda= "cuda:0")
    synthesizer.fit(data=real_data)
    synthetic_data = synthesizer.sample(num_rows=num_synth_samples)
    diagnostic_report = run_diagnostic(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
    quality_report = evaluate_quality(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
    quality_score = np.mean(list(quality_report.get_properties().set_index("Property")["Score"].to_dict().values()))
    results_dict = {**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(),
                    **quality_report.get_properties().set_index("Property")["Score"].to_dict(),
                    **{"Quality Score": quality_score}}
    wandb.log(results_dict)
    wandb.finish()


wandb.agent(sweep_id, function=main, count=20)
