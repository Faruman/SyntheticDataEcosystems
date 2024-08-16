import os

from tqdm import tqdm

import pandas as pd
tqdm.pandas()
import numpy as np

from modified_sitepackages.sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GMMSynthesizer
from modified_sitepackages.sdv.sequential import TIMEGANSynthesizer
from modified_sitepackages.sdv.metadata import SingleTableMetadata

from modified_sitepackages.sdv.evaluation.single_table import run_diagnostic, evaluate_quality

import wandb

operator = "american express"


context_columns = []

#load data
real_data = pd.read_pickle('../../data/IEEE-CIS/processed/data_reduced.pkl')
metadata = SingleTableMetadata()
real_data = real_data.loc[real_data["operator"] == operator].drop(columns= ["operator"])
if real_data.shape[0] > 100000:
    temp_real_data = real_data.sample(100000)
if real_data.shape[0] > 50000:
    num_synth_samples = 50000
else:
    num_synth_samples = real_data.shape[0]
real_data = real_data.drop(columns= ["purchaseremaildomain", "recipientemaildomain"])

metadata.detect_from_dataframe(real_data)
metadata.update_column("isFraud", sdtype= "boolean")
metadata.update_column(column_name='IdentityID', sdtype='id')
metadata.update_column(column_name='TransactionID', sdtype='numerical')
for column in ["productcategory", "cardtype", "cardcountry", "billingcountry"]:
    metadata.update_column(column, sdtype="categorical")
metadata.set_sequence_key(column_name='IdentityID')
metadata.set_sequence_index(column_name='TransactionID')
metadata.remove_primary_key()

## ParamSearch TIMEGAN
sweep_config = {
    "name": "TIMEGAN_{}".format(operator),
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "Quality Score"},
    "parameters": {
        "max_sequence_len": {"min": 5, "max": 40},
        "recursive_module": {"values": ["gru", "lstm", "rnn"]},
        "hidden_dim": {"values": [16, 32, 64]},
        "num_layers": {"min": 2, "max": 4},
        "metric_iteration": {"min": 2, "max": 8},
        "beta1": {"min": 0.5, "max": 0.998},
        "gamma": {"min": 0.5, "max": 2.0},
        "encoder_loss_weight_s": {"min": 0.05, "max": 0.5},
        "encoder_loss_weight_0": {"min": 5, "max": 20},
        "generator_loss_weight": {"min": 50, "max": 200},
        "generator_steps": {"min": 1, "max": 5},
        "learning_rate": {"min": 0.00001, "max": 0.001},
        "epochs": {"min": 100, "max": 1000},
        "batch_size": {"values": [256]}
    },
}

sweep_id = wandb.sweep(sweep=sweep_config, project="IEEE-CIS_paramSearch", entity="SyntheticFinancialDataEcosystem")
#sweep_id = "syntheticFinancialDataEcosystem/IEEE-CIS_paramSearch/kg413ix8"


## Truncate sequences
def truncate_sequence(group, max_len, min_len, id_column):
    if len(group) <= max_len and len(group) >= min_len:
        group[id_column] = group[id_column].apply(lambda x: f"{x}_0")
        return group
    elif len(group) > max_len:
        out = pd.DataFrame(columns=group.columns)
        for i in range(len(group) // max_len):
            seq = group.sample(min(len(group), max_len))
            seq[id_column] = seq[id_column].apply(lambda x: f"{x}_{i}")
            if out.empty:
                out = seq
            else:
                out = pd.concat((out, seq))
            group = group.drop(seq.index)
        return out
    else:
        return pd.DataFrame(columns=group.columns)

def main():
    wandb.init(project="IEEE-CIS_paramSearch", entity="SyntheticFinancialDataEcosystem")

    temp_real_data = real_data.groupby(["IdentityID"]).apply(truncate_sequence,max_len=wandb.config["max_sequence_len"], min_len=2, id_column="IdentityID").reset_index(drop=True)

    synthesizer = TIMEGANSynthesizer(metadata, context_columns= context_columns,
                                  max_sequence_len=wandb.config["max_sequence_len"],
                                  recursive_module=wandb.config["recursive_module"],
                                  hidden_dim=wandb.config["hidden_dim"],
                                  num_layers=wandb.config["num_layers"],
                                  metric_iteration=wandb.config["metric_iteration"],
                                  beta1=wandb.config["beta1"],
                                  gamma=wandb.config["gamma"],
                                  encoder_loss_weight_s=wandb.config["encoder_loss_weight_s"],
                                  encoder_loss_weight_0=wandb.config["encoder_loss_weight_0"],
                                  generator_loss_weight=wandb.config["generator_loss_weight"],
                                  generator_steps=wandb.config["generator_steps"],
                                  learning_rate=wandb.config["learning_rate"],
                                  epochs=wandb.config["epochs"], batch_size=wandb.config["batch_size"],
                                  verbose=True, use_wandb=True, cuda= True)
    synthesizer.fit(data=temp_real_data)
    synthetic_data = synthesizer.sample(num_rows=num_synth_samples)
    diagnostic_report = run_diagnostic(real_data=temp_real_data, synthetic_data=synthetic_data, metadata=metadata)
    quality_report = evaluate_quality(real_data=temp_real_data, synthetic_data=synthetic_data, metadata=metadata)
    quality_score = np.mean(list(quality_report.get_properties().set_index("Property")["Score"].to_dict().values()))
    results_dict = {**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(),
                    **quality_report.get_properties().set_index("Property")["Score"].to_dict(),
                    **{"Quality Score": quality_score}}
    wandb.log(results_dict)
    wandb.finish()


wandb.agent(sweep_id, function=main, count=20)

