import os

import pandas as pd
import numpy as np

from modified_sitepackages.sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GMMSynthesizer
from modified_sitepackages.sdv.sequential import TIMEGANSynthesizer
from modified_sitepackages.sdv.metadata import SingleTableMetadata

from modified_sitepackages.sdv.evaluation.single_table import run_diagnostic, evaluate_quality

import wandb


operator = "discover"

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

## ParamSearch GMM
sweep_config = {
    "name": "GMM_{}".format(operator),
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "Quality Score"},
    "parameters": {
        "n_components": {"min": 1, "max": 40},
        "covariance_type": {"values": ["full", "tied", "diag", "spherical"]},
        "max_iter": {"min": 50, "max": 300},
        "init_params": {"values": ["kmeans", "k-means++", "random", "random_from_data"]}
    },
}

sweep_id = wandb.sweep(sweep=sweep_config, project="IEEE-CIS_paramSearch", entity="SyntheticFinancialDataEcosystem")
#sweep_id = "syntheticFinancialDataEcosystem/IEEE-CIS_paramSearch/kg413ix8"

def main():
    wandb.init(project="IEEE-CIS_paramSearch", entity="SyntheticFinancialDataEcosystem")
    synthesizer = GMMSynthesizer(metadata,
                                 n_components=wandb.config["n_components"],
                                 covariance_type=wandb.config["covariance_type"],
                                 max_iter=wandb.config["max_iter"],
                                 init_params=wandb.config["init_params"],
                                 verbose=True, use_wandb=True)
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
