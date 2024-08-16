import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from modified_sitepackages.sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GMMSynthesizer
from modified_sitepackages.sdv.sequential import TIMEGANSynthesizer
from modified_sitepackages.sdv.metadata import SingleTableMetadata

from modified_sitepackages.sdv.evaluation.single_table import run_diagnostic, evaluate_quality

import wandb
wandb_api = wandb.Api()


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

if not os.path.exists("./model/"):
    os.makedirs("./model/")
if not os.path.exists("./synth/"):
    os.makedirs("./synth/")

# load data
real_data = pd.read_pickle('./data/IEEE-CIS/processed/data_reduced.pkl')
real_data = real_data.drop(columns= ["purchaseremaildomain", "recipientemaildomain"])

# algorithms to evaluate
algorithms = {
    "GMM": (GMMSynthesizer, "table", False),
    "TVAE": (TVAESynthesizer, "table", True),
    "CTGAN": (CTGANSynthesizer, "table", True),
    "TIMEGAN": (TIMEGANSynthesizer, "sequential", True)
}
algorithm_parameters = {
    "CTGAN": [],
    "TVAE": [],
    "GMM": [],
    "TIMEGAN": []
}

num_synth_data = 200000

# split the dataset into operators
real_data["operator"] = real_data["operator"].astype(str).astype("category")
df_all = real_data.groupby('operator')

df_grouped = []
for operator, data_group in df_all:
    data_group = data_group.drop(columns=["operator"])
    data_group_train, _ = train_test_split(data_group, train_size=0.7, random_state=42)
    df_grouped.append((operator, data_group_train))

for algorithm_name, (algorithm, algorithm_type, use_cuda) in algorithms.items():
    for operator, df_group in df_grouped:
        if not os.path.exists("./synth/synth_{}_{}_unbalanced.pkl".format(algorithm_name, operator)):
            print("Processing: {} - {}".format(algorithm_name, operator))

            if "operator" in df_group.columns:
                df_group = df_group.drop(columns= ["operator"])

            if not algorithm_type == "sequential":
                df_group = df_group.drop(columns=['IdentityID', 'TransactionID'])

            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(df_group)
            metadata.update_column("isFraud", sdtype="boolean")
            for column in ["productcategory", "cardtype", "cardcountry", "billingcountry"]:
                metadata.update_column(column, sdtype="categorical")

            if algorithm_type == "sequential":
                metadata.update_column(column_name='IdentityID', sdtype='id')
                metadata.update_column(column_name='TransactionID', sdtype='numerical')
                metadata.set_sequence_key(column_name='IdentityID')
                metadata.set_sequence_index(column_name='TransactionID')
                metadata.remove_primary_key()

            pct_fraud = df_group["isFraud"].mean()

            ## get model_param_dict from wandb
            sweeps = wandb_api.project("IEEE-CIS_paramSearch", entity= "SyntheticFinancialDataEcosystem").sweeps()
            sweeps = [sweep for sweep in sweeps if algorithm_name in sweep.config["name"] and operator in sweep.config["name"]]
            assert len(sweeps) == 1
            run = sweeps[0].best_run()

            # preprocess data
            if algorithm_type == "sequential":
                df_group = df_group.groupby(["IdentityID"]).apply(truncate_sequence, max_len=run.config["max_sequence_len"], min_len=2, id_column="IdentityID").reset_index(drop=True)
            # create synthetic data model
            wandb.init(entity= "syntheticFinancialDataEcosystem", project= "IEEE-CIS_evaluation", config= {**run.config}, tags=[algorithm_name, operator])

            if use_cuda:
                synthesizer = algorithm(metadata, **run.config, verbose=True, use_wandb=True, cuda= True)
            else:
                synthesizer = algorithm(metadata, **run.config, verbose=True, use_wandb=True)
            synthesizer.fit(data= df_group)

            # save the model
            synthesizer.save("./model/{}_{}.pkl".format(algorithm_name, operator))
            synthesizer.load("./model/{}_{}.pkl".format(algorithm_name, operator))

            # generate synthetic data
            max_iters = 15
            pos_samples = num_synth_data * pct_fraud
            neg_samples = num_synth_data * (1 - pct_fraud)
            i = 0
            synth = pd.DataFrame(columns= df_group.columns)
            while (synth.loc[synth["isFraud"] == 1].shape[0] < pos_samples or synth.loc[synth["isFraud"] == 0].shape[0] < neg_samples) and i < max_iters:
                print("Generating Data (iteration: {})".format(i))
                temp_synth = synthesizer.sample(num_rows=num_synth_data)
                # ensure there is no leakage of real data
                temp_synth = pd.concat((temp_synth, df_group)).drop_duplicates(keep= "last").iloc[:len(temp_synth), :]
                synth = pd.concat((synth, temp_synth), axis= 0)
                i += 1

            if len(synth.loc[synth["isFraud"] == 1]) > 0:
                synth = pd.concat((synth.loc[synth["isFraud"] == 1].sample(int(pos_samples)),
                                   synth.loc[synth["isFraud"] == 0].sample(int(neg_samples))), axis= 0)
            else:
                print("No fraudulent cases in synthetic sample.")
                synth = synth.loc[synth["isFraud"] == 0].sample(int(neg_samples))
            synth.to_pickle("./synth/synth_{}_{}_unbalanced.pkl".format(algorithm_name, operator))

            diagnostic_report = run_diagnostic(real_data=df_group, synthetic_data=synth, metadata=metadata)
            quality_report = evaluate_quality(real_data=df_group, synthetic_data=synth, metadata=metadata)
            results_dict = {**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(), **quality_report.get_properties().set_index("Property")["Score"].to_dict()}

            wandb.log(results_dict)

            wandb.finish()
        else:
            print("Skipping run: {} - {}".format(algorithm_name, operator))