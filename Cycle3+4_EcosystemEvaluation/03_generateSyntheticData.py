import pickle

import numpy as np
import pandas as pd
import os
from pathlib import Path

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from modified_sitepackages.sdv.metadata import SingleTableMetadata
from modified_sitepackages.sdv.single_table import TVAESynthesizer
from modified_sitepackages.sdv.evaluation.single_table import run_diagnostic, evaluate_quality

import wandb
wandb_api = wandb.Api()

def sampleDataWithoutDuplicates(sample_synthesizer, num_samples, real_data, target_col= None):
    # generate synthetic data
    max_iters = 15
    i = 0
    synth = pd.DataFrame(columns= real_data.columns)
    if target_col:
        pct_fraud = real_data[target_col].mean()
        pos_samples = num_samples * pct_fraud
        neg_samples = num_samples * (1 - pct_fraud)
        while (synth.loc[synth[target_col].astype(bool)].shape[0] < pos_samples or synth.loc[~synth[target_col].astype(bool)].shape[0] < neg_samples) and i < max_iters:
            print("Generating Data (iteration: {})".format(i))
            temp_synth = sample_synthesizer.sample(num_rows=num_samples)
            # ensure there is no leakage of real data
            temp_synth = pd.concat((temp_synth, real_data)).drop_duplicates(keep="last").drop(real_data.index)
            synth = pd.concat((synth, temp_synth), axis=0)
            i += 1
    else:
        while synth.shape[0] < num_samples and i < max_iters:
            print("Generating Data (iteration: {})".format(i))
            temp_synth = sample_synthesizer.sample(num_rows=num_samples)
            # ensure there is no leakage of real data
            temp_synth = pd.concat((temp_synth, real_data)).drop_duplicates(keep="last").drop(real_data.index)
            synth = pd.concat((synth, temp_synth), axis=0)
            i += 1
    return synth


if not os.path.exists("./working"):
    os.makedirs("./working")
if not os.path.exists("./synth"):
    os.makedirs("./synth")
if not os.path.exists("./model"):
    os.makedirs("./model")

singTab_snyths = [TVAESynthesizer]

data = ["IBM-CCF", "IBM-AML"]
data_paths = ["./data/IBM-CCF/processed/IBM-CCF.pkl", "./data/IBM-AML/processed/IBM-AML.pkl"]

for name, path in zip(data, data_paths):
    X = pd.read_pickle(path)
    X = X.sort_values(by=["Timestamp"])
    X = X.drop(columns=["Timestamp"])

    # reduce dataset for testing
    X = X.sample(50000, random_state=42)

    if name == "IBM-AML":
        X = X.loc[~((X["From Bank"] == "Unknown") | (X["To Bank"] == "Unknown"))]
        X = X.rename(columns={"Is Laundering": "target"})
    elif name == "IBM-CCF":
        X = X.rename(columns={"Is Fraud?": "target"})

    X["target"] = X["target"].astype(bool)

    train_X, test_X = train_test_split(X, shuffle=False, test_size=0.3, random_state=42)
    train_X = train_X.reset_index(drop=False)
    test_X = test_X.reset_index(drop=False)

    metadata_full = SingleTableMetadata()
    metadata_full.detect_from_dataframe(train_X)
    metadata_sep = SingleTableMetadata()
    metadata_sep.detect_from_dataframe(train_X.drop("target", axis=1))
    # update metadata
    if name == "IBM-AML":
        metadata_full.update_column(column_name='Receiving Currency', sdtype='categorical')
        metadata_sep.update_column(column_name='Receiving Currency', sdtype='categorical')
        metadata_full.update_column(column_name='Payment Currency', sdtype='categorical')
        metadata_sep.update_column(column_name='Payment Currency', sdtype='categorical')
        metadata_full.update_column(column_name='Sender', sdtype='id')
        metadata_sep.update_column(column_name='Sender', sdtype='id')
        metadata_full.update_column(column_name='Receiver', sdtype='id')
        metadata_sep.update_column(column_name='Receiver', sdtype='id')
        metadata_full.update_column(column_name='To Branch', sdtype='id')
        metadata_sep.update_column(column_name='To Branch', sdtype='id')
        metadata_full.update_column(column_name='From Branch', sdtype='id')
        metadata_sep.update_column(column_name='From Branch', sdtype='id')
        metadata_full.update_column(column_name='To Bank', sdtype='categorical')
        metadata_sep.update_column(column_name='To Bank', sdtype='categorical')
        metadata_full.update_column(column_name='From Bank', sdtype='categorical')
        metadata_sep.update_column(column_name='From Bank', sdtype='categorical')
        metadata_full.update_column(column_name='Currency Conversion', sdtype='boolean')
        metadata_sep.update_column(column_name='Currency Conversion', sdtype='boolean')
        metadata_full.update_column(column_name='Intrabank Transfer', sdtype='boolean')
        metadata_sep.update_column(column_name='Intrabank Transfer', sdtype='boolean')
        metadata_full.update_column(column_name='target', sdtype='boolean')
        if os.path.isfile("./working/IBM-AML_metadata.json"):
            os.remove("./working/IBM-AML_metadata.json")
        metadata_full.save_to_json("./working/IBM-AML_metadata.json")
    elif name == "IBM-CCF":
        metadata_full.update_column(column_name='User', sdtype='id')
        metadata_sep.update_column(column_name='User', sdtype='id')
        metadata_full.update_column(column_name='Card', sdtype='numerical')
        metadata_sep.update_column(column_name='Card', sdtype='numerical')
        metadata_full.update_column(column_name='Use Chip', sdtype='categorical')
        metadata_sep.update_column(column_name='Use Chip', sdtype='categorical')
        metadata_full.update_column(column_name='Merchant Name', sdtype='id')
        metadata_sep.update_column(column_name='Merchant Name', sdtype='id')
        metadata_full.update_column(column_name='Merchant State', sdtype='categorical')
        metadata_sep.update_column(column_name='Merchant State', sdtype='categorical')
        metadata_full.update_column(column_name='Zip', sdtype='postcode')
        metadata_sep.update_column(column_name='Zip', sdtype='postcode')
        metadata_full.update_column(column_name='MCC', sdtype='categorical')
        metadata_sep.update_column(column_name='MCC', sdtype='categorical')
        metadata_full.update_column(column_name='Errors?', sdtype='categorical')
        metadata_sep.update_column(column_name='Errors?', sdtype='categorical')
        metadata_full.update_column(column_name='Card ID', sdtype='id')
        metadata_sep.update_column(column_name='Card ID', sdtype='id')
        metadata_full.update_column(column_name='in Home State?', sdtype='boolean')
        metadata_sep.update_column(column_name='in Home State?', sdtype='boolean')
        metadata_full.update_column(column_name='target', sdtype='boolean')
        if os.path.isfile("./working/IBM-CCF_metadata.json"):
            os.remove("./working/IBM-CCF_metadata.json")
        metadata_full.save_to_json("./working/IBM-CCF_metadata.json")
    else:
        raise ValueError("Unknown dataset")
    metadata_full.update_column(column_name='index', sdtype='id')
    metadata_sep.update_column(column_name='index', sdtype='id')
    metadata_full.set_primary_key(column_name='index')
    metadata_sep.set_primary_key(column_name='index')

    # split the datasets into banks
    if name == "IBM-AML":
        train_bank = [name for name, _ in train_X.groupby("To Bank")]
        train_X_1 = [x for _, x in train_X.groupby("To Bank")]
        train_X_2 = [train_X.loc[train_X["From Bank"] == bank] for bank in train_bank]
        train_X = [pd.concat([d1, d2]).drop_duplicates() for d1, d2 in zip(train_X_1, train_X_2)]
        test_X_1 = [test_X.loc[test_X["To Bank"] == bank] for bank in train_bank]
        test_X_2 = [test_X.loc[test_X["From Bank"] == bank] for bank in train_bank]
        test_X = [pd.concat([d1, d2]).drop_duplicates() for d1, d2 in zip(test_X_1, test_X_2)]
    elif name == "IBM-CCF":
        train_bank = [name for name, _ in train_X.groupby("Bank")]
        train_X = [x for _, x in train_X.groupby("Bank")]
        test_X = [test_X.loc[test_X["Bank"] == bank] for bank in train_bank]
    else:
        raise ValueError("Unknown dataset")

    for synth in singTab_snyths:
        for type in ["full", "full+sep", "full+sepOS_05", "full+sepOS_10", "full+sepOS_20", "full+sepOS_50", "fullOS_05", "fullOS_10", "fullOS_20", "fullOS_50", "sep", "sepOS_05", "sepOS_10", "sepOS_20", "sepOS_50", "sepPreOS_05", "sepPreOS_10", "sepPreOS_15", "sepPreOS_20", "sepPreOS_25", "sepPreOS_50"]:
            if not os.path.isfile(os.path.join("./synth", "{}_{}_{}.pkl".format(name, synth.__name__, type))):
                train_synth_X = pd.DataFrame()

                if "full" in type:
                    for sub_train_X, sub_test_X in tqdm(list(zip(train_X, test_X)), desc="{} {} {}".format(name, synth.__name__, type)):
                        ## get model_param_dict from wandb
                        if name == "IBM-AML":
                            bank = pd.concat((sub_train_X["To Bank"], sub_train_X["From Bank"]), axis=0).value_counts().index[0]
                        elif name == "IBM-CCF":
                            bank = sub_train_X["Bank"].value_counts().index[0]
                        else:
                            raise ValueError("Unknown dataset")
                        with open('./data/{}/processed/encoders.pkl'.format(name), 'rb') as file:
                            encoder = pickle.load(file)
                        bank_id = encoder["Bank"].transform([bank])[0]
                        sweeps = wandb_api.project("{}_paramSearch".format(name), entity="SyntheticFinancialDataEcosystem").sweeps()
                        sweeps = [sweep for sweep in sweeps if str(bank_id) in sweep.config["name"]]
                        assert len(sweeps) == 1
                        run = sweeps[0].best_run()

                        wandb.init(entity="syntheticFinancialDataEcosystem", project="{}_generation".format(name), config={**run.config}, tags=["TVAE", str(bank_id), type])

                        if "fullOS" in type:
                            # synthesize full OS data table
                            os_ratio = int(type.split("_")[1]) / 100
                            ros = RandomOverSampler(sampling_strategy= os_ratio, random_state=42)
                            sub_train_X_OS, _ = ros.fit_resample(sub_train_X, sub_train_X["target"])
                            sub_train_X_OS.iloc[sub_train_X["index"].shape[0]:, 0] = list(range(sub_train_X["index"].max() + 1, sub_train_X["index"].max() + 1 + (sub_train_X_OS.shape[0] - sub_train_X.shape[0])))
                            synthesizer = synth(metadata_full,
                                  embedding_dim=run.config["embedding_dim"],
                                  compress_dims=run.config["compress_dims"],
                                  decompress_dims=run.config["decompress_dims"],
                                  l2scale=run.config["l2scale"], loss_factor=run.config["loss_factor"],
                                  learning_rate=run.config["learning_rate"],
                                  epochs=run.config["epochs"], batch_size=run.config["batch_size"],
                                  verbose=True, use_wandb=True, cuda= "cuda:0")
                            if not os.path.exists("./model/TVAE_{}_{}_{}.pkl".format(name, bank_id, "fullOS_{}".format(os_ratio))):
                                synthesizer.fit(sub_train_X_OS)
                                synthesizer.save("./model/TVAE_{}_{}_{}.pkl".format(name, bank_id, "fullOS_{}".format(os_ratio)))
                            else:
                                synthesizer = synthesizer.load("./model/TVAE_{}_{}_{}.pkl".format(name, bank_id, "fullOS_{}".format(os_ratio)))
                            sub_train_synth_X = synthesizer.sample(sub_train_X.shape[0])
                            train_synth_X = pd.concat([train_synth_X, sub_train_synth_X])

                            diagnostic_report = run_diagnostic(real_data=sub_train_X_OS, synthetic_data=sub_train_synth_X, metadata=metadata_full)
                            quality_report = evaluate_quality(real_data=sub_train_X_OS, synthetic_data=sub_train_synth_X, metadata=metadata_full)

                            quality_score = np.mean(list(quality_report.get_properties().set_index("Property")["Score"].to_dict().values()))
                            results_dict = {
                                **diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(),
                                **quality_report.get_properties().set_index("Property")["Score"].to_dict(),
                                **{"Quality Score": quality_score}}
                            wandb.log(results_dict)
                            wandb.finish()

                        elif "full" in type:
                            if type == "full":
                                # synthesize full data table
                                synthesizer = synth(metadata_full,
                                                    embedding_dim=run.config["embedding_dim"],
                                                    compress_dims=run.config["compress_dims"],
                                                    decompress_dims=run.config["decompress_dims"],
                                                    l2scale=run.config["l2scale"],
                                                    loss_factor=run.config["loss_factor"],
                                                    learning_rate=run.config["learning_rate"],
                                                    epochs=run.config["epochs"], batch_size=run.config["batch_size"],
                                                    verbose=True, use_wandb=True, cuda="cuda:0")
                                if not os.path.exists("./model/TVAE_{}_{}_{}.pkl".format(name, bank_id, "full")):
                                    synthesizer.fit(sub_train_X)
                                    synthesizer.save("./model/TVAE_{}_{}_{}.pkl".format(name, bank_id, "full"))
                                else:
                                    synthesizer = synthesizer.load("./model/TVAE_{}_{}_{}.pkl".format(name, bank_id, "full"))
                                sub_train_synth_X = synthesizer.sample(sub_train_X.shape[0])
                                train_synth_X = pd.concat([train_synth_X, sub_train_synth_X])

                                diagnostic_report = run_diagnostic(real_data=sub_train_X, synthetic_data=sub_train_synth_X, metadata=metadata_full)
                                quality_report = evaluate_quality(real_data=sub_train_X, synthetic_data=sub_train_synth_X, metadata=metadata_full)

                                quality_score = np.mean(list(quality_report.get_properties().set_index("Property")["Score"].to_dict().values()))
                                results_dict = {
                                    **diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(),
                                    **quality_report.get_properties().set_index("Property")["Score"].to_dict(),
                                    **{"Quality Score": quality_score}}
                                wandb.log(results_dict)
                                wandb.finish()

                            elif "full+sep" in type:
                                # synthesize full data table with separated synthesizer
                                synthesizer = synth(metadata_full,
                                                    embedding_dim=run.config["embedding_dim"],
                                                    compress_dims=run.config["compress_dims"],
                                                    decompress_dims=run.config["decompress_dims"],
                                                    l2scale=run.config["l2scale"],
                                                    loss_factor=run.config["loss_factor"],
                                                    learning_rate=run.config["learning_rate"],
                                                    epochs=run.config["epochs"], batch_size=run.config["batch_size"],
                                                    verbose=True, use_wandb=True, cuda="cuda:0")
                                if not os.path.exists("./model/TVAE_{}_{}_{}.pkl".format(name, bank_id, "full")):
                                    synthesizer.fit(sub_train_X)
                                    synthesizer.save("./model/TVAE_{}_{}_{}.pkl".format(name, bank_id, "full"))
                                else:
                                    synthesizer = synthesizer.load("./model/TVAE_{}_{}_{}.pkl".format(name, bank_id, "full"))
                                sub_train_synth_X = synthesizer.sample(sub_train_X.shape[0])

                                if not os.path.exists("./model/TVAE_{}_{}_{}.pkl".format(name, bank_id, "full+sep_True")):
                                    synthesizer.refit(sub_train_X.loc[sub_train_X["target"] == True], epochs= int(run.config["epochs"]*0.05))
                                    synthesizer.save("./model/TVAE_{}_{}_{}.pkl".format(name, bank_id, "full+sep_True"))
                                else:
                                    synthesizer = synthesizer.load("./model/TVAE_{}_{}_{}.pkl".format(name, bank_id, "full+sep_True"))
                                if not "OS" in type:
                                    if True in sub_train_synth_X["target"].value_counts().index:
                                        sub_train_synth_X_positive = synthesizer.sample(sub_train_X["target"].value_counts()[True] - sub_train_synth_X["target"].value_counts()[True])
                                    else:
                                        sub_train_synth_X_positive = synthesizer.sample(sub_train_X["target"].value_counts()[True])

                                    diagnostic_report = run_diagnostic(real_data=sub_train_X, synthetic_data=pd.concat([sub_train_synth_X, sub_train_synth_X_positive]), metadata=metadata_full)
                                    quality_report = evaluate_quality(real_data=sub_train_X, synthetic_data=pd.concat([sub_train_synth_X, sub_train_synth_X_positive]), metadata=metadata_full)

                                else:
                                    os_ratio = int(type.split("_")[1]) / 100
                                    if True in sub_train_synth_X["target"].value_counts().index:
                                        sub_train_synth_X_positive = synthesizer.sample(int((len(sub_train_synth_X["target"]) * os_ratio) - sub_train_synth_X["target"].value_counts()[True]))
                                    else:
                                        sub_train_synth_X_positive = synthesizer.sample(int(len(sub_train_synth_X["target"]) * os_ratio))

                                    os_ratio = int(type.split("_")[1]) / 100
                                    ros = RandomOverSampler(sampling_strategy=os_ratio, random_state=42)
                                    sub_train_X_OS, _ = ros.fit_resample(sub_train_X, sub_train_X["target"])

                                    diagnostic_report = run_diagnostic(real_data=sub_train_X_OS, synthetic_data=pd.concat([sub_train_synth_X, sub_train_synth_X_positive]), metadata=metadata_full)
                                    quality_report = evaluate_quality(real_data=sub_train_X_OS, synthetic_data=pd.concat([sub_train_synth_X, sub_train_synth_X_positive]), metadata=metadata_full)

                                sub_train_synth_X_positive["target"] = True
                                train_synth_X = pd.concat([train_synth_X, sub_train_synth_X, sub_train_synth_X_positive])

                                quality_score = np.mean(list(quality_report.get_properties().set_index("Property")["Score"].to_dict().values()))
                                results_dict = {
                                    **diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(),
                                    **quality_report.get_properties().set_index("Property")["Score"].to_dict(),
                                    **{"Quality Score": quality_score}}
                                wandb.log(results_dict)
                                wandb.finish()

                else:
                    for i, (sub_train_X, sub_test_X) in enumerate(tqdm(list(zip(train_X, test_X)), desc="{} {} {}".format(name, synth.__name__, type))):

                        ## get model_param_dict from wandb
                        if name == "IBM-AML":
                            bank = pd.concat((sub_train_X["To Bank"], sub_train_X["From Bank"]), axis=0).value_counts().index[0]
                        elif name == "IBM-CCF":
                            bank = sub_train_X["Bank"].value_counts().index[0]
                        else:
                            raise ValueError("Unknown dataset")
                        with open('./data/{}/processed/encoders.pkl'.format(name), 'rb') as file:
                            encoder = pickle.load(file)
                        bank_id = encoder["Bank"].transform([bank])[0]
                        sweeps = wandb_api.project("{}_paramSearch".format(name),
                                                   entity="SyntheticFinancialDataEcosystem").sweeps()
                        sweeps = [sweep for sweep in sweeps if str(bank_id) in sweep.config["name"]]
                        assert len(sweeps) == 1
                        run = sweeps[0].best_run()

                        wandb.init(entity="syntheticFinancialDataEcosystem", project="{}_generation".format(name), config={**run.config}, tags=["TVAE", str(bank_id), type + "_negative"])

                        synthesizer = synth(metadata_sep,
                                            embedding_dim=run.config["embedding_dim"],
                                            compress_dims=run.config["compress_dims"],
                                            decompress_dims=run.config["decompress_dims"],
                                            l2scale=run.config["l2scale"], loss_factor=run.config["loss_factor"],
                                            learning_rate=run.config["learning_rate"],
                                            epochs=run.config["epochs"], batch_size=run.config["batch_size"],
                                            verbose=True, use_wandb=True, cuda="cuda:0")
                        if not os.path.exists("./model/TVAE_{}_{}_{}.pkl".format(name, bank_id, "sep_False")):
                            synthesizer.fit(sub_train_X.loc[sub_train_X["target"] == False, sub_train_X.columns != "target"])
                            synthesizer.save("./model/TVAE_{}_{}_{}.pkl".format(name, bank_id, "sep_False"))
                        else:
                            synthesizer = synthesizer.load("./model/TVAE_{}_{}_{}.pkl".format(name, bank_id, "sep_False"))

                        sub_train_synth_X = synthesizer.sample(sub_train_X.loc[sub_train_X["target"] == False, sub_train_X.columns != "target"].shape[0])
                        sub_train_synth_X["target"] = False
                        train_synth_X = pd.concat([train_synth_X, sub_train_synth_X])
                        sub_train_synth_X_complete = sub_train_synth_X

                        diagnostic_report = run_diagnostic(real_data=sub_train_X.loc[sub_train_X["target"] == False, sub_train_X.columns != "target"], synthetic_data= sub_train_synth_X.loc[:, sub_train_synth_X.columns != "target"], metadata=metadata_sep)
                        quality_report = evaluate_quality(real_data=sub_train_X.loc[sub_train_X["target"] == False, sub_train_X.columns != "target"], synthetic_data= sub_train_synth_X.loc[:, sub_train_synth_X.columns != "target"], metadata=metadata_sep)
                        quality_score = np.mean(list(quality_report.get_properties().set_index("Property")["Score"].to_dict().values()))
                        results_dict = {
                            **diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(),
                            **quality_report.get_properties().set_index("Property")["Score"].to_dict(),
                            **{"Quality Score": quality_score}}
                        wandb.log(results_dict)
                        wandb.finish()

                        wandb.init(entity="syntheticFinancialDataEcosystem", project="{}_generation".format(name), config={**run.config}, tags=["TVAE", str(bank_id), type + "_positive"])

                        if not "sepPre" in type:
                            synthesizer = synth(metadata_sep,
                                  embedding_dim=run.config["embedding_dim"],
                                  compress_dims=run.config["compress_dims"],
                                  decompress_dims=run.config["decompress_dims"],
                                  l2scale=run.config["l2scale"], loss_factor=run.config["loss_factor"],
                                  learning_rate=run.config["learning_rate"],
                                  epochs=run.config["epochs"], batch_size=run.config["batch_size"],
                                  verbose=True, use_wandb=True, cuda= "cuda:0")
                            if not os.path.exists("./model/TVAE_{}_{}_{}.pkl".format(name, bank_id, "sep_True")):
                                synthesizer.fit(sub_train_X.loc[sub_train_X["target"] == True, sub_train_X.columns != "target"])
                                synthesizer.save("./model/TVAE_{}_{}_{}.pkl".format(name, bank_id, "sep_True"))
                            else:
                                synthesizer = synthesizer.load("./model/TVAE_{}_{}_{}.pkl".format(name, bank_id, "sep_True"))
                        else:
                            #synthesizer.fit(sub_train_X.loc[sub_train_X["target"] == True, sub_train_X.columns != "target"])
                            if not os.path.exists("./model/TVAE_{}_{}_{}.pkl".format(name, bank_id, "sep_False+pre_True")):
                                synthesizer.refit(sub_train_X.loc[sub_train_X["target"] == True, sub_train_X.columns != "target"], epochs= int(run.config["epochs"]*0.05))
                                synthesizer.save("./model/TVAE_{}_{}_{}.pkl".format(name, bank_id, "sep_False+pre_True"))
                            else:
                                synthesizer = synthesizer.load("./model/TVAE_{}_{}_{}.pkl".format(name, bank_id, "sep_False+pre_True"))

                        if "OS" in type:
                            os_ratio = int(type.split("_")[1]) / 100
                            sub_train_synth_X = synthesizer.sample(int(sub_train_X.loc[sub_train_X["target"] == False, sub_train_X.columns != "target"].shape[0] * os_ratio))
                        else:
                            sub_train_synth_X = synthesizer.sample(sub_train_X.loc[sub_train_X["target"] == True, sub_train_X.columns != "target"].shape[0])

                        sub_train_synth_X["target"] = True
                        train_synth_X = pd.concat([train_synth_X, sub_train_synth_X])
                        sub_train_synth_X_complete = pd.concat([sub_train_synth_X_complete, sub_train_synth_X], axis=0)

                        diagnostic_report = run_diagnostic(real_data=sub_train_X.loc[sub_train_X["target"] == False, sub_train_X.columns != "target"], synthetic_data=sub_train_synth_X.loc[:, sub_train_synth_X.columns != "target"], metadata=metadata_sep)
                        quality_report = evaluate_quality(real_data=sub_train_X.loc[sub_train_X["target"] == False, sub_train_X.columns != "target"], synthetic_data=sub_train_synth_X.loc[:, sub_train_synth_X.columns != "target"], metadata=metadata_sep)
                        quality_score = np.mean(list(quality_report.get_properties().set_index("Property")["Score"].to_dict().values()))
                        results_dict = {
                            **diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(),
                            **quality_report.get_properties().set_index("Property")["Score"].to_dict(),
                            **{"Quality Score": quality_score}}
                        wandb.log(results_dict)
                        wandb.finish()

                        wandb.init(entity="syntheticFinancialDataEcosystem", project="{}_generation".format(name), tags=["TVAE", str(bank_id), type])
                        diagnostic_report = run_diagnostic(real_data=sub_train_X, synthetic_data=sub_train_synth_X_complete, metadata=metadata_full)
                        quality_report = evaluate_quality(real_data=sub_train_X, synthetic_data=sub_train_synth_X_complete, metadata=metadata_full)
                        quality_score = np.mean(list(quality_report.get_properties().set_index("Property")["Score"].to_dict().values()))
                        results_dict = {
                            **diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(),
                            **quality_report.get_properties().set_index("Property")["Score"].to_dict(),
                            **{"Quality Score": quality_score}}
                        wandb.log(results_dict)
                        wandb.finish()

                # save the synthesized data
                train_synth_X["type"] = "synthetic"
                train_X_save = pd.concat(train_X)
                test_X_save = pd.concat(test_X)
                train_X_save["type"] = "real train"
                test_X_save["type"] = "real test"
                train_synth_X = pd.concat([train_X_save, test_X_save, train_synth_X])
                train_synth_X.to_pickle(os.path.join("./synth", "{}_{}_{}.pkl".format(name,synth.__name__, type)))