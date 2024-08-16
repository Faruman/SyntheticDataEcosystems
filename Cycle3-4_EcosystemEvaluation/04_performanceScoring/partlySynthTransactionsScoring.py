import numpy as np
import pandas as pd
import os
from pathlib import Path
import pickle
from tqdm import tqdm

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import RandomOverSampler

from xgboost import XGBClassifier


data = ["IBM-CCF", "IBM-AML"]
os_ratio = 0.2
synthesizer = "TVAESynthesizer"

if not os.path.exists("../working"):
    os.makedirs("../working")

data_paths = ["../synth/", "../synth/"]

for name, path in zip(data, data_paths):
    #for df_type in ["full", "sep", "sepOS"]:
    ## to speed up only check full and sep, sepOS will be checked later
    for inclusion_pct in [0.5, 0.75]:
        for df_type in ["full",
                        "sep",
                        "fullOS_05", "fullOS_10", "fullOS_20", "fullOS_50",
                        "sepOS_05", "sepOS_10", "sepOS_20", "sepOS_50",
                        "full+sepOS_05",  "full+sepOS_10",  "full+sepOS_20",  "full+sepOS_50",
                        "sepPreOS_05", "sepPreOS_10", "sepPreOS_20", "sepPreOS_50"]:
            path = os.path.join(path, "{}_{}_{}.pkl".format(name, synthesizer, df_type))
            df = pd.read_pickle(path)

            df = df.drop(columns= ["index", "Card ID"])
            non_numerical_columns = [x for x in df.select_dtypes(include=['object']).columns if x not in ["type"]]
            if name == "IBM-AML":
                encoders = pickle.load(open('../data/IBM-AML/processed/encoders.pkl', 'rb'))
            elif name == "IBM-CCF":
                encoders = pickle.load(open('../data/IBM-CCF/processed/encoders.pkl', 'rb'))
            else:
                raise ValueError("Unknown dataset")
            for col in non_numerical_columns:
                print(col)
                df[col] = encoders[col].transform(df[col])

            df = df.reset_index(drop=True)
            # reduce dataset for testing
            # df = df.sample(frac=0.05, random_state=42)
            X = df.drop(columns=["target"])
            y = df["target"]
            included_banks = X["To Bank"].value_counts().index[:int(X["To Bank"].nunique() * inclusion_pct)]

            # split the datasets into banks
            if name == "IBM-AML":
                X = X.loc[~((X["From Bank"] == "Unknown") | (X["To Bank"] == "Unknown"))]
                X["Sender"] = X["Sender"] + X.index.max() + int(X.index.max() * (1 + os_ratio + 0.1))
                X["Receiver"] = X["Receiver"] + X.index.max() + int(X.index.max() * (1 + os_ratio + 0.1))
                train_X = X.loc[X["type"] == "real train"]
                train_X = train_X.drop(columns=["type"])
                train_X_synth = X.loc[X["type"] == "synthetic"]
                train_X_synth = train_X_synth.drop(columns=["type"])
                test_X = X.loc[X["type"] == "real test"]
                test_X = test_X.drop(columns=["type"])
                train_y = y.loc[X["type"] == "real train"]
                train_y_synth = y.loc[X["type"] == "synthetic"]
                test_y = y.loc[X["type"] == "real test"]
                train_bank = [name for name, _ in train_X.groupby("To Bank")]
                train_X_1 = [x for _, x in train_X.groupby("To Bank")]
                train_X_2 = [train_X.loc[train_X["From Bank"] == bank] for bank in train_bank]
                train_X = [pd.concat([d1, d2]).drop_duplicates() for d1, d2 in zip(train_X_1, train_X_2)]
                test_X_1 = [test_X.loc[test_X["To Bank"] == bank] for bank in train_bank]
                test_X_2 = [test_X.loc[test_X["From Bank"] == bank] for bank in train_bank]
                test_X = [pd.concat([d1, d2]).drop_duplicates() for d1, d2 in zip(test_X_1, test_X_2)]
            elif name == "IBM-CCF":
                X["User"] = X["User"] + int(X.index.max() * (1 + os_ratio + 0.1))
                train_X = X.loc[X["type"] == "real train"]
                train_X = train_X.drop(columns=["type"])
                train_X_synth = X.loc[X["type"] == "synthetic"]
                train_X_synth = train_X_synth.drop(columns=["type"])
                test_X = X.loc[X["type"] == "real test"]
                test_X = test_X.drop(columns=["type"])
                train_y = y.loc[X["type"] == "real train"]
                train_y_synth = y.loc[X["type"] == "synthetic"]
                test_y = y.loc[X["type"] == "real test"]
                train_bank = [name for name, _ in train_X.groupby("Bank")]
                train_X = [x for _, x in train_X.groupby("Bank")]
                test_X = [test_X.loc[test_X["Bank"] == bank] for bank in train_bank]
            else:
                raise ValueError("Unknown dataset")

            train_y = [train_y[train_X_sub.index] for train_X_sub in train_X]
            test_y = [test_y[test_X_sub.index] for test_X_sub in test_X]

            skip_transactionModel = False
            skip_transactionModelOS = False

            for mixin_pct in tqdm(list(range(0, 210, 10))[1:], desc="Mixin Percentage"):
                # for mixin_pct in tqdm(list(range(0, 110, 20))[1:], desc="Mixin Percentage"):
                mixin_pct = mixin_pct / 100
                # create transaction based model
                if not skip_transactionModel and not os.path.isfile("../working/partly/{}_{}_{}_{}_synthTransactionScoring_transactions.pkl".format(name, df_type, inclusion_pct, mixin_pct)):
                    print("{} - {} - {} - Transaction based model".format(name, df_type, mixin_pct))
                    results = []
                    test_pred = []
                    for sub_train_X, sub_train_y, sub_test_X, sub_test_y in zip(train_X, train_y, test_X, test_y):
                        # clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose= 1, random_state=42)
                        # clf = GradientBoostingClassifier(n_estimators=50, verbose=1, random_state=42)
                        if name == "IBM-AML":
                            sub_name = pd.concat([sub_train_X["To Bank"], sub_train_X["From Bank"]]).value_counts().idxmax()
                            synth_sample_size = int(sub_train_X.shape[0] * mixin_pct)
                            if sub_name in included_banks:
                                if synth_sample_size > train_X_synth.loc[(train_X_synth["To Bank"] != sub_name) & (train_X_synth["From Bank"] != sub_name)].shape[0]:
                                    train_X_synth_mixin = train_X_synth.loc[(train_X_synth["To Bank"] != sub_name) & (train_X_synth["From Bank"] != sub_name)].sample(synth_sample_size, random_state=42, replace=True)
                                else:
                                    train_X_synth_mixin = train_X_synth.loc[(train_X_synth["To Bank"] != sub_name) & (train_X_synth["From Bank"] != sub_name)].sample(synth_sample_size, random_state=42)
                            else:
                                train_X_synth_mixin = pd.DataFrame(columns=train_X_synth.columns)

                        elif name == "IBM-CCF":
                            sub_name = sub_train_X["Bank"].value_counts().idxmax()
                            synth_sample_size = int(sub_train_X.shape[0] * mixin_pct)
                            if sub_name in included_banks:
                                if synth_sample_size > train_X_synth.loc[train_X_synth["Bank"] != sub_name].shape[0]:
                                    train_X_synth_mixin = train_X_synth.loc[train_X_synth["Bank"] != sub_name].sample(synth_sample_size, random_state=42, replace=True)
                                else:
                                    train_X_synth_mixin = train_X_synth.loc[train_X_synth["Bank"] != sub_name].sample(synth_sample_size, random_state=42)
                            else:
                                train_X_synth_mixin = pd.DataFrame(columns=train_X_synth.columns)
                        else:
                            raise ValueError("Unknown dataset")

                        train_y_synth_mixin = train_y_synth[train_X_synth_mixin.index]
                        sub_train_X = pd.concat([sub_train_X, train_X_synth_mixin])
                        sub_train_y = pd.concat([sub_train_y, train_y_synth_mixin])

                        #clf = XGBClassifier(n_estimators=100, device="cuda", verbosity=2, random_state=42)
                        clf = make_pipeline(StandardScaler(), GridSearchCV(XGBClassifier(random_state=42, device= "cuda"),
                                                                           param_grid={"eta": [0.5, 0.1, 0.05],
                                                                                       "min_child_weight": [0.5, 1, 4],
                                                                                       "max_depth": [3, 6, 9],
                                                                                       "scale_pos_weight": [1, 0.1,
                                                                                                            0.025]},
                                                                           cv=3,
                                                                           refit=True,
                                                                           scoring="roc_auc",
                                                                           n_jobs=-1,
                                                                           verbose=1))
                        clf.fit(sub_train_X, sub_train_y)
                        sub_test_pred = clf.predict(sub_test_X)
                        test_pred.append(pd.DataFrame(sub_test_pred))
                        sub_cls_report = classification_report(sub_test_y, sub_test_pred)
                        sub_cls_report += "roc-auc-score                        {0:.4f}  {0:.4f}".format(
                            roc_auc_score(sub_test_y, sub_test_pred), sub_test_y.shape[0])
                        results.append(sub_cls_report)
                        print(sub_cls_report)

                    test_pred = pd.concat(test_pred)
                    test_real_y = pd.concat(test_y)
                    cls_report = classification_report(test_real_y, test_pred)
                    cls_report += "roc-auc-score                        {0:.4f}  {0:.4f}".format(
                        roc_auc_score(test_real_y, test_pred), test_real_y.shape[0])
                    print(cls_report)

                    with open("../working/partly/{}_{}_{}_{}_synthTransactionScoring_transactions.pkl".format(name, df_type, inclusion_pct, mixin_pct), "wb") as file:
                        pickle.dump({"overall": cls_report, "per_bank": results, "type": "transactions",
                                     "dataset": name, "mixin_percentage": mixin_pct, "included_banks": included_banks, "inclusion_pct": inclusion_pct,
                                     "synth_type": df_type}, file)

                # create transaction based model with Random Oversampling
                if not skip_transactionModelOS and not os.path.isfile("../working/partly/{}_{}_{}_{}_synthTransactionScoring_transactions_ROS_{}.pkl".format(name, df_type, inclusion_pct, mixin_pct, os_ratio)):
                    print("{} - {} - {} - Transaction based model with Random Oversampling".format(name, df_type, mixin_pct))
                    results = []
                    test_pred = []
                    for sub_train_X, sub_train_y, sub_test_X, sub_test_y in zip(train_X, train_y, test_X, test_y):
                        if name == "IBM-AML":
                            sub_name = pd.concat(
                                [sub_train_X["To Bank"], sub_train_X["From Bank"]]).value_counts().idxmax()
                            synth_sample_size = int(sub_train_X.shape[0] * mixin_pct)
                            if synth_sample_size > train_X_synth.loc[(train_X_synth["To Bank"] != sub_name) & (train_X_synth["From Bank"] != sub_name)].shape[0]:
                                train_X_synth_mixin = train_X_synth.loc[(train_X_synth["To Bank"] != sub_name) & (train_X_synth["From Bank"] != sub_name)].sample(synth_sample_size, random_state=42, replace=True)
                            else:
                                train_X_synth_mixin = train_X_synth.loc[(train_X_synth["To Bank"] != sub_name) & (train_X_synth["From Bank"] != sub_name)].sample(synth_sample_size, random_state=42)
                        elif name == "IBM-CCF":
                            sub_name = sub_train_X["Bank"].value_counts().idxmax()
                            synth_sample_size = int(sub_train_X.shape[0] * mixin_pct)
                            if synth_sample_size > train_X_synth.loc[train_X_synth["Bank"] != sub_name].shape[0]:
                                train_X_synth_mixin = train_X_synth.loc[train_X_synth["Bank"] != sub_name].sample(synth_sample_size, random_state=42, replace=True)
                            else:
                                train_X_synth_mixin = train_X_synth.loc[train_X_synth["Bank"] != sub_name].sample(synth_sample_size, random_state=42)
                        else:
                            raise ValueError("Unknown dataset")
                        train_y_synth_mixin = train_y_synth[train_X_synth_mixin.index]
                        sub_train_X = pd.concat([sub_train_X, train_X_synth_mixin])
                        sub_train_y = pd.concat([sub_train_y, train_y_synth_mixin])

                        ros = RandomOverSampler(sampling_strategy=os_ratio, random_state=42)
                        sub_train_X, sub_train_y = ros.fit_resample(sub_train_X, sub_train_y)
                        # clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose= 1, random_state=42)
                        # clf = GradientBoostingClassifier(n_estimators=50, verbose=1, random_state=42)
                        #clf = XGBClassifier(n_estimators=100, device="cuda", verbosity=2, random_state=42)
                        clf = make_pipeline(StandardScaler(), GridSearchCV(XGBClassifier(random_state=42, device= "cuda"),
                                                                           param_grid={"eta": [0.5, 0.1, 0.05],
                                                                                       "min_child_weight": [0.5, 1, 4],
                                                                                       "max_depth": [3, 6, 9],
                                                                                       "scale_pos_weight": [1, 0.1,
                                                                                                            0.025]},
                                                                           cv=3,
                                                                           refit=True,
                                                                           scoring="roc_auc",
                                                                           n_jobs=-1,
                                                                           verbose=1))
                        clf.fit(sub_train_X, sub_train_y)
                        sub_test_pred = clf.predict(sub_test_X)
                        test_pred.append(pd.DataFrame(sub_test_pred))
                        sub_cls_report = classification_report(sub_test_y, sub_test_pred)
                        sub_cls_report += "roc-auc-score                        {0:.4f}  {0:.4f}".format(
                            roc_auc_score(sub_test_y, sub_test_pred), sub_test_y.shape[0])
                        results.append(sub_cls_report)
                        print(sub_cls_report)

                    test_pred = pd.concat(test_pred)
                    test_real_y = pd.concat(test_y)
                    cls_report = classification_report(test_real_y, test_pred)
                    cls_report += "roc-auc-score                        {0:.4f}  {0:.4f}".format(
                        roc_auc_score(test_real_y, test_pred), test_real_y.shape[0])
                    print(cls_report)

                    with open("../working/partly/{}_{}_{}_{}_synthTransactionScoring_transactions_ROS_{}.pkl".format(name, df_type, inclusion_pct, mixin_pct, os_ratio), "wb") as file:
                        pickle.dump({"overall": cls_report, "per_bank": results,
                                     "type": "transactions over-sampled ({})".format(os_ratio),
                                     "dataset": name, "mixin_percentage": mixin_pct, "included_banks": included_banks, "inclusion_pct": inclusion_pct,
                                     "synth_type": df_type}, file)
