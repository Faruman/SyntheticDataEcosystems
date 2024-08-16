import pandas as pd
import os
from pathlib import Path
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import RandomOverSampler

from xgboost import XGBClassifier


if os.path.exists("./working"):
    os.makedirs("./working")

data = ["IBM-CCF", "IBM-AML"]
os_ratio = 0.2

data_paths = ["../data/IBM-CCF/processed/IBM-CCF_encoded.pkl", "../data/IBM-AML/processed/IBM-AML_encoded.pkl"]

for name, path in zip(data, data_paths):
    df = pd.read_pickle(path)
    df = df.sort_values(by=["Timestamp"])
    df = df.drop(columns=["Timestamp", "Card ID"])

    # reduce dataset for testing
    #df = df.sample(frac=0.05, random_state=42)

    X = df.drop(columns=["target"])
    y = df["target"]

    # split the datasets into banks
    if name == "IBM-AML":
        X["Sender"] = X["Sender"] + X.index.max() + int(X.index.max() * (1 + os_ratio + 0.1))
        X["Receiver"] = X["Receiver"] + X.index.max() + int(X.index.max() * (1 + os_ratio + 0.1))
        train_X, test_X, train_y, test_y = train_test_split(X, y, shuffle=False, test_size=0.3, random_state=42)
        train_bank = [name for name, _ in train_X.groupby("To Bank")]
        train_X_1 = [x for _, x in train_X.groupby("To Bank")]
        train_X_2 = [train_X.loc[train_X["From Bank"] == bank] for bank in train_bank]
        train_X = [pd.concat([d1, d2]).drop_duplicates() for d1, d2 in zip(train_X_1, train_X_2)]
        test_X_1 = [test_X.loc[test_X["To Bank"] == bank] for bank in train_bank]
        test_X_2 = [test_X.loc[test_X["From Bank"] == bank] for bank in train_bank]
        test_X = [pd.concat([d1, d2]).drop_duplicates() for d1, d2 in zip(test_X_1, test_X_2)]
    elif name == "IBM-CCF":
        X["User"] = X["User"] + int(X.index.max() * (1 + os_ratio + 0.1))
        train_X, test_X, train_y, test_y = train_test_split(X, y, shuffle=False, test_size=0.3, random_state=42)
        train_bank = [name for name, _ in train_X.groupby("Bank")]
        train_X = [x for _, x in train_X.groupby("Bank")]
        test_X = [test_X.loc[test_X["Bank"] == bank] for bank in train_bank]
    else:
        raise ValueError("Unknown dataset")

    train_y = [train_y[train_X_sub.index] for train_X_sub in train_X]
    test_y = [test_y[test_X_sub.index] for test_X_sub in test_X]

    skip_transactionModel = False
    skip_transactionModelOS = False

    # create transaction based model
    if not skip_transactionModel and not os.path.isfile("../working/{}_baselineScoring_transactions.pkl".format(name.split("/")[-1])):
        print ("{} - Transaction based model".format(name))
        results = []
        test_pred = []
        for sub_train_X, sub_train_y, sub_test_X, sub_test_y in zip(train_X, train_y, test_X, test_y):
            #clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose= 1, random_state=42)
            #clf = GradientBoostingClassifier(n_estimators=50, verbose=1, random_state=42)
            #clf = XGBClassifier(n_estimators=100, device="cuda", verbosity=2, random_state=42)
            clf = make_pipeline(StandardScaler(), GridSearchCV(XGBClassifier(random_state=42, device= "cuda"),
                                                              param_grid={"eta": [0.5, 0.1, 0.05],
                                                                          "min_child_weight": [0.5, 1, 4],
                                                                          "max_depth": [3, 6, 9],
                                                                          "scale_pos_weight": [1, 0.1, 0.025]},
                                                              cv=3,
                                                              refit=True,
                                                              scoring="roc_auc",
                                                              n_jobs=-1,
                                                              verbose=1))
            clf.fit(sub_train_X, sub_train_y)
            sub_test_pred = clf.predict(sub_test_X)
            test_pred.append(pd.DataFrame(sub_test_pred))
            sub_cls_report = classification_report(sub_test_y, sub_test_pred)
            sub_cls_report += "roc-auc-score                        {0:.4f}  {0:.4f}".format(roc_auc_score(sub_test_y, sub_test_pred), sub_test_y.shape[0])
            results.append(sub_cls_report)
            print(sub_cls_report)

        test_pred = pd.concat(test_pred)
        test_real_y = pd.concat(test_y)
        cls_report = classification_report(test_real_y, test_pred)
        cls_report += "roc-auc-score                        {0:.4f}  {0:.4f}".format(roc_auc_score(test_real_y, test_pred), test_real_y.shape[0])
        print(cls_report)

        with open("../working/{}_baselineScoring_transactions.pkl".format(name.split("/")[-1]), "wb") as file:
            pickle.dump({"overall": cls_report, "per_bank": results, "type": "transactions", "dataset": name.split("/")[-1]}, file)

    # create transaction based model
    if not skip_transactionModelOS and not os.path.isfile("../working/{}_baselineScoring_transactions_ROS_{}.pkl".format(name.split("/")[-1], os_ratio)):
        print("{} - Transaction based model with Random Oversampling".format(name))
        results = []
        test_pred = []
        for sub_train_X, sub_train_y, sub_test_X, sub_test_y in zip(train_X, train_y, test_X, test_y):
            ros = RandomOverSampler(sampling_strategy= os_ratio, random_state=42)
            sub_train_X, sub_train_y = ros.fit_resample(sub_train_X, sub_train_y)
            #clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose= 1, random_state=42)
            #clf = GradientBoostingClassifier(n_estimators=50, verbose=1, random_state=42)
            #clf = XGBClassifier(n_estimators=100, device="cuda", verbosity= 2, random_state=42)
            clf = make_pipeline(StandardScaler(), GridSearchCV(XGBClassifier(random_state=42, device= "cuda"),
                                                               param_grid={"eta": [0.5, 0.1, 0.05],
                                                                           "min_child_weight": [0.5, 1, 4],
                                                                           "max_depth": [3, 6, 9],
                                                                           "scale_pos_weight": [1, 0.1, 0.025]},
                                                               cv=3,
                                                               refit=True,
                                                               scoring="roc_auc",
                                                               n_jobs=-1,
                                                               verbose=1))
            clf.fit(sub_train_X, sub_train_y)
            sub_test_pred = clf.predict(sub_test_X)
            test_pred.append(pd.DataFrame(sub_test_pred))
            sub_cls_report = classification_report(sub_test_y, sub_test_pred)
            sub_cls_report += "roc-auc-score                        {0:.4f}  {0:.4f}".format(roc_auc_score(sub_test_y, sub_test_pred), sub_test_y.shape[0])
            results.append(sub_cls_report)
            print(sub_cls_report)

        test_pred = pd.concat(test_pred)
        test_real_y = pd.concat(test_y)
        cls_report = classification_report(test_real_y, test_pred)
        cls_report += "roc-auc-score                        {0:.4f}  {0:.4f}".format(roc_auc_score(test_real_y, test_pred), test_real_y.shape[0])

        print(cls_report)

        with open("../working/{}_baselineScoring_transactions_ROS_{}.pkl".format(name.split("/")[-1], os_ratio), "wb") as file:
            pickle.dump({"overall": cls_report, "per_bank": results, "type": "transactions over-sampled ({})".format(os_ratio), "dataset": name.split("/")[-1]}, file)