import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

path = "data/IEEE-CIS"

if not os.path.exists("{}/processed".format(path)):
    os.makedirs("{}/processed".format(path))

if not os.path.exists("{}/processed/transactions.pkl".format(path)):
    transactions = pd.read_csv("{}/ieee-fraud-detection/train_transaction.csv".format(path))
    transactions.to_pickle("{}/processed/transactions.pkl".format(path))
else:
    transactions = pd.read_pickle("{}/processed/transactions.pkl".format(path))

if not os.path.exists("{}/processed/identities.pkl".format(path)):
    identities = pd.read_csv("{}/ieee-fraud-detection/train_identity.csv".format(path))
    identities = identities.reset_index().rename(columns={"index": "IdentityID"})
    identities = identities.drop(columns=["DeviceInfo"])
    identities.to_pickle("{}/processed/identities.pkl".format(path))
else:
    identities = pd.read_pickle("{}/processed/identities.pkl".format(path))

# merge transactions, identity and uid data
df = pd.merge(transactions, identities, on='TransactionID', how='left')

for column in df.columns:
    if df[column].dtype == "object":
        df[column] = df[column].replace(np.nan, "")
        df[column] = df[column].replace(0, "")
        df[column] = LabelEncoder().fit_transform(df[column])
    else:
        df[column] = df[column].fillna(df[column].mean())

rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(df.drop(columns=["isFraud"]), df["isFraud"])

print(rfc.feature_importances_)

importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)

forest_importances = pd.Series(importances, index=list(set(df.columns) - {"isFraud"}))

print(forest_importances.sort_values(ascending= False))

forest_importances = forest_importances.to_frame().reset_index().rename(columns= {"index": "Feature", 0: "Importance"})
forest_importances["Group"] = forest_importances["Feature"].apply(lambda feature: "Transaction" if feature in transactions.columns else "Identity")

print(forest_importances.drop(columns= ["Feature"]).groupby("Group").sum())