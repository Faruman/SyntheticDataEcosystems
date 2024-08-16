import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from prince import FAMD, PCA, MCA

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
    identities.to_pickle("{}/processed/identities.pkl".format(path))
else:
    identities = pd.read_pickle("{}/processed/identities.pkl".format(path))

# merge transactions, identity and uid data
identities = identities[["IdentityID", "TransactionID"]]
df = pd.merge(transactions, identities, on='TransactionID', how='left')

# filter data
df = df[df['isFraud'].notna()]
df = df[df['card4'].notna()]

# clean data
df = df.fillna(0)
df = df.replace([np.inf, -np.inf], 0)
df = df.replace({"": 0, "F": -1, "T": 1})
df = df.rename(columns={"TransactionDT": "datetime",
                        "TransactionAmt": "amount",
                        "ProductCD": "productcategory",
                        "card4": "operator",
                        "card3": "cardcountry",
                        "card6": "cardtype",
                        "addr1": "billingregion",
                        "addr2": "billingcountry",
                        "P_emaildomain": "purchaseremaildomain",
                        "R_emaildomain": "recipientemaildomain",
                        "dist1": "E1",
                        "dist2": "E2"})
df["isFraud"] = df["isFraud"] == 1
df["isFraud"] = df["isFraud"].astype(bool)
df = df.drop(columns= ["billingregion"])
df["cardcountry"] = df["cardcountry"].apply(lambda x: 0 if x == 150 else 1 if x == 185 else 2)
df["billingcountry"] = df["billingcountry"].apply(lambda x: 0 if x == 87 else 1 if x == 0 else 2)

for column, dtype in zip(df.columns, df.dtypes):
    if dtype == "object":
        df[column] = df[column].replace(np.nan, "")
        df[column] = df[column].replace(0, "")
        df[column] = df[column].astype("category")

#remove columns with same value:
df = df.loc[:, df.apply(pd.Series.nunique) > 1]

#remove columns with more then 90% empty values
df = df.loc[:, (df == "").mean() < 0.9]

# print summary statistics
print("Pandas Info:")
print(df.info())
print("Pandas Describe:")
print(df.describe())

# save data
df.to_pickle("{}/processed/data.pkl".format(path))

# remove nans
df = df.dropna()

# combine masked features to retain 90% variance
transaction_obscured_columns = ["C{}".format(x+1) for x in range(14)] + ["D{}".format(x+1) for x in range(15)] + ["V{}".format(x+1) for x in range(339)] +["M{}".format(x+1) for x in range(9)]
transaction_category_columns = ["ProductCD", "billingregion", "billingcountry", "P_emaildomain", "R_emaildomain"] + ["card{}".format(x+1) for x in range(6)] + ["M{}".format(x+1) for x in range(9)]
for column in transaction_obscured_columns:
    if column in transaction_category_columns:
        df[column] = df[column].astype("category")
    else:
        df[column] = df[column].astype(float)
df_obscured_transactions = df[transaction_obscured_columns]

famd_retained_variance = 0
i = 57
while famd_retained_variance < 0.9:
    famd = FAMD(n_components=i)
    famd.fit(df_obscured_transactions.sample(100000))
    famd_retained_variance = float(famd.eigenvalues_summary["% of variance (cumulative)"].iloc[-1].strip("%"))/100
    print("TRS Variance Ratio ({}): {}".format(i, famd_retained_variance))
    i += 1
famd_features = famd.transform(df_obscured_transactions)
df = df.drop(transaction_obscured_columns, axis=1)
df[['TRS_{}'.format(i) for i in range(famd_features.shape[1])]] = pd.DataFrame(famd_features, index=df.index)

# remove rows with nans
df = df.dropna()

# transform categories to sdtype: category
for column in df.columns:
    if df[column].dtype == "category":
        df[column] = df[column].astype(str)

# print summary statistics
print("Pandas Info:")
print(df.info())
print("Pandas Describe:")
print(df.describe())

# save data
df.to_pickle("{}/processed/data_reduced.pkl".format(path))