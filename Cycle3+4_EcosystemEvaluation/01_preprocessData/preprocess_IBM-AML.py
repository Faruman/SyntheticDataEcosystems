import os

import numpy as np
import pandas as pd
import json

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
import pickle


if not os.path.exists("../data/IBM-AML/processed/"):
    os.makedirs("../data/IBM-AML/processed/")

#support
def split_array_with_variable_splits(arr, num_splits):
    # Ensure the number of splits does not exceed the number of elements in the array
    num_splits = min(num_splits, len(arr))

    # Initialize subarrays and indices arrays with variable length
    subarrays = [[] for _ in range(num_splits)]
    indices = [[] for _ in range(num_splits)]
    current_sum = [0] * num_splits
    target_sum = sum(arr) / num_splits
    current_index = 0

    for i, num in enumerate(arr):
        # Distribute at least one element to each subarray first
        if i < num_splits:
            current_index = i
        else:
            # Once each subarray has at least one element, check if we should move to the next subarray
            if current_sum[current_index] + num > target_sum and current_index < num_splits - 1:
                current_index += 1

        # Add the number to the current subarray and its index to the indices array
        subarrays[current_index].append(num)
        indices[current_index].append(i)
        current_sum[current_index] += num

    return subarrays, indices

def create_attribute_based_on_frequency(df_column, num_splits= 3):
    column_values = df_column.value_counts()
    column_values_counts = column_values.value_counts().sort_index()
    _, indices = split_array_with_variable_splits(column_values_counts.tolist(), num_splits)
    max_indices = [max(index) for index in indices]
    max_values = [column_values_counts.index[index] for index in max_indices]
    column_values = column_values.reset_index()
    column_values["frequency group"] = column_values.iloc[:, -1].apply(lambda x: next((i for i, max_val in enumerate(max_values) if x <= max_val), len(max_values)))
    freq_group_dict = column_values.set_index(column_values.columns[0]).to_dict()["frequency group"]
    return df_column.map(freq_group_dict)


# load data
transactions = pd.read_csv("../data/IBM-AML/archive/HI-Medium_Trans.csv")
print(transactions.info())

# preprocessing
transactions['Timestamp'] = pd.to_datetime(transactions['Timestamp'])
transactions['Year'] = transactions['Timestamp'].dt.year
transactions['Month'] = transactions['Timestamp'].dt.month
transactions['Day'] = transactions['Timestamp'].dt.day
transactions['Hours'] = transactions['Timestamp'].dt.hour
transactions['Minutes'] = transactions['Timestamp'].dt.minute
#transactions['Timestamp'] = transactions['Timestamp'].apply(lambda x: x.value)
#transactions['Timestamp'] = (transactions['Timestamp']-transactions['Timestamp'].min())/(transactions['Timestamp'].max()-transactions['Timestamp'].min())

transactions['Sender'] = transactions['From Bank'].astype(str) + '_' + transactions['Account']
transactions['Receiver'] = transactions['To Bank'].astype(str) + '_' + transactions['Account.1']
transactions = transactions.drop(['Account', 'Account.1'], axis=1)

transactions["To Branch"] = transactions["To Bank"]
transactions["From Branch"] = transactions["From Bank"]
transactions = transactions.drop(['To Bank', 'From Bank'], axis=1)

transactions["Currency Conversion"] = transactions["Payment Currency"] != transactions["Receiving Currency"]

transactions["Sender Size"] = create_attribute_based_on_frequency(transactions["Sender"], 5)
transactions["Receiver Size"] = create_attribute_based_on_frequency(transactions["Receiver"], 5)

transactions['Day of Week'] = transactions['Timestamp'].dt.dayofweek

## removed due to information leakage concerns
# get number and amount of transactions per receiver
#currencyConversion_dict = json.loads(open("currencyConversion.json").read())
#receiver_df = transactions.groupby(["Receiver", "Receiving Currency"])["Amount Received"].sum().reset_index().pivot(index='Receiver', columns='Receiving Currency').fillna(0)
#receiver_df.columns = receiver_df.columns.droplevel()
#for column in receiver_df.columns:
#    receiver_df[column] = receiver_df[column] * currencyConversion_dict[column]
#receiver_df = receiver_df.sum(axis=1).reset_index()
#receiver_df = receiver_df.merge(transactions.groupby("Receiver")["Timestamp"].count().reset_index(), on="Receiver", how= "outer")
#receiver_df = receiver_df.rename(columns={"Timestamp": "Count Received", 0: "Total Amount Received"})
#sender_df = transactions.groupby(["Sender", "Payment Currency"])["Amount Paid"].sum().reset_index().pivot(index='Sender', columns='Payment Currency').fillna(0)
#sender_df.columns = sender_df.columns.droplevel()
#for column in sender_df.columns:
#    sender_df[column] = sender_df[column] * currencyConversion_dict[column]
#sender_df = sender_df.sum(axis=1).reset_index()
#sender_df = sender_df.merge(transactions.groupby("Sender")["Timestamp"].count().reset_index(), on= "Sender", how= "outer")
#sender_df = sender_df.rename(columns={"Timestamp": "Count Paid", 0: "Total Amount Paid"})
#transactions = transactions.merge(receiver_df, on="Receiver", how="left").merge(sender_df, on="Sender", how="left")

# create artificial banks (by merging the existing ones), seperating the dataset
def get_most_frequent_currency(currencies):
    currency = currencies.value_counts().idxmax()
    return currency
currencyPerBank = transactions.groupby("To Branch")["Payment Currency"].apply(get_most_frequent_currency).reset_index()
currencyToSuperBank_dict = json.loads(open("currencyToBank.json").read())
currencyPerBank["Bank"] = currencyPerBank["Payment Currency"].map(currencyToSuperBank_dict)
BankToSuperBank_dict = currencyPerBank.loc[:,["To Branch","Bank"]].set_index("To Branch").to_dict()["Bank"]
transactions["To Bank"] = transactions["To Branch"].map(BankToSuperBank_dict)
transactions["From Bank"] = transactions["From Branch"].map(BankToSuperBank_dict)

# remove nans
transactions["From Bank"] = transactions["From Bank"].fillna("Unknown")
transactions["To Bank"] = transactions["To Bank"].fillna("Unknown")
transactions = transactions.loc[~((transactions["From Bank"] == "Unknown") | (transactions["To Bank"] == "Unknown"))]

# plot transaction per bank
g = sns.FacetGrid(transactions, col="Is Laundering", sharey=False)
g.map_dataframe(sns.histplot, x="To Bank", multiple="stack")
g.set_xticklabels(rotation=90)
plt.tight_layout()
plt.savefig("../plots/IBM-AML_ToBanks.png")
plt.show()

g = sns.FacetGrid(transactions, col="Is Laundering", sharey=False)
g.map_dataframe(sns.histplot, x="From Bank", multiple="stack")
g.set_xticklabels(rotation=90)
plt.tight_layout()
plt.savefig("../plots/IBM-AML_FromBanks.png")
plt.show()

# plot interbank and intrabank laundering
transactions["Intrabank Transfer"] = transactions["To Bank"] == transactions["From Bank"]
transactions["Intrabank Transfer"] = transactions["Intrabank Transfer"].astype(bool)
g = sns.FacetGrid(transactions, col="Is Laundering", sharey=False)
g.map_dataframe(sns.histplot, x="Intrabank Transfer", multiple="stack")
g.set_xticklabels(rotation=90)
plt.tight_layout()
plt.savefig("../plots/IBM-AML_IntrabankLaunderings.png")
plt.show()

# sort data by timestamp
transactions = transactions.sort_values(by="Timestamp", ascending=True)

# save data
transactions.to_pickle("../data/IBM-AML/processed/IBM-AML.pkl")

# encode the data
# make target column generic
transactions["target"]= transactions["Is Laundering"].astype(bool)
transactions = transactions.drop(['Is Laundering'],axis=1)

# Identify columns with non-numerical data
non_numerical_columns = transactions.select_dtypes(include=['object']).columns

# Initialize a dictionary to hold the LabelEncoders
encoders = {}

# Loop through each non-numerical column and apply LabelEncoder
for col in non_numerical_columns:
    if col not in ["From Bank", "To Bank"]:
        le = LabelEncoder()
        transactions[col] = le.fit_transform(transactions[col])
        encoders[col] = le
    else:
        if not "Bank" in encoders:
            le = LabelEncoder()
            le.fit(pd.concat([transactions["From Bank"], transactions["To Bank"]]))
            encoders["Bank"] = le
        transactions[col] = encoders["Bank"].transform(transactions[col])

# Save the encoders
with open('../data/IBM-AML/processed/encoders.pkl', 'wb') as file:
    pickle.dump(encoders, file)

# save data
transactions.to_pickle("../data/IBM-AML/processed/IBM-AML_encoded.pkl")
print(transactions.info())

# get data statistics
df_stats = pd.concat([transactions["To Bank"], transactions.loc[transactions["To Bank"] != transactions["From Bank"], "From Bank"]]).value_counts()
df_stats = pd.DataFrame(df_stats).join(pd.DataFrame(pd.DataFrame(transactions.groupby("To Bank")["target"].sum()).join(transactions.loc[transactions["To Bank"] != transactions["From Bank"]].groupby("From Bank")["target"].sum(), lsuffix=" to", rsuffix=" from", how="outer").sum(axis=1)), lsuffix= "Count", rsuffix="Target", how="outer")
df_stats = df_stats.fillna(0).rename(columns={"count": "Count", 0: "Target Count"})
df_stats["Target Ratio"] = df_stats["Target Count"] / df_stats["Count"]
df_stats["Pct of Total"] = df_stats["Count"] / df_stats["Count"].sum()
df_stats.to_excel("../data/IBM-AML/processed/IBM-AML_stats.xlsx")