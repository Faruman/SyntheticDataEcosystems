import os.path

import numpy as np
import pandas as pd
import json

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
import pickle

from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="SynthDataGenerator")


if not os.path.exists("../data/IBM-CCF/processed/"):
    os.makedirs("../data/IBM-CCF/processed/")

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
transactions = pd.read_csv("../data/IBM-CCF/archive/credit_card_transactions-ibm_v2.csv")
print(transactions.info())

users = pd.read_csv("../data/IBM-CCF/archive/sd254_users.csv")

# preprocessing
transactions["Amount"]=transactions["Amount"].str.replace("$","").astype(float)

transactions["Hour"] = transactions["Time"].str [0:2].astype('float')
transactions["Minute"] = transactions["Time"].str [3:5].astype('float')
transactions = transactions.drop(['Time'],axis=1)

transactions["Is Fraud?"] = transactions["Is Fraud?"].apply(lambda x: 1 if x == 'Yes' else 0)

transactions["Card ID"] = transactions["User"].astype(str) + "_" + transactions["Card"].astype(str)

transactions["Errors?"]= transactions["Errors?"].fillna("No error")

transactions = transactions.join(users["State"], on = "User", how = "left")
transactions["in Home State?"] = transactions["State"] == transactions["Merchant State"]
transactions = transactions.drop(['State'],axis=1)

transactions["Merchant Size"] = create_attribute_based_on_frequency(transactions["Merchant Name"], 5)

transactions['Timestamp'] = pd.to_datetime(transactions[['Year', 'Month', 'Day']].assign(hour=transactions["Hour"], minute=transactions['Minute']))
transactions['Day of Week'] = transactions['Timestamp'].dt.dayofweek
transactions['Is Online'] = transactions['Merchant City'] == "Online"

# remove city name
#transactions['Merchant Location'] = (transactions['Merchant City'] + ", " + transactions['Merchant State']).apply(lambda x: geolocator.geocode(x) if x != "Online" else None)
#transactions['Merchant Longitude'] = transactions['Merchant Location'].apply(lambda x: x.longitude if x != None else None)
#transactions['Merchant Latitude'] = transactions['Merchant Location'].apply(lambda x: x.latitude if x != None else None)
#transactions = transactions.drop(['Merchant City', 'Merchant Location'],axis=1)
transactions = transactions.drop(['Merchant City'],axis=1)

#transactions['Timestamp'] = transactions['Timestamp'].apply(lambda x: x.value)
#transactions['Timestamp'] = (transactions['Timestamp']-transactions['Timestamp'].min())/(transactions['Timestamp'].max()-transactions['Timestamp'].min())

# create artificial banks, seperating the dataset
## geographical location of the user was chosen as the seperation criteria for the banks, the regions are the foru statistical regions of the US: Northeast, Midwest, South, West

StateToStatRegion_dict = json.loads(open("StateToStatRegion.json").read())

users["StatRegion"] = users["State"].map(StateToStatRegion_dict)
users["Bank"] = users["StatRegion"] + " Bank"

transactions = transactions.join(users["Bank"], on = "User", how = "left")

## remove pacific bank as it is far smaller than the other ones
transactions = transactions[transactions["Bank"] != "Pacific Bank"]

# remove nans
transactions["Merchant State"] = transactions["Merchant State"].fillna("Other")
transactions["Zip"] = transactions["Zip"].fillna(0)

# plot transaction per bank
g = sns.FacetGrid(transactions, col="Is Fraud?", sharey=False)
g.map_dataframe(sns.histplot, x="Bank", multiple="stack")
g.set_xticklabels(rotation=90)
plt.tight_layout()
plt.savefig("../plots/IBM-CCF_Banks.png")
plt.show()

# sort data by timestamp
transactions.sort_values(by="Timestamp", ascending=True)

# save data
transactions.to_pickle("../data/IBM-CCF/processed/IBM-CCF.pkl")

# encode the data
# make target column generic
transactions["target"]= transactions["Is Fraud?"].astype(bool)
transactions = transactions.drop(['Is Fraud?'],axis=1)

# Identify columns with non-numerical data
non_numerical_columns = transactions.select_dtypes(include=['object']).columns

# Initialize a dictionary to hold the LabelEncoders
encoders = {}

# Loop through each non-numerical column and apply LabelEncoder
for col in non_numerical_columns:
    le = LabelEncoder()
    transactions[col] = le.fit_transform(transactions[col])
    encoders[col] = le

# Save the encoders
with open('../data/IBM-CCF/processed/encoders.pkl', 'wb') as file:
    pickle.dump(encoders, file)

# save data
transactions.to_pickle("../data/IBM-CCF/processed/IBM-CCF_encoded.pkl")
print(transactions.info())

# get data statistics
df_stats = transactions["Bank"].value_counts()
df_stats = pd.DataFrame(df_stats).join(transactions.groupby("Bank")["target"].sum(),how = "outer")
df_stats = df_stats.rename(columns={"count":"Count", "target":"Target Count"})
df_stats["Target Ratio"] = df_stats["Target Count"] / df_stats["Count"]
df_stats["Pct of Total"] = df_stats["Count"] / df_stats["Count"].sum()
df_stats.to_excel("../data/IBM-CCF/processed/IBM-CCF_stats.xlsx")