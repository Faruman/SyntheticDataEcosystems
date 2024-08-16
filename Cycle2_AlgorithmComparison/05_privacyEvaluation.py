import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


from modified_sitepackages.privacyBenchmarking import calculate_nnaa, calculate_mir, calculate_air


if not os.path.exists("./working"):
    os.makedirs("./working")

#custom classes
class MyException(Exception):
    pass

class CustomEncoder():
    def __init__(self, encoder=LabelEncoder):
        self.base_encoder = encoder
        self.encoder = {}

    def fit(self, data):
        for column in data.columns:
            self.encoder[column] = self.base_encoder().fit(data[column].values.reshape(-1, 1))

    def transform(self, data):
        for column in data.columns:
            if column in self.encoder.keys():
                if self.encoder[column] == OneHotEncoder:
                    data = pd.concat((pd.DataFrame(self.encoder[column].transform(data[column].values.reshape(-1, 1)).todense(), index= data.index, columns= [x.replace("x0", column) for x in self.encoder[column].get_feature_names_out()]), data), axis= 1)
                    data = data.drop(columns=[column])
                else:
                    data[column] = self.encoder[column].transform(data[column].values.reshape(-1, 1))
            else:
                raise MyException("fit first with all necessary columns")
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


# variables
methods = ["GMM", "CTGAN", "TVAE", "TIMEGAN"]

# load data
real_data = pd.read_pickle('./data/IEEE-CIS/processed/data_reduced.pkl')
real_data = real_data.drop(columns=['IdentityID', 'TransactionID', 'purchaseremaildomain', 'recipientemaildomain'])
operators = real_data['operator'].unique()
data_grouped = real_data.groupby('operator')

# ensure real data exists in the right format
for operator, data_group in data_grouped:
    data_group = data_group.drop(columns=["operator"])
    data_group_train, data_group_test = train_test_split(data_group, train_size=0.7, random_state=42)
    data_group_train.to_pickle("./working/real_{}_train_unbalanced.pkl".format(operator))
    data_group_test.to_pickle("./working/real_{}_test_unbalanced.pkl".format(operator))

#define metrics
metrics = {
    "mia_risk"  : {"num_eval_iter": 5},
    "att_discl" : {}
}

privacy_metrics = {}

for method in methods:
    print("Processing method: {}".format(method))

    synth_train = pd.DataFrame()
    real_train = pd.DataFrame()
    test_data = pd.DataFrame()

    for operator in operators:
        test_data = pd.concat((test_data, pd.read_pickle("./working/real_{}_test_unbalanced.pkl".format(operator))), axis= 0, ignore_index= True)
        real_train = pd.concat((real_train, pd.read_pickle("./working/real_{}_train_unbalanced.pkl".format(operator))), axis= 0, ignore_index= True)
        synth_train = pd.concat((synth_train, pd.read_pickle("./synth/synth_{}_{}_unbalanced.pkl".format(method, operator))), axis= 0, ignore_index= True)

    if "TransactionID" in test_data.columns:
        test_data = test_data.drop("TransactionID", axis= 1)
    if "TransactionID" in real_train.columns:
        real_train = real_train.drop("TransactionID", axis= 1)
    if "TransactionID" in synth_train.columns:
        synth_train = synth_train.drop("TransactionID", axis= 1)
    if "IdentityID" in test_data.columns:
        test_data = test_data.drop("IdentityID", axis= 1)
    if "IdentityID" in real_train.columns:
        real_train = real_train.drop("IdentityID", axis= 1)
    if "IdentityID" in synth_train.columns:
        synth_train = synth_train.drop("IdentityID", axis= 1)

    test_data = test_data.sample(40000)
    real_train = real_train.sample(400000)
    synth_train = synth_train.sample(400000, random_state= 42)

    le = CustomEncoder()
    category_columns = ["productcategory", "cardtype", "cardcountry", "billingcountry"]
    le.fit(pd.concat((real_train[category_columns], test_data[category_columns], synth_train[category_columns]), axis= 0))

    test_data[category_columns] = le.transform(test_data[category_columns]).astype(float)
    real_train[category_columns] = le.transform(real_train[category_columns]).astype(float)
    synth_train[category_columns] = le.transform(synth_train[category_columns]).astype(float)

    test_data = test_data.fillna(test_data.mean())
    real_train = real_train.fillna(real_train.mean())
    synth_train = synth_train.fillna(synth_train.mean())

    nnaa = calculate_nnaa(train= real_train, test= test_data, fake= synth_train, cont_cols= list(set(real_train.columns) - set(category_columns)), batchsize= 5000)
    mir = calculate_mir(train= real_train, test= test_data, fake= synth_train, cont_cols= list(set(real_train.columns) - set(category_columns)), batchsize= 5000, performance_metric= "precision")
    #real_train = real_train.sample(0000, random_state= 42)
    #sensitive_cols = ['productcategory', 'card1', 'card2', 'cardcountry', 'card5', 'cardtype', 'billingcountry']
    #air = calculate_air(train= real_train, test= test_data, fake= synth_train, cont_cols= list(set(real_train.columns) - set(category_columns)), sensitive_cols= sensitive_cols, x= 0, y= 8, batchsize=5000)

    privacy_metrics[method] = {"NNAA Risk": nnaa, "Membership Inference Risk": mir}
    # privacy_metrics[method] = {"NNAA Risk": nnaa, "Membership Inference Risk": mir, "Attribute Inference Risk": air}

    pd.DataFrame(privacy_metrics).to_excel("./results/privacy.xlsx")

