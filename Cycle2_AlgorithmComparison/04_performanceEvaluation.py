import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc
import pickle
from tqdm import tqdm, trange
import json

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
                data = pd.concat((pd.DataFrame(self.encoder[column].transform(data[column].values.reshape(-1, 1)).todense(), index= data.index, columns= [x.replace("x0", column) for x in self.encoder[column].get_feature_names_out()]), data), axis= 1)
                data = data.drop(columns= [column])
            else:
                raise MyException("fit first with all necessary columns")
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

if not os.path.exists("./model/rf"):
    os.makedirs("./model/rf")
if not os.path.exists("./results"):
    os.makedirs("./results")
if not os.path.exists("./working"):
    os.makedirs("./working")

# variables
methods = ["GMM", "CTGAN", "TVAE", "TIMEGAN"]

# load data
real_data = pd.read_pickle('./data/IEEE-CIS/processed/data_reduced.pkl')
real_data = real_data.drop(columns=['IdentityID', 'TransactionID', 'purchaseremaildomain', 'recipientemaildomain'])
data_grouped = real_data.groupby('operator')

# ensure real data exists in the right format
for operator, data_group in data_grouped:
    data_group = data_group.drop(columns=["operator"])
    
    data_group_train, data_group_test = train_test_split(data_group, train_size=0.7, random_state=42)

    data_group_train.to_pickle("./working/real_{}_train_unbalanced.pkl".format(operator))
    data_group_test.to_pickle("./working/real_{}_test_unbalanced.pkl".format(operator))


performance_metrics = {}

le = CustomEncoder(encoder= OneHotEncoder)
category_columns = [column for column, dtype in zip(real_data.columns, real_data.dtypes) if (dtype == "category" or dtype == "object") and column != "operator"]
le.fit(real_data[category_columns])
pickle.dump(le, open("./working/labelEncoder.pkl", 'wb'))

performance_per_operator = dict(zip(list(real_data["operator"].unique()), [(0,0,"") for i in range(len(real_data["operator"].unique()))]))

for method in methods:
    print("Processing method: {}".format(method))

    method_train = pd.DataFrame()

    for operator in real_data["operator"].unique():
        print("Processing operator: {}".format(operator))

        performance_metrics[operator + "_real"] = {}

        # load test data
        test_data = pd.read_pickle("./working/real_{}_test_unbalanced.pkl".format(operator))
        test_data = pd.concat((test_data, le.transform(test_data[category_columns])), axis= 1)
        test_data = test_data.drop(columns= category_columns)

        #performance real data
        real_train = pd.read_pickle("./working/real_{}_train_unbalanced.pkl".format(operator))
        print("Real data fraud rate: {}".format(real_train["isFraud"].mean()))
        real_train = pd.concat((real_train, le.transform(real_train[category_columns])), axis= 1)
        real_train = real_train.drop(columns= category_columns)

        if os.path.exists("./model/rf/real_{}_rf.pkl".format(operator)):
            rf = pickle.load(open("./model/rf/real_{}_rf.pkl".format(operator), 'rb'))
        else:
            #rf = RandomForestClassifier(random_state=42, verbose=2, n_jobs=-1)
            #rf = make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=42, verbose=2))
            rf = make_pipeline(StandardScaler(), GridSearchCV(XGBClassifier(random_state=42),
                                                              param_grid={"eta": [0.5, 0.1, 0.05],
                                                                          "min_child_weight": [0.5, 1, 4],
                                                                          "max_depth": [3, 6, 9],
                                                                          "scale_pos_weight": [1, 0.1, 0.025]},
                                                              cv=5,
                                                              refit=True,
                                                              scoring="roc_auc",
                                                              n_jobs=-1,
                                                              verbose= 1))
            rf.fit(real_train.drop(columns=["isFraud"]), real_train["isFraud"])
            pickle.dump(rf, open("./model/rf/real_{}_rf.pkl".format(operator), 'wb'))

        test_preds = rf.predict(test_data.drop(columns=["isFraud"]))
        performance_metrics[operator + "_real"]["accuracy"] = accuracy_score(test_preds, test_data["isFraud"])
        performance_metrics[operator + "_real"]["f1score"] = f1_score(test_preds, test_data["isFraud"])
        performance_metrics[operator + "_real"]["rocaucscore"] = roc_auc_score(test_preds, test_data["isFraud"])
        precision, recall, thresholds = precision_recall_curve(test_data["isFraud"], test_preds)
        performance_metrics[operator + "_real"]["praucscore"] = auc(recall, precision)
        performance_metrics[operator + "_real"]["precision"] = precision_score(test_preds, test_data["isFraud"])
        performance_metrics[operator + "_real"]["recall"] = recall_score(test_preds, test_data["isFraud"])
        print("Real data Performance: {}".format(performance_metrics[operator + "_real"]))

        pd.DataFrame(performance_metrics).to_excel("./results/performance.xlsx")

        performance_metrics[operator + "_" + method] = {}

        synth_train = pd.read_pickle("./synth/synth_{}_{}_unbalanced.pkl".format(method, operator))
        if "TransactionID" in synth_train.columns:
            synth_train = synth_train.drop(columns= ["TransactionID"])
        if "IdentityID" in synth_train.columns:
            synth_train = synth_train.drop(columns= ["IdentityID"])
        if "operator" in synth_train.columns:
            synth_train = synth_train.drop(columns=["operator"])
        print("Synth data ({}, {}) fraud rate: {}".format(method, operator, synth_train["isFraud"].mean()))

        synth_train = pd.concat((synth_train, le.transform(synth_train[category_columns])), axis= 1)
        synth_train = synth_train.drop(columns= category_columns)

        synth_train = synth_train.fillna(synth_train.mean())

        method_train = pd.concat([method_train, synth_train.sample(real_train.shape[0], random_state= 42, replace= True)], ignore_index=True)

        if os.path.exists("./model/rf/synth_{}_{}_rf.pkl".format(method, operator)):
            rf = pickle.load(open("./model/rf/synth_{}_{}_rf.pkl".format(method, operator), 'rb'))
        else:
            #rf = RandomForestClassifier(random_state=42, verbose=2, n_jobs=-1)
            #rf = make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=42, verbose=2))
            rf = make_pipeline(StandardScaler(), GridSearchCV(XGBClassifier(random_state=42),
                                                              param_grid={"eta": [0.5, 0.1, 0.05],
                                                                          "min_child_weight": [0.5, 1, 4],
                                                                          "max_depth": [3, 6, 9],
                                                                          "scale_pos_weight": [1, 0.1, 0.025]},
                                                              cv=5,
                                                              refit=True,
                                                              scoring="roc_auc",
                                                              n_jobs=-1,
                                                              verbose=1))
            rf.fit(synth_train.drop(columns=["isFraud"]), synth_train["isFraud"].astype(int))
            pickle.dump(rf, open("./model/rf/synth_{}_{}_rf.pkl".format(method, operator), 'wb'))

        test_preds = rf.predict(test_data.loc[:, synth_train.columns].drop(columns=["isFraud"]))
        performance_metrics[operator + "_" + method]["accuracy"] = accuracy_score(test_preds, test_data["isFraud"])
        performance_metrics[operator + "_" + method]["f1score"] = f1_score(test_preds, test_data["isFraud"])
        try:
            performance_metrics[operator + "_" + method]["rocaucscore"] = roc_auc_score(test_preds, test_data["isFraud"])
        except:
            performance_metrics[operator + "_" + method]["rocaucscore"] = 0
        try:
            precision, recall, thresholds = precision_recall_curve(test_data["isFraud"], test_preds)
            performance_metrics[operator + "_" + method]["praucscore"] = auc(recall, precision)
        except:
            performance_metrics[operator + "_" + method]["praucscore"] = 0

        performance_metrics[operator + "_" + method]["precision"] = precision_score(test_preds, test_data["isFraud"])
        performance_metrics[operator + "_" + method]["recall"] = recall_score(test_preds, test_data["isFraud"])
        print("Synth data ({}, {}) Performance: {}".format(method, operator, performance_metrics[operator + "_" + method]))
        del synth_train

        pd.DataFrame(performance_metrics).to_excel("./results/performance.xlsx")

        #if performance_per_operator[operator][0] < performance_metrics[operator + "_" + method]["rocaucscore"]:
        #    performance_per_operator[operator] = (performance_metrics[operator + "_" + method]["rocaucscore"], real_train.shape[0], method)
        if performance_per_operator[operator][0] < performance_metrics[operator + "_" + method]["rocaucscore"]:
            performance_per_operator[operator] = (performance_metrics[operator + "_" + method]["rocaucscore"], real_train.shape[0], method)



    for operator in real_data["operator"].unique():
        print("Processing operator: {}".format(operator))

        operator = "comb" + operator

        # load test data
        test_data = pd.read_pickle("./working/real_{}_test_unbalanced.pkl".format(operator.replace("comb", "")))
        test_data = pd.concat((test_data, le.transform(test_data[category_columns])), axis= 1)
        test_data = test_data.drop(columns= category_columns)

        performance_metrics[operator + "_" + method] = {}

        if os.path.exists("./model/rf/synth_{}_{}_rf.pkl".format(method, operator)):
            rf = pickle.load(open("./model/rf/synth_{}_{}_rf.pkl".format(method, operator), 'rb'))
        else:
            #rf = RandomForestClassifier(random_state=42, verbose=2, n_jobs=-1)
            #rf = make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=42, verbose=2))
            rf = make_pipeline(StandardScaler(), GridSearchCV(XGBClassifier(random_state=42),
                                                              param_grid={"eta": [0.5, 0.1, 0.05],
                                                                          "min_child_weight": [0.5, 1, 4],
                                                                          "max_depth": [3, 6, 9],
                                                                          "scale_pos_weight": [1, 0.1, 0.025]},
                                                              cv=5,
                                                              refit=True,
                                                              scoring="roc_auc",
                                                              n_jobs=-1,
                                                              verbose=1))
            rf.fit(method_train.drop(columns=["isFraud"]), method_train["isFraud"].astype(int))
            pickle.dump(rf, open("./model/rf/synth_{}_{}_rf.pkl".format(method, operator), 'wb'))

        test_preds = rf.predict(test_data.loc[:, method_train.columns].drop(columns=["isFraud"]))
        performance_metrics[operator + "_" + method]["accuracy"] = accuracy_score(test_preds, test_data["isFraud"])
        performance_metrics[operator + "_" + method]["f1score"] = f1_score(test_preds, test_data["isFraud"])
        try:
            performance_metrics[operator + "_" + method]["rocaucscore"] = roc_auc_score(test_preds, test_data["isFraud"])
        except:
            performance_metrics[operator + "_" + method]["rocaucscore"] = 0
        try:
            precision, recall, thresholds = precision_recall_curve(test_data["isFraud"], test_preds)
            performance_metrics[operator + "_" + method]["praucscore"] = auc(recall, precision)
        except:
            performance_metrics[operator + "_" + method]["praucscore"] = 0
        performance_metrics[operator + "_" + method]["precision"] = precision_score(test_preds, test_data["isFraud"])
        performance_metrics[operator + "_" + method]["recall"] = recall_score(test_preds, test_data["isFraud"])
        print("Synth data ({}, {}) Performance: {}".format(method, operator, performance_metrics[operator + "_" + method]))

        pd.DataFrame(performance_metrics).to_excel("./results/performance.xlsx")

best_mixins = {}
for operator in real_data["operator"].unique():
    print("Processing operator: {}".format(operator))

    # load test data
    test_data = pd.read_pickle("./working/real_{}_test_unbalanced.pkl".format(operator))
    test_data = pd.concat((test_data, le.transform(test_data[category_columns])), axis= 1)
    test_data = test_data.drop(columns= category_columns)

    # load synth data
    best_method = pd.DataFrame()
    for temp_operator in performance_per_operator.keys():
        temp = pd.read_pickle("./synth/synth_{}_{}_unbalanced.pkl".format(performance_per_operator[temp_operator][2], temp_operator)).sample(performance_per_operator[temp_operator][1], random_state= 42, replace= True)
        temp["operator"] = temp_operator
        best_method = pd.concat([best_method, temp], ignore_index=True)
        if "TransactionID" in best_method.columns:
            best_method = best_method.drop(columns= ["TransactionID"])
        if "IdentityID" in best_method.columns:
            best_method = best_method.drop(columns= ["IdentityID"])
    best_method = pd.concat((best_method, le.transform(best_method[category_columns])), axis= 1)
    best_method = best_method.drop(columns= category_columns)

    method = "best"
    operator = "comb" + operator
    performance_metrics[operator + "_" + method] = {}

    if os.path.exists("./model/rf/synth_{}_{}_rf.pkl".format(method, operator)):
        rf = pickle.load(open("./model/rf/synth_{}_{}_rf.pkl".format(method, operator), 'rb'))
    else:
        #rf = RandomForestClassifier(random_state=42, verbose=2, n_jobs=-1)
        #rf = make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=42, verbose=2))
        rf = make_pipeline(StandardScaler(), GridSearchCV(XGBClassifier(random_state=42),
                                                          param_grid={"eta": [0.5, 0.1, 0.05],
                                                                      "min_child_weight": [0.5, 1, 4],
                                                                      "max_depth": [3, 6, 9],
                                                                      "scale_pos_weight": [1, 0.1, 0.025]},
                                                          cv=5,
                                                          refit=True,
                                                          scoring="roc_auc",
                                                          n_jobs=-1,
                                                          verbose=1))
        rf.fit(best_method.drop(columns=["isFraud", "operator"]), best_method["isFraud"].astype(int))
        pickle.dump(rf, open("./model/rf/synth_{}_{}_rf.pkl".format(method, operator), 'wb'))

    test_preds = rf.predict(test_data.drop(columns=["isFraud"]))
    performance_metrics[operator + "_" + method]["accuracy"] = accuracy_score(test_preds, test_data["isFraud"])
    performance_metrics[operator + "_" + method]["f1score"] = f1_score(test_preds, test_data["isFraud"])
    performance_metrics[operator + "_" + method]["rocaucscore"] = roc_auc_score(test_preds, test_data["isFraud"])
    precision, recall, thresholds = precision_recall_curve(test_data["isFraud"], test_preds)
    performance_metrics[operator + "_" + method]["praucscore"] = auc(recall, precision)
    performance_metrics[operator + "_" + method]["precision"] = precision_score(test_preds, test_data["isFraud"])
    performance_metrics[operator + "_" + method]["recall"] = recall_score(test_preds, test_data["isFraud"])
    print("Synth data ({}, {}) Performance: {}".format(method, operator, performance_metrics[operator + "_" + method]))

    pd.DataFrame(performance_metrics).to_excel("./results/performance.xlsx")

    real_train = pd.read_pickle("./working/real_{}_train_unbalanced.pkl".format(operator.replace("comb", "")))
    real_train = pd.concat((real_train, le.transform(real_train[category_columns])), axis= 1)
    real_train = real_train.drop(columns= category_columns)


    method = "best-1"
    operator = operator.replace("comb", "comb+real")
    performance_metrics[operator + "_" + method] = {}

    comb_method = pd.concat([best_method, real_train], ignore_index=True)
    if os.path.exists("./model/rf/synth_{}_{}_rf.pkl".format(method, operator)):
        rf = pickle.load(open("./model/rf/synth_{}_{}_rf.pkl".format(method, operator), 'rb'))
    else:
        #rf = RandomForestClassifier(random_state=42, verbose=2, n_jobs=-1)
        #rf = make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=42, verbose=2))
        rf = make_pipeline(StandardScaler(), GridSearchCV(XGBClassifier(random_state=42),
                                                          param_grid={"eta": [0.5, 0.1, 0.05],
                                                                      "min_child_weight": [0.5, 1, 4],
                                                                      "max_depth": [3, 6, 9],
                                                                      "scale_pos_weight": [1, 0.1, 0.025]},
                                                          cv=5,
                                                          refit=True,
                                                          scoring="roc_auc",
                                                          n_jobs=-1,
                                                          verbose=1))
        rf.fit(comb_method.drop(columns=["isFraud", "operator"]), comb_method["isFraud"].astype(int))
        pickle.dump(rf, open("./model/rf/synth_{}_{}_rf.pkl".format(method, operator), 'wb'))

    test_preds = rf.predict(test_data.drop(columns=["isFraud"]))
    performance_metrics[operator + "_" + method]["accuracy"] = accuracy_score(test_preds, test_data["isFraud"])
    performance_metrics[operator + "_" + method]["f1score"] = f1_score(test_preds, test_data["isFraud"])
    performance_metrics[operator + "_" + method]["rocaucscore"] = roc_auc_score(test_preds, test_data["isFraud"])
    precision, recall, thresholds = precision_recall_curve(test_data["isFraud"], test_preds)
    performance_metrics[operator + "_" + method]["praucscore"] = auc(recall, precision)
    performance_metrics[operator + "_" + method]["precision"] = precision_score(test_preds, test_data["isFraud"])
    performance_metrics[operator + "_" + method]["recall"] = recall_score(test_preds, test_data["isFraud"])
    print("Synth data ({}, {}) Performance: {}".format(method, operator, performance_metrics[operator + "_" + method]))

    pd.DataFrame(performance_metrics).to_excel("./results/performance.xlsx")

    method = "best-2"
    performance_metrics[operator + "_" + method] = {}

    comb_method = pd.concat([best_method.sample(real_train.shape[0], random_state= 42, replace= True), real_train], ignore_index=True)
    if os.path.exists("./model/rf/synth_{}_{}_rf.pkl".format(method, operator)):
        rf = pickle.load(open("./model/rf/synth_{}_{}_rf.pkl".format(method, operator), 'rb'))
    else:
        #rf = RandomForestClassifier(random_state=42, verbose=2, n_jobs=-1)
        #rf = make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=42, verbose=2))
        rf = make_pipeline(StandardScaler(), GridSearchCV(XGBClassifier(random_state=42),
                                                          param_grid={"eta": [0.5, 0.1, 0.05],
                                                                      "min_child_weight": [0.5, 1, 4],
                                                                      "max_depth": [3, 6, 9],
                                                                      "scale_pos_weight": [1, 0.1, 0.025]},
                                                          cv=5,
                                                          refit=True,
                                                          scoring="roc_auc",
                                                          n_jobs=-1,
                                                          verbose=1))
        rf.fit(comb_method.drop(columns=["isFraud", "operator"]), comb_method["isFraud"].astype(int))
        pickle.dump(rf, open("./model/rf/synth_{}_{}_rf.pkl".format(method, operator), 'wb'))

    test_preds = rf.predict(test_data.drop(columns=["isFraud"]))
    performance_metrics[operator + "_" + method]["accuracy"] = accuracy_score(test_preds, test_data["isFraud"])
    performance_metrics[operator + "_" + method]["f1score"] = f1_score(test_preds, test_data["isFraud"])
    performance_metrics[operator + "_" + method]["rocaucscore"] = roc_auc_score(test_preds,test_data["isFraud"])
    precision, recall, thresholds = precision_recall_curve(test_data["isFraud"], test_preds)
    performance_metrics[operator + "_" + method]["praucscore"] = auc(recall, precision)
    performance_metrics[operator + "_" + method]["precision"] = precision_score(test_preds,test_data["isFraud"])
    performance_metrics[operator + "_" + method]["recall"] = recall_score(test_preds, test_data["isFraud"])
    print("Synth data ({}, {}) Performance: {}".format(method, operator, performance_metrics[operator + "_" + method]))

    pd.DataFrame(performance_metrics).to_excel("./results/performance.xlsx")

    method = "best-3"
    performance_metrics[operator + "_" + method] = {}

    comb_method = pd.concat([best_method[best_method["operator"] != operator].sample(real_train.shape[0], random_state=42, replace= True), real_train], ignore_index=True)
    if os.path.exists("./model/rf/synth_{}_{}_rf.pkl".format(method, operator)):
        rf = pickle.load(open("./model/rf/synth_{}_{}_rf.pkl".format(method, operator), 'rb'))
    else:
        #rf = RandomForestClassifier(random_state=42, verbose=2, n_jobs=-1)
        #rf = make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=42, verbose=2))
        rf = make_pipeline(StandardScaler(), GridSearchCV(XGBClassifier(random_state=42),
                                                          param_grid={"eta": [0.5, 0.1, 0.05],
                                                                      "min_child_weight": [0.5, 1, 4],
                                                                      "max_depth": [3, 6, 9],
                                                                      "scale_pos_weight": [1, 0.1, 0.025]},
                                                          cv=5,
                                                          refit=True,
                                                          scoring="roc_auc",
                                                          n_jobs=-1,
                                                          verbose=1))
        rf.fit(comb_method.drop(columns=["isFraud", "operator"]), comb_method["isFraud"].astype(int))
        pickle.dump(rf, open("./model/rf/synth_{}_{}_rf.pkl".format(method, operator), 'wb'))

    test_preds = rf.predict(test_data.drop(columns=["isFraud"]))
    performance_metrics[operator + "_" + method]["accuracy"] = accuracy_score(test_preds, test_data["isFraud"])
    performance_metrics[operator + "_" + method]["f1score"] = f1_score(test_preds, test_data["isFraud"])
    performance_metrics[operator + "_" + method]["rocaucscore"] = roc_auc_score(test_preds, test_data["isFraud"])
    precision, recall, thresholds = precision_recall_curve(test_data["isFraud"], test_preds)
    performance_metrics[operator + "_" + method]["praucscore"] = auc(recall, precision)
    performance_metrics[operator + "_" + method]["precision"] = precision_score(test_preds, test_data["isFraud"])
    performance_metrics[ operator + "_" + method]["recall"] = recall_score(test_preds, test_data["isFraud"])
    print("Synth data ({}, {}) Performance: {}".format(method, operator, performance_metrics[operator + "_" + method]))

    pd.DataFrame(performance_metrics).to_excel("./results/performance.xlsx")

    method = "best-4"
    performance_metrics[operator + "_" + method] = {}

    mixin_dict = {}
    for i in trange(1, 100, 5):
        real_train_train, real_train_test = train_test_split(real_train, train_size=0.8, random_state= 42)

        if i == 0:
            comb_method = real_train
            comb_method["operator"] = operator
        else:
            comb_method = pd.concat(
                [best_method[best_method["operator"] != operator].sample(int(real_train_train.shape[0]*(i/100)), random_state=42, replace=True), real_train_train], ignore_index=True)
        rf = make_pipeline(StandardScaler(), GridSearchCV(XGBClassifier(random_state=42),
                                                          param_grid={"eta": [0.5, 0.1, 0.05],
                                                                      "min_child_weight": [0.5, 1, 4],
                                                                      "max_depth": [3, 6, 9],
                                                                      "scale_pos_weight": [1, 0.1, 0.025]},
                                                          cv=3,
                                                          refit=True,
                                                          scoring="roc_auc",
                                                          n_jobs=-1,
                                                          verbose=1))
        rf.fit(comb_method.drop(columns=["isFraud", "operator"]), comb_method["isFraud"].astype(int))
        test_preds = rf.predict(real_train_test.drop(columns=["isFraud"]))
        #precision, recall, thresholds = precision_recall_curve(real_train_test["isFraud"], test_preds)
        #mixin_dict[i] = auc(recall, precision)
        #mixin_dict[i] = f1_score(real_train_test["isFraud"], test_preds)
        mixin_dict[i] = roc_auc_score(real_train_test["isFraud"], test_preds)
    best_mixin = max(mixin_dict, key=mixin_dict.get)
    best_mixins[operator] = best_mixin/100
    comb_method = pd.concat(
        [best_method[best_method["operator"] != operator].sample(int(real_train.shape[0] * (best_mixin / 100)), random_state=42, replace=True),
         real_train], ignore_index=True)
    if os.path.exists("./model/rf/synth_{}_{}_rf.pkl".format(method, operator)):
        rf = pickle.load(open("./model/rf/synth_{}_{}_rf.pkl".format(method, operator), 'rb'))
    else:
        rf = make_pipeline(StandardScaler(), GridSearchCV(XGBClassifier(random_state=42),
                                                          param_grid={"eta": [0.5, 0.1, 0.05],
                                                                      "min_child_weight": [0.5, 1, 4],
                                                                      "max_depth": [3, 6, 9],
                                                                      "scale_pos_weight": [1, 0.1, 0.025]},
                                                          cv=5,
                                                          refit=True,
                                                          scoring="roc_auc",
                                                          n_jobs=-1,
                                                          verbose=1))
        rf.fit(comb_method.drop(columns=["isFraud", "operator"]), comb_method["isFraud"].astype(int))
        pickle.dump(rf, open("./model/rf/synth_{}_{}_rf.pkl".format(method, operator), 'wb'))

    test_preds = rf.predict(test_data.drop(columns=["isFraud"]))
    performance_metrics[operator + "_" + method]["accuracy"] = accuracy_score(test_preds, test_data["isFraud"])
    performance_metrics[operator + "_" + method]["f1score"] = f1_score(test_preds, test_data["isFraud"])
    performance_metrics[operator + "_" + method]["rocaucscore"] = roc_auc_score(test_preds, test_data["isFraud"])
    precision, recall, thresholds = precision_recall_curve(test_data["isFraud"], test_preds)
    performance_metrics[operator + "_" + method]["praucscore"] = auc(recall, precision)
    performance_metrics[operator + "_" + method]["precision"] = precision_score(test_preds, test_data["isFraud"])
    performance_metrics[operator + "_" + method]["recall"] = recall_score(test_preds, test_data["isFraud"])
    print("Synth data ({}, {}) Performance: {}".format(method, operator, performance_metrics[operator + "_" + method]))

    pd.DataFrame(performance_metrics).to_excel("./results/performance.xlsx")

with open("mixin.json", "w") as fp:
    json.dump(best_mixins, fp)