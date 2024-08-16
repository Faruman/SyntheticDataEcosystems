import os
import pickle
import glob

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from scipy.stats import norm

def apply_blackAndWhite_to_facetgrid(g, plot_type='bar', hatches= ['xx', 'oo', '--', '||', '//', 'OO', '..', '**'], markers=['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', '+']):
    if plot_type == 'bar' and hatches is None:
        raise ValueError("Hatch patterns must be provided for bar plots.")
    if plot_type == 'scatter' and markers is None:
        raise ValueError("Marker styles must be provided for scatter plots.")
    if plot_type == 'bar':
        color_hatch_map = {}
        hatch_index = 0
        for ax in g.axes.flat:
            for patch in ax.patches:
                color = patch.get_facecolor()
                color_key = tuple(color)
                if color_key not in color_hatch_map:
                    current_hatch = hatches[hatch_index % len(hatches)]
                    color_hatch_map[color_key] = current_hatch
                    hatch_index += 1
                patch.set_hatch(color_hatch_map[color_key])
                patch.set_facecolor('none')  # Remove the fill color
        for patch in g._legend.legend_handles:
            color = patch.get_facecolor()
            color_key = tuple(color)
            patch.set_hatch(color_hatch_map[color_key])
            patch.set_facecolor('none')
    else:
        raise ValueError("Unsupported plot type. Currently supports 'bar' and 'scatter'.")

if os.path.exists("../results"):
    os.makedirs("../results")
if os.path.exists("../plots"):
    os.makedirs("../plots")

# load information on datasets
ibmaml_df = pd.read_excel("./synth/IBM-AML/working/IBM-AML_stats.xlsx")
ibmaml_df["dataset"] = "IBM-AML"
ibmccf_df = pd.read_excel("./synth/IBM-CCF/working/IBM-CCF_stats.xlsx")
ibmccf_df["dataset"] = "IBM-CCF"
dataset_df = pd.concat([ibmaml_df, ibmccf_df])
dataset_df = dataset_df.rename({"Unnamed: 0": "Bank Id"}, axis=1)
dataset_df["data-bank"] = dataset_df["dataset"].astype(str) + "-" + dataset_df["Bank Id"].astype(int).astype(str)
mu, std = norm.fit(dataset_df["Pct of Total"])
dataset_df["bank_size"] = "medium"
dataset_df.loc[dataset_df["Pct of Total"] < mu - 2*std, "bank_size"] = "small"
dataset_df.loc[dataset_df["Pct of Total"] > mu + 2*std, "bank_size"] = "large"

# extract data
metricsRegulator_df = pd.DataFrame(columns=["dataset", "type", "mixin_percentage", "accuracy", "precision", "recall", "f1-score", "roc-auc-score"])
metrics_df = pd.DataFrame(columns=["dataset", "type", "mixin_percentage", "accuracy", "precision", "recall", "f1-score", "roc-auc-score"])


# regulator models
file_path = glob.glob("./working/synth/*Scoring_*.pkl")
for file in file_path:
    with open(file, "rb") as f:
        data = pickle.load(f)
        lines = data["overall"].strip().split("\n")
        lines = [line.replace(" avg", "_avg").split() for line in lines]
        classification_df = [line for line in lines if len(line) == 5]
        sub_metrics_df = [line for line in lines if len(line) == 3]
        classification_df = pd.DataFrame(classification_df, columns=["index", "precision", "recall", "f1-score", "support"])
        classification_df["id"] = data["type"]
        classification_df["dataset"] = data["dataset"]
        sub_metrics_df = pd.DataFrame(sub_metrics_df, columns=["index", "value", "other"]).drop("other", axis=1)
        sub_metrics_df = pd.concat([sub_metrics_df, classification_df.loc[classification_df["index"] == "macro_avg", ["precision", "recall", "f1-score"]].iloc[0].reset_index().rename({2: "value"}, axis=1)])
        sub_metrics_df = sub_metrics_df.set_index("index").T
        sub_metrics_df = sub_metrics_df.astype(float)
        sub_metrics_df["type"] = data["type"]
        sub_metrics_df["dataset"] = data["dataset"]
        if "mixin_percentage" in data:
            sub_metrics_df["mixin_percentage"] = data["mixin_percentage"]
        else:
            sub_metrics_df["mixin_percentage"] = 0
        if "synth_type" in data:
            sub_metrics_df["synth_type"] = data["synth_type"]
        else:
            sub_metrics_df["synth_type"] = "real"
        metricsRegulator_df = pd.concat([metricsRegulator_df, sub_metrics_df])

# Save the data
metricsRegulator_df.to_excel("../results/metricsRegulators_df.xlsx")

# bank models
file_path = glob.glob("./working/*Scoring_*.pkl")
for file in file_path:
    with open(file, "rb") as f:
        data = pickle.load(f)
        lines = data["overall"].strip().split("\n")
        lines = [line.replace(" avg", "_avg").split() for line in lines]
        classification_df = [line for line in lines if len(line) == 5]
        sub_metrics_df = [line for line in lines if len(line) == 3]
        classification_df = pd.DataFrame(classification_df, columns=["index", "precision", "recall", "f1-score", "support"])
        classification_df["id"] = data["type"]
        classification_df["dataset"] = data["dataset"]
        sub_metrics_df = pd.DataFrame(sub_metrics_df, columns=["index", "value", "other"]).drop("other", axis=1)
        sub_metrics_df = pd.concat([sub_metrics_df, classification_df.loc[classification_df["index"] == "macro_avg", ["precision", "recall", "f1-score"]].iloc[0].reset_index().rename({2: "value"}, axis=1)])
        sub_metrics_df = sub_metrics_df.set_index("index").T
        sub_metrics_df = sub_metrics_df.astype(float)
        sub_metrics_df["type"] = data["type"]
        sub_metrics_df["dataset"] = data["dataset"]
        if "mixin_percentage" in data:
            sub_metrics_df["mixin_percentage"] = data["mixin_percentage"]
        else:
            sub_metrics_df["mixin_percentage"] = 0
        if "synth_type" in data:
            sub_metrics_df["synth_type"] = data["synth_type"]
        else:
            sub_metrics_df["synth_type"] = "real"
        metrics_df = pd.concat([metrics_df, sub_metrics_df])

# Prepare data for plotting
metricsRegulator_df = pd.melt(metricsRegulator_df, id_vars=["dataset", "type", "mixin_percentage", "synth_type"], var_name="metric", value_name="value")
metricsRegulator_df["type"] = metricsRegulator_df["type"].str.replace(" ", "\n")
## do not show graph data
metricsRegulator_df = metricsRegulator_df.loc[metricsRegulator_df["type"].isin(["transactions\nover-sampled\n(0.2)"])]
metricsRegulator_df["sort_index"] = metricsRegulator_df["synth_type"].apply(lambda x: 0 if x == "real" else 5 if "full+sep" in x else 1 if "full" in x else 4 if "sepPre" in x else 3 if "sep" in x else 6)
metricsRegulator_df = metricsRegulator_df.sort_values(by= ["sort_index", "synth_type"]).reset_index(drop=True)
metricsRegulator_df = metricsRegulator_df.drop("sort_index", axis=1)
metrics_df = pd.melt(metrics_df, id_vars=["dataset", "type", "mixin_percentage", "synth_type"], var_name="metric", value_name="value")
metrics_df["type"] = metrics_df["type"].str.replace(" ", "\n")
metrics_df = metrics_df.loc[metrics_df["type"].isin(["transactions\nover-sampled\n(0.2)"])]
## do not show graph data
#metrics_df = metrics_df.loc[metrics_df["type"].isin(["transactions", "transactions\nover-sampled\n(0.2)"])]

metrics_perBank_df = pd.read_excel("../results/metrics_perBank_df.xlsx")
metrics_perBank_df = metrics_perBank_df.drop("Unnamed: 0", axis=1)
metrics_perBank_df = pd.melt(metrics_perBank_df, id_vars=["dataset", "type", "mixin_percentage", "synth_type", "use_synth_data", "bank"], var_name="metric", value_name="value")
metrics_perBank_df["type"] = metrics_perBank_df["type"].str.replace(" ", "\n")
metrics_perBank_df = metrics_perBank_df.loc[metrics_perBank_df["type"].isin(["transactions\nover-sampled\n(0.2)"])]

metrics_perBank_df = metrics_perBank_df.groupby(["dataset", "metric", "type", "synth_type", "bank"]).apply(lambda x: x.loc[x["value"].idxmax()]).reset_index(drop=True)
metrics_perBank_df.loc[metrics_perBank_df["synth_type"] != "real", "user"] = "Banks with real & synthetic data\n(weighted avg)"
metrics_perBank_df.loc[metrics_perBank_df["synth_type"] == "real", "user"] = "Banks with real data\n(weighted avg)"
metrics_perBank_df = metrics_perBank_df.sort_values(by="user")

# Plotting best synth generation strategy
g = sns.catplot(data=metricsRegulator_df.loc[(metricsRegulator_df["metric"] == "roc-auc-score") & (metricsRegulator_df["synth_type"] != "real")], kind="bar", x="type", y="value", hue="synth_type", col="dataset", linewidth=1, edgecolor="black")
g.fig.suptitle('Evaluation of Synth Generation Strategy for purely Synthetic Data')
g.set_ylabels('Value')
g.set_xlabels('Learning Scheme')
g.set(ylim=(0.5, 1))
apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
plt.subplots_adjust(top=0.85)
plt.savefig("../plots/evaluation_Regulators_synthType_bw.png")
plt.show()

for synth_base_type in ["full", "sepPre", "full+sep"]:
    metricsRegulator_df_filtered = metricsRegulator_df.loc[metricsRegulator_df["synth_type"].apply(lambda x: True if synth_base_type in x else False)]
    # Plotting best synth generation strategy
    plot_df = metricsRegulator_df_filtered.sort_values(by=["synth_type", "dataset"])
    g = sns.catplot(data=plot_df.loc[(plot_df["metric"] == "roc-auc-score") & (plot_df["synth_type"] != "real")], kind="bar", x="dataset", y="value", hue="synth_type", linewidth=1, edgecolor="black")
    g.fig.suptitle('Evaluation of Synthetic Generation Strategy\n for purely Synthetic Data')
    g.set_ylabels('Value')
    g.set_xlabels('Learning Scheme')
    g.set(ylim=(0.5, 1))
    apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
    plt.subplots_adjust(top=0.85)
    plt.savefig("../plots/evaluation_Regulators_{}_bw.png".format(synth_base_type))
    plt.show()

    metricsRegulator_df = metricsRegulator_df.groupby(["dataset", "metric", "type", "synth_type"]).apply(lambda x: x.loc[x["value"].idxmax()]).reset_index(drop= True)
    metricsRegulator_df["user"] = "Regulator with synthetic data"
    #metrics_df = metrics_df.loc[metrics_df["synth_type"] != "real"].groupby(["dataset", "metric", "type", "synth_type"]).apply(lambda x: x.loc[x["value"].idxmax()]).reset_index(drop= True)
    metrics_df = metrics_df.groupby(["dataset", "metric", "type", "synth_type"]).apply(lambda x: x.loc[x["value"].idxmax()]).reset_index(drop= True)
    metrics_df.loc[metrics_df["synth_type"] != "real", "user"] = "Banks with real & synthetic data\n(weighted avg)"
    metrics_df.loc[metrics_df["synth_type"] == "real", "user"] = "Banks with real data\n(weighted avg)"
    metrics_df = metrics_df.sort_values(by= "user")

    # filter for selected synth_type
    selected_synth_type = metricsRegulator_df_filtered.loc[metricsRegulator_df_filtered["metric"] == "roc-auc-score"].groupby(["synth_type", "dataset"])["value"].max().groupby("synth_type").mean().idxmax()
    metricsRegulator_df_filtered = metricsRegulator_df.loc[metricsRegulator_df["synth_type"] == selected_synth_type]
    metrics_df_filtered = metrics_df.loc[metrics_df["synth_type"].isin(["real", synth_base_type])]

    plot_df = pd.concat([metricsRegulator_df_filtered, metrics_df_filtered])
    plot_df = plot_df.sort_values(by="user")

    # Plotting bank performance vs regulator performance
    g = sns.catplot(data=plot_df.loc[(plot_df["metric"] == "roc-auc-score") & (plot_df["type"] == "transactions\nover-sampled\n(0.2)")], kind= "bar", x="dataset", y="value", hue="user", linewidth=1, edgecolor="black")
    g.fig.suptitle('Evaluation of Performance between\nRegulators and Banks')
    g.set_ylabels('ROC-AUC Score')
    g.set_xlabels('Dataset')
    g.set(ylim=(0.5, 1))
    apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
    plt.subplots_adjust(top=0.85)
    plt.savefig("../plots/evaluation_RegulatorsAndBanks_{}_bw.png".format(synth_base_type))
    plt.show()


    metrics_perBank_df_filtered = metrics_perBank_df.loc[metrics_perBank_df["synth_type"].isin(["real", selected_synth_type])]

    plot_df = pd.concat([metricsRegulator_df_filtered, metrics_perBank_df_filtered])

    # Plotting bank performance vs regulator performance
    g = sns.catplot(data=plot_df.loc[(plot_df["metric"] == "roc-auc-score") & (plot_df["type"] == "transactions\nover-sampled\n(0.2)")], kind= "bar", x="dataset", y="value", hue="user", linewidth=1, edgecolor="black")
    g.fig.suptitle('Evaluation of Performance between\nRegulators and Banks')
    g.set_ylabels('ROC-AUC Score')
    g.set_xlabels('Dataset')
    g.set(ylim=(0.5, 1))
    apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
    plt.subplots_adjust(top=0.85)
    plt.savefig("../plots/evaluation_RegulatorsAndBanks_perBank_{}_bw.png".format(synth_base_type))
    plt.show()