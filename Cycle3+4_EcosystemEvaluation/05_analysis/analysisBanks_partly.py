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


if os.path.exists("../plots"):
    os.makedirs("../plots")

# bank models
file_path = glob.glob("./working/*Scoring_*.pkl") + glob.glob("./working/partly/*Scoring_*.pkl")

# load information on datasets
ibmaml_df = pd.read_excel("../data/synth/IBM-AML/working/IBM-AML_stats.xlsx")
ibmaml_df["dataset"] = "IBM-AML"
ibmccf_df = pd.read_excel("../data/synth/IBM-CCF/working/IBM-CCF_stats.xlsx")
ibmccf_df["dataset"] = "IBM-CCF"
dataset_df = pd.concat([ibmaml_df, ibmccf_df])
dataset_df = dataset_df.rename({"Unnamed: 0": "Bank Id"}, axis=1)
dataset_df["data-bank"] = dataset_df["dataset"].astype(str) + "-" + dataset_df["Bank Id"].astype(int).astype(str)
mu, std = norm.fit(dataset_df["Pct of Total"])
dataset_df["bank_size"] = "medium"
dataset_df.loc[dataset_df["Pct of Total"] < mu - std, "bank_size"] = "small"
dataset_df.loc[dataset_df["Pct of Total"] > mu + std, "bank_size"] = "large"

# extract data
complete_df = pd.DataFrame(columns=["index", "precision", "recall", "f1-score", "support", "id"])
metrics_df = pd.DataFrame(columns=["dataset", "type", "mixin_percentage", "accuracy", "precision", "recall", "f1-score", "roc-auc-score"])
metrics_perBank_df = pd.DataFrame(columns=["dataset", "type", "mixin_percentage", "accuracy", "precision", "recall", "f1-score", "roc-auc-score"])

for file in file_path:
    with open(file, "rb") as f:
        data = pickle.load(f)
        for i, perBankData in enumerate(data["per_bank"]):
            perBankData = perBankData.strip().split("\n")
            perBankData = [line.replace(" avg", "_avg").split() for line in perBankData]
            classification_perBankData = [line for line in perBankData if len(line) == 5]
            sub_metrics_perBankData = [line for line in perBankData if len(line) == 3]
            classification_perBankData = pd.DataFrame(classification_perBankData, columns=["index", "precision", "recall", "f1-score", "support"])
            sub_metrics_perBankData = pd.DataFrame(sub_metrics_perBankData, columns=["index", "value", "other"]).drop("other", axis=1)
            sub_metrics_perBankData = pd.concat([sub_metrics_perBankData, classification_perBankData.loc[classification_perBankData["index"] == "macro_avg", ["precision", "recall", "f1-score"]].iloc[0].reset_index().rename({2: "value"}, axis=1)])
            sub_metrics_perBankData = sub_metrics_perBankData.set_index("index").T
            sub_metrics_perBankData = sub_metrics_perBankData.astype(float)
            sub_metrics_perBankData["type"] = data["type"]
            sub_metrics_perBankData["dataset"] = data["dataset"]
            sub_metrics_perBankData["bank"] = i
            if "mixin_percentage" in data:
                sub_metrics_perBankData["mixin_percentage"] = data["mixin_percentage"]
            else:
                sub_metrics_perBankData["mixin_percentage"] = 0
            if "network_selection" in data:
                sub_metrics_perBankData["network_selection"] = data["network_selection"]
            else:
                sub_metrics_perBankData["network_selection"] = ""
            if "synth_type" in data:
                sub_metrics_perBankData["synth_type"] = data["synth_type"]
            else:
                sub_metrics_perBankData["synth_type"] = "real"
            if "included_banks" in data:
                sub_metrics_perBankData["included_banks"] = [list(data["included_banks"])]
            else:
                if sub_metrics_perBankData["synth_type"].iloc[0] == "real":
                    sub_metrics_perBankData["included_banks"] = [[]]
                else:
                    sub_metrics_perBankData["included_banks"] = [list(range(len(data["per_bank"])))]
            if "inclusion_pct" in data:
                sub_metrics_perBankData["inclusion_pct"] = data["inclusion_pct"]
            else:
                if sub_metrics_perBankData["synth_type"].iloc[0] == "real":
                    sub_metrics_perBankData["inclusion_pct"] = 0
                else:
                    sub_metrics_perBankData["inclusion_pct"] = 1
            metrics_perBank_df = pd.concat([metrics_perBank_df, sub_metrics_perBankData])
        lines = data["overall"].strip().split("\n")
        lines = [line.replace(" avg", "_avg").split() for line in lines]
        classification_df = [line for line in lines if len(line) == 5]
        sub_metrics_df = [line for line in lines if len(line) == 3]
        classification_df = pd.DataFrame(classification_df, columns=["index", "precision", "recall", "f1-score", "support"])
        classification_df["id"] = data["type"]
        classification_df["dataset"] = data["dataset"]
        complete_df = pd.concat([complete_df, classification_df])
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
        if "network_selection" in data:
            sub_metrics_df["network_selection"] = data["network_selection"]
        else:
            sub_metrics_df["network_selection"] = ""
        if "synth_type" in data:
            sub_metrics_df["synth_type"] = data["synth_type"]
        else:
            sub_metrics_df["synth_type"] = "real"
        if "included_banks" in data:
            sub_metrics_df["included_banks"] = [list(data["included_banks"])]
        else:
            if sub_metrics_df["synth_type"].iloc[0] == "real":
                sub_metrics_df["included_banks"] = [[]]
            else:
                sub_metrics_df["included_banks"] = [list(range(len(data["per_bank"])))]
        if "inclusion_pct" in data:
            sub_metrics_df["inclusion_pct"] = data["inclusion_pct"]
        else:
            if sub_metrics_df["synth_type"].iloc[0] == "real":
                sub_metrics_df["inclusion_pct"] = 0
            else:
                sub_metrics_df["inclusion_pct"] = 1
        metrics_df = pd.concat([metrics_df, sub_metrics_df])

# clearly label mixin percentage
metrics_df["use_synth_data"] = metrics_df["mixin_percentage"].apply(lambda x: True if x > 0 else False)
metrics_perBank_df["use_synth_data"] = metrics_perBank_df["mixin_percentage"].apply(lambda x: "Yes" if x > 0 else "No")
sub_metrics_df["inclusion_pct"] = sub_metrics_df["inclusion_pct"].astype(float)

# Prepare data for plotting
plot_df = pd.melt(metrics_df, id_vars=["dataset", "type", "mixin_percentage", "use_synth_data", "synth_type", "included_banks", "inclusion_pct", "network_selection"], var_name="metric", value_name="value")
plot_df["type"] = plot_df["type"].str.replace(" ", "\n")
## do not show sepOS data
#plot_df = plot_df.loc[plot_df["synth_type"].isin(["full", "fullOS", "sep", "real"])]
## do not show graph data
plot_df = plot_df.loc[plot_df["type"].isin(["transactions\nover-sampled\n(0.2)"])]

plot_perBank_df = pd.melt(metrics_perBank_df, id_vars=["dataset", "type", "mixin_percentage", "use_synth_data", "synth_type", "included_banks", "inclusion_pct", "network_selection", "bank"], var_name="metric", value_name="value")
plot_perBank_df["type"] = plot_perBank_df["type"].str.replace(" ", "\n")
## do not show sepOS data
#plot_perBank_df = plot_perBank_df.loc[plot_perBank_df["synth_type"].isin(["full", "fullOS", "sep", "real"])]
## do not show graph data
plot_perBank_df = plot_perBank_df.loc[plot_perBank_df["type"].isin(["transactions\nover-sampled\n(0.2)"])]

# get best mix-in percentage
def getBestMixinPercentage(x, score, use_average=False):
    if len(x["mixin_percentage"].value_counts()) > 1:
        if not use_average:
            best_mixin_pct = x.loc[x.loc[x["metric"] == score, "value"].idxmax(), "mixin_percentage"]
        else:
            best_mixin_pct = x.loc[x["metric"] == score].groupby("mixin_percentage")["value"].max().idxmax()
        return x.loc[x["mixin_percentage"] == best_mixin_pct]
    else:
        return x

plot_df = plot_df.groupby(["type", "dataset", "inclusion_pct"]).apply(lambda x: getBestMixinPercentage(x, "roc-auc-score", False)).reset_index(drop=True)
plot_perBank_df = plot_perBank_df.groupby(["type", "dataset", "inclusion_pct", "bank"]).apply(lambda x: getBestMixinPercentage(x, "roc-auc-score", False)).reset_index(drop=True)
plot_df = plot_df.loc[plot_df["synth_type"].isin(["real", "sepPre"])]
#plot_df = plot_df.loc[plot_df["type"] == "transactions\nover-sampled\n(0.2)"]
plot_perBank_df = plot_perBank_df.loc[plot_perBank_df["synth_type"].isin(["real", "sepPre"])]
#plot_perBank_df = plot_perBank_df.loc[plot_perBank_df["type"] == "transactions\nover-sampled\n(0.2)"]

plot_df["inclusion_pct"] = plot_df["inclusion_pct"].apply(lambda x: "{:,.0%}".format(x))

# Plotting Comparison between Networks with Different Sizes
g = sns.catplot(data=plot_df.loc[(plot_df["metric"] == "roc-auc-score")], kind="bar", x="inclusion_pct", y="value", hue="synth_type", col="dataset", linewidth=1, edgecolor="black")
#g = sns.catplot(data=plot_df.loc[(plot_df["metric"] == "roc-auc-score")], kind="bar", x="inclusion_pct", y="value", hue="synth_type", col="dataset", linewidth=1, edgecolor="black")
g.fig.suptitle('Comparison between Networks with Different Sizes')
g.set_ylabels('ROC-AUC Score')
g.set_xlabels('% of banks included in FinDEx')
g.set(ylim=(0.5, 1))
apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
plt.subplots_adjust(top=0.85)
plt.savefig("../plots/evaluation_networkGrowth_bw.png")
plt.show()

# Plotting Comparison between Networks with Different Sizes (per Bank)
g = sns.catplot(data=plot_perBank_df.loc[(plot_perBank_df["metric"] == "roc-auc-score")], kind="bar", x="inclusion_pct", y="value", hue="synth_type", col="dataset", errorbar= None,  linewidth=1, edgecolor="black")
#g = sns.catplot(data=plot_perBank_df.loc[(plot_perBank_df["metric"] == "roc-auc-score")], kind="bar", x="inclusion_pct", y="value", hue="synth_type", col="dataset", errorbar= None,  linewidth=1, edgecolor="black")
g.fig.suptitle('Comparison between Networks with Different Sizes (per Bank)')
g.set_ylabels('ROC-AUC Score')
g.set_xlabels('% of banks included in FinDEx')
g.set(ylim=(0.5, 1))
apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
plt.subplots_adjust(top=0.85)
plt.savefig("../plots/evaluation_networkGrowthPerBank_bw.png")
plt.show()

plot_df = plot_df.loc[plot_df["synth_type"].isin(["real", "sepPre"])]
plot_df = plot_df.sort_values(by=["dataset", "inclusion_pct", "synth_type"])
plot_perBank_df = plot_perBank_df.loc[plot_perBank_df["synth_type"].isin(["real", "sepPre"])]
plot_perBank_df = plot_perBank_df.sort_values(by=["dataset", "inclusion_pct", "synth_type"])

# combine both datasets
g = sns.catplot(data=plot_df.loc[(plot_df["metric"] == "roc-auc-score")], kind="bar", x="inclusion_pct", y="value", hue="dataset", linewidth=1, edgecolor="black")
#g = sns.catplot(data=plot_df.loc[(plot_df["metric"] == "roc-auc-score")], kind="bar", x="inclusion_pct", y="value", hue="synth_type", col="dataset", linewidth=1, edgecolor="black")
g.fig.suptitle('Comparison between Networks with Different Sizes')
g.set_ylabels('ROC-AUC Score')
g.set_xlabels('% of banks included in FinDEx')
g.set(ylim=(0.5, 1))
apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
plt.subplots_adjust(top=0.85)
plt.savefig("../plots/evaluation_networkGrowth_combined_bw.png")
plt.show()

# Plotting Comparison between Networks with Different Sizes (per Bank)
g = sns.catplot(data=plot_perBank_df.loc[(plot_perBank_df["metric"] == "roc-auc-score")], kind="bar", x="inclusion_pct", hue="dataset", y="value", errorbar= None,  linewidth=1, edgecolor="black")
#g = sns.catplot(data=plot_perBank_df.loc[(plot_perBank_df["metric"] == "roc-auc-score")], kind="bar", x="inclusion_pct", y="value", hue="synth_type", col="dataset", errorbar= None,  linewidth=1, edgecolor="black")
g.fig.suptitle('Comparison between Networks with Different Sizes\n(per Bank)')
g.set_ylabels('ROC-AUC Score')
g.set_xlabels('% of banks included in FinDEx')
g.set(ylim=(0.5, 1))
apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
plt.subplots_adjust(top=0.85)
plt.savefig("../plots/evaluation_networkGrowthPerBank_combined_bw.png")
plt.show()


## only for testing
g = sns.catplot(data=plot_df.loc[(plot_df["metric"] == "roc-auc-score")], kind="bar", x="inclusion_pct", y="value", hue="synth_type", col="dataset", row="network_selection", linewidth=1, edgecolor="black")
#g = sns.catplot(data=plot_df.loc[(plot_df["metric"] == "roc-auc-score")], kind="bar", x="inclusion_pct", y="value", hue="synth_type", col="dataset", linewidth=1, edgecolor="black")
g.fig.suptitle('Comparison between Networks with Different Sizes')
g.set_ylabels('ROC-AUC Score')
g.set_xlabels('% of banks included in FinDEx')
g.set(ylim=(0.5, 1))
apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
plt.subplots_adjust(top=0.85)
plt.show()

# Plotting Comparison between Networks with Different Sizes (per Bank)
g = sns.catplot(data=plot_perBank_df.loc[(plot_perBank_df["metric"] == "roc-auc-score")], kind="bar", x="inclusion_pct", y="value", hue="synth_type", col="dataset", row="network_selection", errorbar= None,  linewidth=1, edgecolor="black")
#g = sns.catplot(data=plot_perBank_df.loc[(plot_perBank_df["metric"] == "roc-auc-score")], kind="bar", x="inclusion_pct", y="value", hue="synth_type", col="dataset", errorbar= None,  linewidth=1, edgecolor="black")
g.fig.suptitle('Comparison between Networks with Different Sizes (per Bank)')
g.set_ylabels('ROC-AUC Score')
g.set_xlabels('% of banks included in FinDEx')
g.set(ylim=(0.5, 1))
apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
plt.subplots_adjust(top=0.85)
plt.show()