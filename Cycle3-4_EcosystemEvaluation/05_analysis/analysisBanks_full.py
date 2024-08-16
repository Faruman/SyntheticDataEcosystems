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

# bank models
file_path = glob.glob("./working/*Scoring_*.pkl")

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
            if "synth_type" in data:
                sub_metrics_perBankData["synth_type"] = data["synth_type"]
            else:
                sub_metrics_perBankData["synth_type"] = "real"
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
        if "synth_type" in data:
            sub_metrics_df["synth_type"] = data["synth_type"]
        else:
            sub_metrics_df["synth_type"] = "real"
        metrics_df = pd.concat([metrics_df, sub_metrics_df])

# clearly label mixin percentage
metrics_df["use_synth_data"] = metrics_df["mixin_percentage"].apply(lambda x: True if x > 0 else False)
metrics_perBank_df["use_synth_data"] = metrics_perBank_df["mixin_percentage"].apply(lambda x: "Yes" if x > 0 else "No")

# Save the data
metrics_df.to_excel("../results/metrics_df.xlsx")
metrics_perBank_df.to_excel("../results/metrics_perBank_df.xlsx")

# Prepare data for plotting
plot_df = pd.melt(metrics_df, id_vars=["dataset", "type", "mixin_percentage", "use_synth_data", "synth_type"], var_name="metric", value_name="value")
plot_df["type"] = plot_df["type"].str.replace(" ", "\n")
## do not show sepOS data
#plot_df = plot_df.loc[plot_df["synth_type"].isin(["full", "fullOS", "sep", "real"])]
## do not show graph data
#plot_df = plot_df.loc[plot_df["type"].isin(["transactions", "transactions\nover-sampled\n(0.2)"])]
plot_df = plot_df.loc[plot_df["type"].isin(["transactions\nover-sampled\n(0.2)"])]

plot_perBank_df = pd.melt(metrics_perBank_df, id_vars=["dataset", "type", "mixin_percentage", "use_synth_data", "synth_type", "bank"], var_name="metric", value_name="value")
plot_perBank_df["type"] = plot_perBank_df["type"].str.replace(" ", "\n")
## do not show sepOS data
#plot_perBank_df = plot_perBank_df.loc[plot_perBank_df["synth_type"].isin(["full", "fullOS", "sep", "real"])]
## do not show graph data
#plot_perBank_df = plot_perBank_df.loc[plot_perBank_df["type"].isin(["transactions", "transactions\nover-sampled\n(0.2)"])]
plot_perBank_df = plot_perBank_df.loc[plot_perBank_df["type"].isin(["transactions\nover-sampled\n(0.2)"])]

# sort data
plot_df = plot_df.sort_values(by=["dataset", "type", "synth_type"])
plot_perBank_df = plot_perBank_df.sort_values(by=["dataset", "type", "synth_type"])
plot_df["sort_index"] = plot_df["synth_type"].apply(lambda x: 0 if x == "real" else 5 if "full+sep" in x else 1 if "full" in x else 4 if "sepPre" in x else 3 if "sep" in x else 6)
plot_df = plot_df.sort_values(by= ["sort_index", "synth_type"]).reset_index(drop=True)
plot_df = plot_df.drop("sort_index", axis=1)
plot_perBank_df["sort_index"] = plot_perBank_df["synth_type"].apply(lambda x: 0 if x == "real" else 5 if "full+sep" in x else 1 if "full" in x else 4 if "sepPre" in x else 3 if "sep" in x else 6)
plot_perBank_df = plot_perBank_df.sort_values(by= ["sort_index", "synth_type"]).reset_index(drop=True)
plot_perBank_df = plot_perBank_df.drop("sort_index", axis=1)

# Plotting best synth generation strategy
g = sns.catplot(data=plot_df.loc[(plot_df["metric"] == "roc-auc-score") & (plot_df["synth_type"] != "real")], kind="bar", x="type", y="value", hue="synth_type", col="dataset", linewidth=1, edgecolor="black")
g.fig.suptitle('Evaluation of Synth Generation Strategy')
g.set_ylabels('Value')
g.set_xlabels('Learning Scheme')
g.set(ylim=(0.5, 1))
apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
plt.subplots_adjust(top=0.85)
plt.savefig("../plots/evaluation_synthType_rocauc_bw.png")
plt.show()

g = sns.catplot(data=plot_perBank_df.loc[(plot_perBank_df["metric"] == "roc-auc-score") & (plot_perBank_df["synth_type"] != "real")], kind="bar", x="type", y="value", hue="synth_type", col="dataset", linewidth=1, edgecolor="black")
#def add_vline(data, **kwargs):
#    dataset = data["dataset"].value_counts().idxmax()
#    plt.axhline(plot_perBank_df.loc[(plot_perBank_df["metric"] == "roc-auc-score") & (plot_perBank_df["synth_type"] == "real") & (plot_perBank_df["dataset"] == dataset)]["value"].mean(), color='black')
#g.map_dataframe(add_vline)
g.fig.suptitle('Evaluation of Synth Generation Strategy (per Bank)')
g.set_ylabels('Value')
g.set_xlabels('Learning Scheme')
g.set(ylim=(0.5, 1))
apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
plt.subplots_adjust(top=0.85)
plt.savefig("../plots/evaluation_synthType_perBank_rocauc_bw.png")
plt.show()

g = sns.catplot(data=plot_perBank_df.loc[(plot_perBank_df["metric"] == "roc-auc-score") & (plot_perBank_df["synth_type"] != "real")  & (plot_perBank_df["synth_type"].isin(["full", "fullOS_05", "fullOS_10", "fullOS_20", "sep", "sepPre", "full+sep"]))], kind="bar", x="type", y="value", hue="synth_type", col="dataset", linewidth=1, edgecolor="black")
g.fig.suptitle('Evaluation of Synth Generation Strategy (per Bank)')
g.set_ylabels('Value')
g.set_xlabels('Learning Scheme')
g.set(ylim=(0.5, 1))
apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
plt.subplots_adjust(top=0.85)
plt.savefig("../plots/evaluation_synthType_perBank_selected_rocauc_bw.png")
plt.show()

g = sns.catplot(data=plot_perBank_df.loc[(plot_perBank_df["metric"] == "f1-score")], kind="bar", x="type", y="value", hue="synth_type", col="dataset", linewidth=1, edgecolor="black")
g.fig.suptitle('Evaluation of Synth Generation Strategy (per Bank)')
g.set_ylabels('Value')
g.set_xlabels('Learning Scheme')
apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
plt.subplots_adjust(top=0.85)
plt.savefig("../plots/evaluation_synthType_perBank_withReal_rocauc_bw.png")
plt.tight_layout()
plt.show()

plot_perBank_df.loc[(plot_perBank_df["metric"] == "roc-auc-score") & (plot_perBank_df["synth_type"].isin(["real", "full", "fullOS_05", "fullOS_10", "fullOS_20", "sep", "sepPre", "full+sep"]))].groupby(["dataset", "type", "synth_type"])["value"].mean().to_excel("../results/evaluation_synthType_perBank.xlsx")


temp = plot_perBank_df.loc[(plot_perBank_df["metric"] == "roc-auc-score")]
temp.groupby(["dataset", "type", "synth_type"])["value"].mean()

# filter best synth generation strategy
selected_synthtype = "sepPre"
plot_df = plot_df.loc[plot_df["synth_type"].isin([selected_synthtype, "real"])]
plot_perBank_df = plot_perBank_df.loc[plot_perBank_df["synth_type"].isin([selected_synthtype, "real"])]

# Plotting best mix-in percentage
g = sns.relplot(data=plot_df.loc[plot_df["metric"] == "roc-auc-score"], x="mixin_percentage", y="value", style="type", col="dataset", kind="line", color="black")
def add_vline(data, **kwargs):
    plt.axvline(data.set_index("mixin_percentage")["value"].idxmax(), color='grey', linestyle='dotted')
g.map_dataframe(add_vline)
g.fig.suptitle('Evaluation of Mix-in Percentage')
g.set_ylabels('Value')
g.set_xlabels('Mix-in Percentage')
g.set(ylim=(0.5, 1))
for ax in g.axes.flat:
    ax.xaxis.set_major_formatter(PercentFormatter(1))
legend = g._legend
dotted_line = mlines.Line2D([], [], color='grey', linestyle='dotted', label='Optimum')
handles, labels = legend.legendHandles, [t.get_text() for t in legend.texts]
handles.append(dotted_line)
labels.append('Optimum')
legend.set_title("type")
g._legend.remove()
g.fig.legend(handles=handles, labels=labels, loc='center right', title="type", frameon=False)
plt.subplots_adjust(top=0.85)
plt.savefig("../plots/evaluation_mixinPercentage_rocauc_bw.png")
plt.show()

g = sns.relplot(data=plot_perBank_df.loc[plot_perBank_df["metric"] == "roc-auc-score"], x="mixin_percentage", y="value", style="type", col="dataset", kind="line", color="black")
def add_vline(data, **kwargs):
    plt.axvline(data.groupby("mixin_percentage")["value"].mean().idxmax(), color='grey', linestyle='dotted')
g.map_dataframe(add_vline)
g.fig.suptitle('Evaluation of Mix-in Percentage (per Bank)')
g.set_ylabels('Value')
g.set_xlabels('Mix-in Percentage')
g.set(ylim=(0.5, 1))
for ax in g.axes.flat:
    ax.xaxis.set_major_formatter(PercentFormatter(1))
legend = g._legend
red_dotted_line = mlines.Line2D([], [], color='grey', linestyle='dotted', label='Optimum')
handles, labels = legend.legendHandles, [t.get_text() for t in legend.texts]
handles.append(red_dotted_line)
labels.append('Optimum')
legend.set_title("type")
g._legend.remove()
g.fig.legend(handles=handles, labels=labels, loc='center right', title="type", frameon=False)
plt.subplots_adjust(top=0.85)
plt.savefig("../plots/evaluation_mixinPercentage_perBank_rocauc_bw.png")
plt.show()

#g = sns.relplot(data=plot_perBank_df.loc[plot_perBank_df["metric"] == "roc-auc-score"], x="mixin_percentage", y="value", hue="type", col="dataset", style= "bank", kind="line", errorbar= None, palette="Greys")
#g.fig.suptitle('Evaluation of Mix-in Percentage (per Bank)')
#g.set_ylabels('Value')
#g.set_xlabels('Mix-in Percentage')
#g.set(ylim=(0.5, 1))
#plt.subplots_adjust(top=0.85)
#plt.savefig("../plots/evaluation_mixinPercentage_perBank_single_rocauc_bw.png")
#plt.show()

g = sns.relplot(data=plot_perBank_df.loc[plot_perBank_df["metric"] == "roc-auc-score"], x="mixin_percentage", y="value", col="dataset", style= "bank", kind="line", errorbar= None, color= "grey")
def add_meanline(data, **kwargs):
    sns.lineplot(data= data[data["metric"] == "roc-auc-score"].groupby(["dataset", "mixin_percentage"])["value"].mean().reset_index(), x="mixin_percentage", y="value", color='black', )
g.map_dataframe(add_meanline)
g.fig.suptitle('Evaluation of Mix-in Percentage (per Bank)')
g.set_ylabels('Value')
g.set_xlabels('Mix-in Percentage')
g.set(ylim=(0.5, 1))
legend = g._legend
black_line = mlines.Line2D([], [], color='black', label='Mean')
handles, labels = legend.legendHandles, [t.get_text() for t in legend.texts]
handles.append(black_line)
labels.append('Mean')
legend.set_title("Bank")
g._legend.remove()
g.fig.legend(handles=handles, labels=labels, loc='center right', title="Bank", frameon=False)
plt.subplots_adjust(top=0.85)
plt.savefig("../plots/evaluation_mixinPercentage_perBank_single_rocauc_bw.png")
plt.show()


for dataset in plot_perBank_df["dataset"].unique():
    print(dataset)
    temp = plot_perBank_df.loc[(plot_perBank_df["dataset"] == dataset) & (plot_perBank_df["metric"] == "roc-auc-score")]
    est = sm.OLS(temp["value"].values, sm.tools.tools.add_constant(temp["mixin_percentage"].astype(float)))
    est = est.fit()
    print(est.summary())

plot_perBank_df = plot_perBank_df.merge(plot_perBank_df.loc[plot_perBank_df["mixin_percentage"] == 0, ["dataset", "type", "bank", "metric", "value"]], how="left", on=  ["dataset", "type", "bank", "metric"], suffixes= ("", "_baseline"))
plot_perBank_df = plot_perBank_df.rename(columns={"value_baseline": "baseline"})
plot_perBank_df["perf gain"] = plot_perBank_df["value"] - plot_perBank_df["baseline"]
plot_perBank_df["perf gain pct"] = plot_perBank_df["perf gain"] / plot_perBank_df["baseline"]
plot_perBank_df["data-bank"] = plot_perBank_df["dataset"].astype(str) + "-" + plot_perBank_df["bank"].astype(int).astype(str)
plot_perBank_df = plot_perBank_df.merge(dataset_df.drop(columns= ["Bank Id", "dataset"]), on= "data-bank")

# plotting the relationship between performance gain and mix-in percentage
for type in plot_perBank_df["type"].unique():
    for dataset in plot_perBank_df["dataset"].unique():
        fig, g = plt.subplots()
        sns.scatterplot(data=plot_perBank_df.loc[(plot_perBank_df["type"] == type) & (plot_perBank_df["dataset"] == dataset)], x="mixin_percentage", y="perf gain pct", palette="Greys", ax=g,
                        hue= "bank_size", style= "bank_size")
        for handle, bank_size in zip(*g.get_legend_handles_labels()):
            color = handle.get_c()
            temp = plot_perBank_df.loc[(plot_perBank_df["type"] == type) & (plot_perBank_df["dataset"] == dataset) & (plot_perBank_df["bank_size"] == bank_size)]
            poly = PolynomialFeatures(degree=2, include_bias=True)
            x = poly.fit_transform(temp["mixin_percentage"].values.reshape(-1, 1))
            est = sm.OLS(temp["perf gain pct"], sm.tools.tools.add_constant(x))
            est = est.fit()
            print(est.summary())
            sns.lineplot(x=np.arange(*g.xaxis.get_data_interval(), 0.01),
                         y=est.predict(poly.transform(np.arange(*g.xaxis.get_data_interval(), 0.01).reshape(-1, 1))),
                         color=color,
                         linestyle="--", ax=g)
        g.xaxis.set_major_formatter(PercentFormatter(1))
        g.yaxis.set_major_formatter(PercentFormatter(1))
        g.set(ylim=(round(plot_perBank_df["perf gain"].min() * 1.2, 2), round(plot_perBank_df["perf gain"].max() * 1.2, 2)))
        plt.xlabel('Mix-in Percentage')
        plt.ylabel('Performance Gain Synthetic Data (ROC-AUC)')
        plt.title('Performance Gain by Mix-In Perfcentage\n(' + type.replace("\n", "") + " - " + dataset + ")")
        plt.subplots_adjust(top=0.90)
        plt.tight_layout()
        plt.savefig("../plots/perfGainVsMixinPct_{}_{}_rocauc_bw.png".format(type.replace("\n", ""), dataset))
        plt.show()

# get best mix-in percentage
def getBestMixinPercentage(x, score, use_average=False):
    if len(x["mixin_percentage"].value_counts()) > 1:
        if not use_average:
            best_mixin_pct = x.loc[x.loc[x["metric"] == score, "value"].idxmax(), "mixin_percentage"]
        else:
            best_mixin_pct = x.groupby("mixin_percentage")["value"].max().idxmax()
        return x.loc[x["mixin_percentage"] == best_mixin_pct]
    else:
        return x
plot_df = plot_df.groupby(["type", "dataset", "use_synth_data"]).apply(lambda x: getBestMixinPercentage(x, "roc-auc-score", True)).reset_index(drop=True)
plot_perBank_df = plot_perBank_df.groupby(["type", "dataset", "use_synth_data", "bank"]).apply(lambda x: getBestMixinPercentage(x, "roc-auc-score", False)).reset_index(drop=True)

# Plotting comparison of evaluation models
g = sns.catplot(data=plot_df.loc[(plot_df["metric"] == "roc-auc-score")], kind="bar", x= "type", y="value", hue="use_synth_data", col="dataset", linewidth=1, edgecolor="black")
g.fig.suptitle('Evaluation of Detection Models')
g.set_ylabels('ROC-AUC Score')
g.set_xlabels('Synthetic Data Usage')
g.set(ylim=(0.5, 1))
apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
plt.subplots_adjust(top=0.85)
plt.savefig("../plots/evaluation_detectionModel_rocauc_bw.png")
plt.show()

g = sns.catplot(data=plot_perBank_df.loc[(plot_perBank_df["metric"] == "roc-auc-score")], kind="bar", x= "type", y="value", hue="use_synth_data", col="dataset", errorbar=None, linewidth=1, edgecolor="black")
g.fig.suptitle('Evaluation of Detection Models (per Bank)')
g.set_ylabels('ROC-AUC Score')
g.set_xlabels('Synthetic Data Usage')
g.set(ylim=(0.5, 1))
apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
plt.subplots_adjust(top=0.85)
plt.savefig("../plots/evaluation_detectionModel_perBank_rocauc_bw.png")
plt.show()

# Plotting the metrics in overview
g = sns.catplot(data=plot_df, kind="bar", x="type", y="value", col="dataset", row= "metric", hue="use_synth_data", linewidth=1, edgecolor="black")
g.fig.suptitle('Performance Metrics by Financial Institution')
g.set_ylabels('Metric')
g.set_xlabels('Financial Institution')
g.set(ylim=(0.5, 1))
apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
plt.subplots_adjust(top=0.95)
plt.savefig("../plots/metrics_overview_rocauc_bw.png")
plt.show()

g = sns.catplot(data=plot_perBank_df, kind="bar", x="type", y="value", col="dataset", row= "metric", hue="use_synth_data", linewidth=1, edgecolor="black")
g.fig.suptitle('Performance Metrics by Learning Scheme')
g.set_ylabels('Metric')
g.set_xlabels('Learning Scheme')
g.set(ylim=(0.5, 1))
apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
plt.subplots_adjust(top=0.95)
plt.savefig("../plots/metrics_overview_perBank_rocauc_bw.png")
plt.show()

#plotting the roc-auc score
g = sns.catplot(data=plot_df.loc[plot_df["metric"] == "roc-auc-score"], kind="bar", x="type", y="value", hue="use_synth_data", col="dataset", linewidth=1, edgecolor="black")
g.fig.suptitle('ROC-AUC Score by Learning Scheme')
g.set_ylabels('ROC-AUC Score')
g.set_xlabels('Learning Scheme')
g.set(ylim=(0.5, 1))
apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
plt.subplots_adjust(top=0.85)
plt.savefig("../plots/metrics_rocAuc_rocauc_bw.png")
plt.show()

g = sns.catplot(data=plot_perBank_df.loc[plot_perBank_df["metric"] == "roc-auc-score"], kind="bar", x="type", y="value", hue="use_synth_data", col="dataset", linewidth=1, edgecolor="black")
g.fig.suptitle('ROC-AUC Score by Learning Scheme')
g.set_ylabels('ROC-AUC Score')
g.set_xlabels('Learning Scheme')
g.set(ylim=(0.5, 1))
apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
plt.subplots_adjust(top=0.85)
plt.savefig("../plots/metrics_rocAuc_perBank_rocauc_bw.png")
plt.show()

g = sns.catplot(data=plot_perBank_df.loc[plot_perBank_df["metric"] == "roc-auc-score"], kind="bar", x="bank", y="value", hue="use_synth_data", row="type", col="dataset", sharex=False, linewidth=1, edgecolor="black")
g.fig.suptitle('ROC-AUC Score by Learning Scheme')
g.set_ylabels('ROC-AUC Score')
g.set_xlabels('Bank ID')
g.set(ylim=(0.5, 1))
apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
plt.subplots_adjust(top=0.90)
plt.savefig("../plots/metrics_rocAuc_perBank_single_rocauc_bw.png")
plt.show()

#plotting performance gain
perfGain_df = plot_perBank_df.loc[plot_perBank_df["metric"] == "roc-auc-score"].pivot(index= ['dataset', 'type', 'bank'], columns= "synth_type", values= "value").reset_index()
perfGain_df["perf gain"] = perfGain_df[selected_synthtype] - perfGain_df["real"]
perfGain_df["perf gain pct"] = perfGain_df["perf gain"] / perfGain_df["real"]
perfGain_df["data-bank"] = perfGain_df["dataset"].astype(str) + "-" + perfGain_df["bank"].astype(int).astype(str)
perfGain_df = perfGain_df.merge(dataset_df.drop(columns= ["Bank Id", "dataset"]), on= "data-bank")

plot_perBank_df["data-bank"] = plot_perBank_df["dataset"].astype(str) + "-" + plot_perBank_df["bank"].astype(int).astype(str)
#plot_perBank_df = plot_perBank_df.merge(dataset_df.drop(columns= ["Bank Id", "dataset"]), on= "data-bank")

g = sns.scatterplot(data=perfGain_df, x="Pct of Total", y="perf gain", hue="dataset", style="dataset", palette= "Greys")
g.axhline(0, color='grey', lw=1)
g.xaxis.set_major_formatter(PercentFormatter(1))
plt.xlabel('Institution Size \n(% of Total Dataset)')
plt.ylabel('Performance Gain Synthetic Data (ROC-AUC)')
plt.title('Performance Gain by Dataset Size')
plt.subplots_adjust(top=0.90)
plt.savefig("../plots/perfGainVsDatasetSize_rocauc_bw.png")
plt.show()

# plotting the influence of performance gain due to synthetic data by instituion size (absolute)
for type in perfGain_df["type"].unique():
    fig, g = plt.subplots()
    sns.scatterplot(data=perfGain_df.loc[perfGain_df["type"] == type], x="Pct of Total", y="perf gain", style="dataset", hue= "dataset", palette= "Greys", ax= g)
    g.axhline(0, color='grey', lw=1)
    g.xaxis.set_major_formatter(PercentFormatter(1))
    for handle, dataset in zip(*g.get_legend_handles_labels()):
        color = handle.get_c()
        temp = perfGain_df.loc[(perfGain_df["type"] == type) & (perfGain_df["dataset"] == dataset)]
        #z = np.polyfit(x["Pct of Total"], x["perf gain"], 1)
        #p = np.poly1d(z)
        x = sm.add_constant(temp["Pct of Total"])
        est = sm.OLS(temp["perf gain"], sm.tools.tools.add_constant(x))
        est = est.fit()
        print(est.summary())
        sns.lineplot(x=np.arange(*g.xaxis.get_data_interval(), 0.05), y=est.predict(sm.add_constant(np.arange(*g.xaxis.get_data_interval(), 0.05))), color=color, linestyle="--", ax= g)
        #g.axhline(perfGain_df.loc[(perfGain_df["type"] == type) & (perfGain_df["dataset"] == dataset), "perf gain pct"].mean(), color=color, lw=1, linestyle="--")
    g.set(ylim=(round(perfGain_df["perf gain"].min() * 1.2, 2), round(perfGain_df["perf gain"].max() * 1.2, 2)))
    plt.xlabel('Institution Size \n(% of Total Dataset)')
    plt.ylabel('Performance Gain Synthetic Data (ROC-AUC)')
    plt.title('Performance Gain by Dataset Size\n(' + type.replace("\n", "") + ")")
    plt.subplots_adjust(top=0.90)
    avg_line = mlines.Line2D([], [], color='black', marker='', linestyle="--", label='avg')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(avg_line)
    labels.append('Regression\nLine')
    plt.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("../plots/perfGainVsDatasetSize_{}_rocauc_bw.png".format(type.replace("\n", "")))
    plt.show()

    fig, g = plt.subplots()
    sns.scatterplot(data=perfGain_df.loc[perfGain_df["type"] == type], x="Pct of Total", y="perf gain", style="dataset", hue= "dataset", palette= "Greys", ax= g)
    g.axhline(0, color='grey', lw=1)
    temp = perfGain_df.loc[perfGain_df["type"] == type]
    x = sm.add_constant(temp["Pct of Total"])
    est = sm.OLS(temp["perf gain"], sm.tools.tools.add_constant(x))
    est = est.fit()
    print(est.summary())
    sns.lineplot(x=np.arange(*g.xaxis.get_data_interval(), 0.05),
                 y=est.predict(sm.add_constant(np.arange(*g.xaxis.get_data_interval(), 0.05))), color="black",
                 linestyle="--", ax=g, label="Regression\nLine")
    g.xaxis.set_major_formatter(PercentFormatter(1))
    g.set(ylim=(round(perfGain_df["perf gain"].min() * 1.2, 2), round(perfGain_df["perf gain"].max() * 1.2, 2)))
    plt.xlabel('Institution Size \n(% of Total Dataset)')
    plt.ylabel('Performance Gain Synthetic Data (ROC-AUC)')
    plt.title('Performance Gain by Dataset Size\n(' + type.replace("\n", "") + ")")
    plt.subplots_adjust(top=0.90)
    plt.tight_layout()
    plt.savefig("../plots/perfGainVsDatasetSize_{}_combReg_rocauc_bw.png".format(type.replace("\n", "")))
    plt.show()

sns.set_theme(style="white", font_scale=0.8)
g = sns.FacetGrid(perfGain_df, col="type", hue = "dataset", palette= "Greys")
g.fig.suptitle('Performance Gain by Institution Size')
g.map(plt.scatter, "Pct of Total", "perf gain", s=7.5)
g.set_xlabels('Institution Size \n(% of Total Dataset)')
g.set_ylabels('Performance Gain Synthetic\n Data (ROC-AUC)')
for ax in g.axes.flat:
    ax.axhline(0, color='black', lw=1, linestyle= "--")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
plt.tight_layout()
plt.savefig("../plots/perfGainVsDatasetSize_byType_rocauc_bw.png")
plt.show()

# plotting the influence of performance gain due to synthetic data by instituion size (relative)
for type in perfGain_df["type"].unique():
    fig, g = plt.subplots()
    sns.scatterplot(data=perfGain_df.loc[perfGain_df["type"] == type], x="Pct of Total", y="perf gain pct", hue="dataset", style= "dataset", palette= "Greys", ax= g)
    g.axhline(0, color='black', lw=1, linestyle= "--")
    g.yaxis.set_major_formatter(PercentFormatter(1))
    g.xaxis.set_major_formatter(PercentFormatter(1))
    for handle, dataset in zip(*g.get_legend_handles_labels()):
        color = handle.get_c()
        temp = perfGain_df.loc[(perfGain_df["type"] == type) & (perfGain_df["dataset"] == dataset)]
        poly = PolynomialFeatures(degree=1, include_bias=True)
        x = poly.fit_transform(temp["Pct of Total"].values.reshape(-1, 1))
        est = sm.OLS(temp["perf gain pct"], sm.tools.tools.add_constant(x))
        est = est.fit()
        print(est.summary())
        sns.lineplot(x=np.arange(*g.xaxis.get_data_interval(), 0.01),
                     y=est.predict(poly.transform(np.arange(*g.xaxis.get_data_interval(), 0.01).reshape(-1, 1))), color=color,
                     linestyle="--", ax=g)
        #g.axhline(perfGain_df.loc[(perfGain_df["type"] == type) & (perfGain_df["dataset"] == dataset), "perf gain pct"].mean(), color=color, lw=1, linestyle= "--")
    g.set(ylim=(round(perfGain_df["perf gain pct"].min() * 1.2, 2), round(perfGain_df["perf gain pct"].max() * 1.2, 2)))
    plt.xlabel('Institution Size \n(% of Total Dataset)')
    plt.ylabel('Performance Gain Synthetic Data (ROC-AUC)')
    plt.title('Performance Gain by Dataset Size\n(' + type.replace("\n", "") + ")")
    plt.subplots_adjust(top=0.90)
    avg_line = mlines.Line2D([], [], color='black', marker='', linestyle="--", label='avg')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(avg_line)
    labels.append('Regression\nLine')
    plt.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("../plots/perfGainVsDatasetSize_{}_relative_rocauc_bw.png".format(type.replace("\n", "")))
    plt.show()

    fig, g = plt.subplots()
    sns.scatterplot(data=perfGain_df.loc[perfGain_df["type"] == type], x="Pct of Total", y="perf gain pct", style="dataset", hue= "dataset", palette= "Greys", ax= g)
    g.axhline(0, color='grey', lw=1)
    temp = perfGain_df.loc[perfGain_df["type"] == type]
    poly = PolynomialFeatures(degree=1, include_bias=True)
    x = poly.fit_transform(temp["Pct of Total"].values.reshape(-1, 1))
    est = sm.OLS(temp["perf gain pct"], sm.tools.tools.add_constant(x))
    est = est.fit()
    print(est.summary())
    sns.lineplot(x=np.arange(*g.xaxis.get_data_interval(), 0.01),
                 y=est.predict(poly.transform(np.arange(*g.xaxis.get_data_interval(), 0.01).reshape(-1, 1))), color="black",
                 linestyle="--", ax=g, label="Regression\nLine")
    g.xaxis.set_major_formatter(PercentFormatter(1))
    g.set(ylim=(round(perfGain_df["perf gain pct"].min() * 1.2, 2), round(perfGain_df["perf gain pct"].max() * 1.2, 2)))
    plt.xlabel('Institution Size \n(% of Total Dataset)')
    plt.ylabel('Performance Gain Synthetic Data (ROC-AUC)')
    plt.title('Performance Gain by Dataset Size\n(' + type.replace("\n", "") + ")")
    plt.subplots_adjust(top=0.90)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("../plots/perfGainVsDatasetSize_{}_combReg_relative_rocauc_bw.png".format(type.replace("\n", "")))
    plt.show()

sns.set_theme(style="white", font_scale=0.8)
g = sns.FacetGrid(perfGain_df, col="type", hue = "dataset", palette= "Greys")
g.fig.suptitle('Performance Gain by Institution Size')
g.map(plt.scatter, "Pct of Total", "perf gain pct", s=7.5)
g.set_xlabels('Institution Size \n(% of Total Dataset)')
g.set_ylabels('Performance Gain Synthetic\n Data (% of ROC-AUC)')
for ax in g.axes.flat:
    ax.axhline(0, color='black', lw=1, linestyle= "--")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
plt.tight_layout()
plt.savefig("../plots/perfGainVsDatasetSize_byType_relative_rocauc_bw.png")
plt.show()

#plotting the influence of the optimal mixin percentage by instituion size
sns.set_theme(style="white", font_scale=0.8)
g = sns.FacetGrid(plot_perBank_df.loc[plot_perBank_df["use_synth_data"] == "Yes"], col="type", hue = "dataset", palette= "Greys")
g.fig.suptitle('Optimal Mix-in Percentage by Institution Size')
g.map(plt.scatter, "Pct of Total", "mixin_percentage", s=7.5)
g.set_xlabels('Institution Size \n(% of Total Dataset)')
g.set_ylabels('Mix-in Percentage')
for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
plt.tight_layout()
plt.savefig("../plots/mixinPercentageVsDatasetSize_byType_rocauc_bw.png")
plt.show()

# plotting the influence of mix-in by instituion size (relative)
for type in plot_perBank_df["type"].unique():
    g = sns.scatterplot(data=plot_perBank_df.loc[(plot_perBank_df["type"] == type) & (plot_perBank_df["use_synth_data"] == "Yes")], x="Pct of Total", y="mixin_percentage", hue="dataset", style= "dataset", palette= "Greys")
    #x = plot_perBank_df.loc[(plot_perBank_df["type"] == type) & (plot_perBank_df["use_synth_data"] == "Yes") & (plot_perBank_df["metric"] == "roc-auc-score")]
    #z = np.polyfit(x["Pct of Total"], x["mixin_percentage"].astype(float), 1)
    #p = np.poly1d(z)
    #sns.lineplot(x=np.arange(*g.xaxis.get_data_interval(), 0.05), y=p(np.arange(*g.xaxis.get_data_interval(), 0.05)), color="black", linestyle="--")
    for handle, dataset in zip(*g.get_legend_handles_labels()):
        color = handle.get_c()
        temp = plot_perBank_df.loc[(plot_perBank_df["type"] == type) & (plot_perBank_df["dataset"] == dataset) & (plot_perBank_df["use_synth_data"] == "Yes")]
        poly = PolynomialFeatures(degree=1, include_bias=True)
        x = poly.fit_transform(temp["Pct of Total"].values.reshape(-1, 1))
        est = sm.OLS(temp["mixin_percentage"].astype(float), sm.tools.tools.add_constant(x))
        est = est.fit()
        print(est.summary())
        sns.lineplot(x=np.arange(*g.xaxis.get_data_interval(), 0.01),
                     y=est.predict(poly.transform(np.arange(*g.xaxis.get_data_interval(), 0.01).reshape(-1, 1))), color=color,
                     linestyle="--", ax=g)
    g.yaxis.set_major_formatter(PercentFormatter(1))
    g.xaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel('Institution Size \n(% of Total Dataset)')
    plt.ylabel('Mix-in Percentage')
    plt.title('Optimal Mix-in Percentage by Dataset Size\n(' + type.replace("\n", "") + ")")
    plt.subplots_adjust(top=0.90)
    avg_line = mlines.Line2D([], [], color='black', marker='', linestyle="--", label='avg')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(avg_line)
    labels.append('Regression\nLine')
    plt.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("../plots/mixinPercentageVsDatasetSize_{}_relative_rocauc_bw.png".format(type.replace("\n", "")))
    plt.show()

    g = sns.scatterplot(data=plot_perBank_df.loc[(plot_perBank_df["type"] == type) & (plot_perBank_df["use_synth_data"] == "Yes")], x="Pct of Total", y="mixin_percentage", hue="dataset", style= "dataset", palette= "Greys")
    #x = plot_perBank_df.loc[(plot_perBank_df["type"] == type) & (plot_perBank_df["use_synth_data"] == "Yes") & (plot_perBank_df["metric"] == "roc-auc-score")]
    #z = np.polyfit(x["Pct of Total"], x["mixin_percentage"].astype(float), 1)
    #p = np.poly1d(z)
    #sns.lineplot(x=np.arange(*g.xaxis.get_data_interval(), 0.05), y=p(np.arange(*g.xaxis.get_data_interval(), 0.05)), color="black", linestyle="--")
    temp = plot_perBank_df.loc[(plot_perBank_df["type"] == type) & (plot_perBank_df["use_synth_data"] == "Yes")]
    poly = PolynomialFeatures(degree=1, include_bias=True)
    x = poly.fit_transform(temp["Pct of Total"].values.reshape(-1, 1))
    est = sm.OLS(temp["mixin_percentage"].astype(float), sm.tools.tools.add_constant(x))
    est = est.fit()
    print(est.summary())
    sns.lineplot(x=np.arange(*g.xaxis.get_data_interval(), 0.01),
                 y=est.predict(poly.transform(np.arange(*g.xaxis.get_data_interval(), 0.01).reshape(-1, 1))), color="black",
                 linestyle="--", ax=g, label="Regression\nLine")
    g.yaxis.set_major_formatter(PercentFormatter(1))
    g.xaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel('Institution Size \n(% of Total Dataset)')
    plt.ylabel('Mix-in Percentage')
    plt.title('Optimal Mix-in Percentage by Dataset Size\n(' + type.replace("\n", "") + ")")
    plt.subplots_adjust(top=0.90)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("../plots/mixinPercentageVsDatasetSize_{}_combReg_relative_rocauc_bw.png".format(type.replace("\n", "")))
    plt.show()