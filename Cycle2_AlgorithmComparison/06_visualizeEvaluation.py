import os.path

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

def apply_blackAndWhite_to_plot(ax, plot_type='bar', hatches= ['xx', 'oo', '--', '||', '//', 'OO', '..', '**'], markers=['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', '+']):
    if plot_type == 'bar' and hatches is None:
        raise ValueError("Hatch patterns must be provided for bar plots.")
    if plot_type == 'scatter' and markers is None:
        raise ValueError("Marker styles must be provided for scatter plots.")
    if plot_type == 'bar':
        color_hatch_map = {}
        hatch_index = 0
        for patch in ax.patches:
            color = patch.get_facecolor()
            color_key = tuple(color)
            if color_key not in color_hatch_map:
                current_hatch = hatches[hatch_index % len(hatches)]
                color_hatch_map[color_key] = current_hatch
                hatch_index += 1
            patch.set_hatch(color_hatch_map[color_key])
            patch.set_facecolor('none')  # Remove the fill color
        #for patch in g._legend.legend_handles:
        #    color = patch.get_facecolor()
        #    color_key = tuple(color)
        #    patch.set_hatch(color_hatch_map[color_key])
        #    patch.set_facecolor('none')
    else:
        raise ValueError("Unsupported plot type. Currently supports 'bar' and 'scatter'.")

if not os.path.exists("./plots"):
    os.makedirs("./plots")

metrics_dict = {"rocaucscore": "ROC-AUC Score", "praucscore": "Precision-Recall-AUC Score", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1score": "F1 Score"}

performance = pd.read_excel("./results/performance.xlsx")
performance.index = performance["Unnamed: 0"]
performance = performance.drop(columns=["Unnamed: 0"])

performance_meta = pd.DataFrame([column.split("_") for column in performance.columns], columns= ["operator", "method"])

# create performance groups
performance_agg = pd.DataFrame()
for method in performance_meta["method"].unique():
    temp_meta = [column for column in performance.columns if method in column and "comb" not in column]
    temp = performance.loc[:, temp_meta]
    temp_agg = pd.concat([pd.Series([method]*len(temp), index= temp.index), temp.mean(axis= 1), temp.std(axis= 1)], axis=1)
    temp_agg.columns = ["method", "mean", "std"]
    performance_agg = pd.concat([performance_agg, temp_agg], axis=0)

performance = performance.reset_index()
performance = performance.rename(columns={"Unnamed: 0": "metric"})
performance = performance.melt(id_vars=["metric"])
performance["method"] = [column.split("_")[1] for column in performance["variable"]]
performance["operator"] = [column.split("_")[0] for column in performance["variable"]]

# plot performance indiv groups
for metric in performance["metric"].unique():
    sns.barplot(data= performance.loc[(performance["metric"] == metric).values & np.array(["comb" not in x for x in performance["operator"]])],
                x= "method",
                y= "value",
                errorbar= None,
                palette= "Blues")
    plt.ylabel(metrics_dict[metric])
    plt.savefig("./plots/indiv_{}_performance.png".format(metric))
    #plt.show()
    plt.close()

    g = sns.barplot(data= performance.loc[(performance["metric"] == metric).values & np.array(["comb" not in x for x in performance["operator"]])],
                x= "method",
                y= "value",
                errorbar= None,
                palette= "Blues",
                linewidth=1,
                edgecolor="black")
    apply_blackAndWhite_to_plot(g, plot_type='bar')
    plt.ylabel(metrics_dict[metric])
    plt.savefig("./plots/indiv_{}_performance_bw.png".format(metric))
    #plt.show()
    plt.close()

best_indiv = performance.loc[(performance["metric"] == "rocaucscore").values & np.array(["comb" not in x for x in performance["operator"]]) & (performance["method"] != "real").values].groupby(["method", "metric"])["value"].mean().idxmax()[0]

# plot performance comp groups
for metric in performance["metric"].unique():
    sns.barplot(data= performance.loc[(performance["metric"] == metric).values & np.array(["comb" in x and "+real" not in x for x in performance["operator"]])],
                x= "method",
                y= "value",
                errorbar= None,
                palette= "Blues")
    plt.ylabel(metrics_dict[metric])
    plt.savefig("./plots/comb_{}_performance.png".format(metric))
    #plt.show()
    plt.close()

    g = sns.barplot(data=performance.loc[(performance["metric"] == metric).values & np.array(
        ["comb" in x and "+real" not in x for x in performance["operator"]])],
                x="method",
                y="value",
                errorbar=None,
                palette="Blues",
                linewidth=1,
                edgecolor="black")
    apply_blackAndWhite_to_plot(g, plot_type='bar')
    plt.ylabel(metrics_dict[metric])
    plt.savefig("./plots/comb_{}_performance_bw.png".format(metric))
    #plt.show()
    plt.close()

best_comb = performance.loc[(performance["metric"] == "rocaucscore").values & np.array(["comb" in x and "+real" not in x for x in performance["operator"]]) & (performance["method"] != "real").values].groupby(["method", "metric"])["value"].mean().idxmax()[0]

# plot performance comp groups
for metric in performance["metric"].unique():
    sns.barplot(data= performance.loc[(performance["metric"] == metric).values & np.array(["comb+real" in x for x in performance["operator"]])],
                x= "method",
                y= "value",
                errorbar= None,
                palette= "Blues")
    plt.ylabel(metrics_dict[metric])
    plt.savefig("./plots/comb+real_{}_performance.png".format(metric))
    #plt.show()
    plt.close()

    g = sns.barplot(data=performance.loc[
        (performance["metric"] == metric).values & np.array(["comb+real" in x for x in performance["operator"]])],
                x="method",
                y="value",
                errorbar=None,
                palette="Blues",
                linewidth=1,
                edgecolor="black")
    apply_blackAndWhite_to_plot(g, plot_type='bar')
    plt.ylabel(metrics_dict[metric])
    plt.savefig("./plots/comb+real_{}_performance.png".format(metric))
    #plt.show()
    plt.close()

best_combreal = performance.loc[(performance["metric"] == "rocaucscore").values & np.array(["comb+real" in x for x in performance["operator"]]) & (performance["method"].isin(["real", "best-4"])).values].groupby(["method", "metric"])["value"].mean().idxmax()[0]

best_chart = ["real", best_indiv, best_comb, best_combreal]
real_data = performance.loc[performance["method"] == "real"]
indiv_data = performance.loc[(performance["method"] == best_indiv).values & np.array(["comb" not in x for x in performance["operator"]])]
indiv_data["method"] = "synthetic\n individual"
comb_data = performance.loc[(performance["method"] == best_comb).values & np.array(["comb" in x and "+real" not in x for x in performance["operator"]])]
comb_data["method"] = "synthetic\n shared"
combreal_data = performance.loc[(performance["method"] == best_combreal).values & np.array(["comb+real" in x for x in performance["operator"]])]
combreal_data["method"] = "synthetic shared\n + real"
best_chart_data = pd.concat([real_data,
                             indiv_data,
                             comb_data,
                             combreal_data],
                            axis=0)
for metric in performance["metric"].unique():
    ax_lower = best_chart_data.loc[(performance["metric"] == metric)].groupby("method")["value"].mean().min() - 0.1
    ax_higher = best_chart_data.loc[(performance["metric"] == metric)].groupby("method")["value"].mean().max() + 0.1
    if ax_lower > 0:
        ax_lower = round(ax_lower, 1)
    else:
        ax_lower = 0
    if ax_higher < 1:
        ax_higher = round(ax_higher, 1)
    else:
        ax_higher = 1
    fig, ax = plt.subplots()
    sns.barplot(data=best_chart_data.loc[(performance["metric"] == metric)],
                x="method",
                y="value",
                errorbar=None,
                palette="Blues",
                ax=ax)
    ax.set_ylim([0, 1])
    ax.set_ylabel(metrics_dict[metric])
    plt.tight_layout()
    plt.savefig("./plots/best_{}_performance.png".format(metric))
    #plt.show()
    plt.close()

    fig, ax = plt.subplots()
    sns.barplot(data=best_chart_data.loc[(performance["metric"] == metric)],
                x="method",
                y="value",
                errorbar=None,
                palette="Blues",
                ax=ax)
    ax.set_ylim([ax_lower, ax_higher])
    ax.set_ylabel(metrics_dict[metric])
    plt.tight_layout()
    plt.savefig("./plots/best_{}_performance_axlim.png".format(metric))
    #plt.show()
    plt.close()

    fig, ax = plt.subplots()
    sns.barplot(data=best_chart_data.loc[(performance["metric"] == metric)],
                x="method",
                y="value",
                errorbar=None,
                palette="Blues",
                ax=ax,
                linewidth=1,
                edgecolor="black")
    ax_lower = best_chart_data.loc[(performance["metric"] == metric)].groupby("method")["value"].mean().min() - 0.1
    ax.set_ylim([0, 1])
    ax.set_ylabel(metrics_dict[metric])
    apply_blackAndWhite_to_plot(ax, plot_type='bar')
    plt.tight_layout()
    plt.savefig("./plots/best_{}_performance_bw.png".format(metric))
    #plt.show()
    plt.close()

    fig, ax = plt.subplots()
    sns.barplot(data=best_chart_data.loc[(performance["metric"] == metric)],
                x="method",
                y="value",
                errorbar=None,
                palette="Blues",
                ax=ax,
                linewidth=1,
                edgecolor="black")
    ax.set_ylim([ax_lower, ax_higher])
    ax.set_ylabel(metrics_dict[metric])
    apply_blackAndWhite_to_plot(ax, plot_type='bar')
    plt.tight_layout()
    plt.savefig("./plots/best_{}_performance_axlim_bw.png".format(metric))
    # plt.show()
    plt.close()