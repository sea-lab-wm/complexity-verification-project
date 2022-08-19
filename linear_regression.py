import os
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

# -------------------
if __name__ == "__main__":
    # -----------------

    # create output folder
    os.makedirs("scatter_plots", exist_ok=True)

    # -----------------

    #read data
    data = pd.read_csv(f"data/raw_correlation_data.csv")
    
    #read data for metric types
    correlation_data = pd.read_excel(
        "data/correlation_analysis.xlsx", sheet_name="all_tools")

    #select metrics of interest
    ds_metrics = correlation_data[["dataset_id", "metric", "metric_type"]]
    #ds_metrics.rename(columns={"dataset_id": "dataset"}, inplace=True)

    #string converstion, there should be other ways to do it
    ds_metrics["dataset"] = list(map(lambda x: str(x), ds_metrics.iloc[:,0]))

    #convert to best possible datatypes
    data = data.convert_dtypes()
    ds_metrics = ds_metrics.convert_dtypes(infer_objects=False)

    #assign metric type to data
    data = data.merge(ds_metrics, on=['dataset','metric'], how='left')
    
    # dsm_by_metric_type = ds_metrics.groupby("metric_type")

    # for mtype, mtype_ds_metrics in dsm_by_metric_type:
    #     print(f"First 2 entries for {mtype!r}")
    #     print("------------------------")
    #     print(mtype_ds_metrics.head(2), end="\n\n")

    #     #data
    #     #distro = distro.merge(counts_by_dataset, on='dataset', how='left')
    # print(ds_metrics.dtypes)
    # #print(data)
    # print("------")
    # print(data.dtypes)
    # print()
    data.to_csv("scatter_plots/data2.csv", index=False)

    data_by_tdm = data.groupby(["tool", "dataset", "metric"])
    
    color = "#1f78b4"
    graph_label = dict(color='#101010', alpha=0.95)

    for key, group in data_by_tdm:
        print(f"Processing {key!r}")

        # select columns of interest
        df = group[["metric_value", "#_of_warnings"]]
        df["metric_value"] = list(map(lambda x: x.item(), df.iloc[:,0]))
        df["#_of_warnings"] = list(map(lambda x: x.item(), df.iloc[:,1]))

        #draw the scatter plot, figsize is in inches
        ax1 = df.plot(kind='scatter', x='metric_value', y="#_of_warnings", s=30, c=color, figsize=(7, 5))

        # least squares polynomial fit (linear)
        z = numpy.polyfit(df['metric_value'], df["#_of_warnings"], 1)
        p = numpy.poly1d(z)

        # plot the line
        plt.plot(df['metric_value'], p(df['metric_value']), linewidth=1)
        
        #lot labels
        plt.ylabel(f"# of warnings ({key[0]})")
        plt.xlabel(f"{key[1]}_{key[2]}")

        #compute correlation measures
        corr = df['metric_value'].corr(df["#_of_warnings"], method='kendall')
        #print('metric_value: ~  "#_of_warnings"')
        #print('Kendall corr:', corr)

        #linear least-squares regression
        #r_value = the Pearson correlation coefficient. The square of rvalue is equal to the coefficient of determination.
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df['metric_value'], df["#_of_warnings"])
        #print('r squared:', r_value ** 2)
        
        #add the correlation to the plots
        left, right = plt.xlim()
        bottom, top = plt.ylim()
        # text(x, y, ...)
        ax1.text(left + ((right - left) / 40), bottom + ((top - bottom) / 15), "Kendall's τ: " + format(corr, '.2f'), fontdict=graph_label)
        ax1.text(left + ((right - left) / 40), bottom + ((top - bottom) / 40), 'r squared: ' + format(r_value ** 2, '.2f'), fontdict=graph_label)

        #additional cosmetic changes
        sns.despine()
        plt.tight_layout()
        
        #save plot
        prefix = f"{key[0]}_{key[1]}_{key[2]}"
        plt.savefig(os.path.join("scatter_plots", str(prefix) + '.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)
        
        #clear plot
        plt.clf()

        #break

    
