import os
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns


if __name__ == "__main__":
    # -----------------

    # create output folder
    os.makedirs("box_plots", exist_ok=True)

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

    print(data.loc[:,"metric_value"])
    data["metric_value"] = list(map(lambda x: x.item(), data.loc[:,"metric_value"]))
    data["#_of_warnings"] = list(map(lambda x: x.item(), data.loc[:,"#_of_warnings"]))

    data = data[data["tool"] == "all_tools"]

    #ax1 = data.boxplot(["metric_value", "#_of_warnings"], by =["dataset", "metric"])
    #plt.savefig('box_plots/metrics.pdf')  
    

    #group by dataset and metric
    data_by_dm = data.groupby(["dataset", "metric"])

    #---------------

    # Subplots are organized in a Rows x Cols Grid
    # Tot and Cols are known

    Tot = len(data_by_dm.groups.keys())
    Cols = 4

    # Compute Rows required

    Rows = Tot // Cols 

    #     EDIT for correct number of rows:
    #     If one additional row is necessary -> add one:

    if Tot % Cols != 0:
        Rows += 1

    # Create a Position index

    Position = range(1,Tot + 1)

    fig = plt.figure(1)
    k = 0

    plt.subplots_adjust(left=0.1,
                 bottom=0.1, 
                 right=0.9, 
                 top=0.9, 
                 wspace=0.4, 
                 hspace=0.3)

    # for each group
    for key, group in data_by_dm:
        print(f"Processing {key!r}")

        # select columns of interest
        df = group[["metric_value", "#_of_warnings"]]
        #df["metric_value"] = list(map(lambda x: x.item(), df.iloc[:,0]))
        #df["#_of_warnings"] = list(map(lambda x: x.item(), df.iloc[:,1]))

        xlabel = f"{key[0]}_{key[1]}"
        ax1 = fig.add_subplot(Rows,Cols,Position[k])
        k = k + 1
        plt.xlabel(xlabel, fontweight='bold')
        df.plot(kind='box', column = "metric_value", ax=ax1, grid = False, showmeans=True, showfliers=True
             ,figsize=(3*Cols, 4*Rows)
             )

        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                            ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(12)

    
    plt.savefig(f'box_plots/metrics.pdf', 
    bbox_inches='tight', 
        pad_inches=0.1)  
    
    #clear plot
    plt.clf()


    #------------------


