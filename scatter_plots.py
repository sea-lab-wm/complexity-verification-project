from cmath import isnan
import os
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

def plot_data(df, xlabel, ylabel, file_prefix, save_file = True): 
    color = "#1f78b4"
    graph_label = dict(color='#101010', alpha=0.95)

    #draw the scatter plot, figsize is in inches
    ax1 = df.plot(kind='scatter', x='metric_value', y="#_of_warnings", s=30, c=color, figsize=(7, 5))

    # least squares polynomial fit (linear)
    z = numpy.polyfit(df['metric_value'], df["#_of_warnings"], 1)
    p = numpy.poly1d(z)

    # plot the line
    plt.plot(df['metric_value'], p(df['metric_value']), linewidth=1)
    
    #lot labels
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

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
    if save_file:
        plt.savefig(os.path.join("scatter_plots", str(file_prefix) + '.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)
    
    #clear plot
    plt.clf()

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
    
    #save the data, for sanitity check
    data.to_csv("scatter_plots/data2.csv", index=False)

    #group by tool, dataset, and metric
    data_by_tdm = data.groupby(["tool", "dataset", "metric"])

    # for each group
    for key, group in data_by_tdm:
        print(f"Processing {key!r}")

        # select columns of interest
        df = group[["metric_value", "#_of_warnings"]]
        df["metric_value"] = list(map(lambda x: x.item(), df.iloc[:,0]))
        df["#_of_warnings"] = list(map(lambda x: x.item(), df.iloc[:,1]))

        xlabel = f"{key[1]}_{key[2]}"
        ylabel = f"# of warnings ({key[0]})"
        file_prefix = f"{key[0]}_{key[1]}_{key[2]}"

        #plot_data(df, xlabel, ylabel, file_prefix)

        #break

    #-------


    data_by_tool = data.groupby("tool")
   

    for tool, tool_data in data_by_tool:
        print(f"Processing {tool!r}")
    
         #group by tool, dataset, and metric
        data_by_dm = tool_data.groupby(["dataset", "metric"])

        ylabel = f"# of warnings ({tool})"
        ylabel = f""
        file_prefix = f"{tool}"


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
                    wspace=0.1, 
                    hspace=0.3)

        for key, group in data_by_dm:
             # select columns of interest
            df = group[["metric_value", "#_of_warnings"]]
            df["metric_value"] = list(map(lambda x: x.item(), df.iloc[:,0]))
            df["#_of_warnings"] = list(map(lambda x: x.item(), df.iloc[:,1]))

            corr = df['metric_value'].corr(df["#_of_warnings"], method='kendall')

            if numpy.isnan(corr):
                continue

            xlabel = f"{key[0]}_{key[1]}"

            color = "#1f78b4"
            graph_label = dict(color='#101010', alpha=0.95, size = 14, weight='bold')

            ax1 = fig.add_subplot(Rows,Cols,Position[k])
            k = k + 1
            #draw the scatter plot, figsize is in inches
            df.plot(kind='scatter', x='metric_value', y="#_of_warnings", s=30
            , c=color
            , figsize=(7*Cols, 5*Rows)
            , ax = ax1
            )

            # least squares polynomial fit (linear)
            z = numpy.polyfit(df['metric_value'], df["#_of_warnings"], 1)
            p = numpy.poly1d(z)

            # plot the line
            plt.plot(df['metric_value'], p(df['metric_value']), linewidth=1, color="black")
            
            #lot labels
            plt.ylabel(ylabel)
            plt.xlabel(xlabel, fontweight='bold')

            #compute correlation measures
            
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

            gap = ((top - bottom) / 18)
            ax1.text(right - ((right - left) / 2.7), top - gap, 
                "Kendall's τ: " + format(corr, '.2f'), fontdict=graph_label)
            ax1.text(right - ((right - left) / 2.7), top - gap*2, 
                'r squared: ' + format(r_value ** 2, '.2f'), fontdict=graph_label)
            ax1.text(right - ((right - left) / 2.7), top - gap*3, 
                '# of points: ' + str(len(df)), fontdict=graph_label)

            plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
            

            #additional cosmetic changes
            #sns.despine()
            #plt.tight_layout()

            for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                                ax1.get_xticklabels() + ax1.get_yticklabels()):
                item.set_fontsize(14)

            
        #save plot
        #fig.tight_layout()
        plt.savefig(os.path.join("scatter_plots", str(file_prefix) + '.pdf'), dpi=300, 
        bbox_inches='tight', 
        pad_inches=0.1)
        
        #clear plot
        plt.clf()

        #break
