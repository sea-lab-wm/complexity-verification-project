from cmath import isnan
import os
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import scipy.stats as scpy

#Deprecated
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


def no_outliers(data):

    # calculate interquartile range
    q25, q75 = numpy.percentile(data, 25), numpy.percentile(data, 75)
    iqr = q75 - q25

    # calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    
    # true if x is NOT an outlier, false otherwise
    return list(map(lambda x: x > lower and x < upper, data))

# -------------------
if __name__ == "__main__":
    # -----------------

    remove_outliers = False
    input_correlation_excel_file = "data/correlation_analysis.xlsx" # this file is only used to extract metric types
    suffix_files = "_no_outliers" if remove_outliers else ""

    timeout_approach = "_timeout_max"
    # timeout_approach = "_timeout_remove"
    # timeout_approach = "_timeout_zero"

    #aggregate # of warnings (ablation)
    input_file = f"data/raw_correlation_data_ablation{timeout_approach}.csv"
    output_folder = f"scatter_plots_ablation{suffix_files}{timeout_approach}"

    #aggregate # of warnings (sum)
    #input_file = f"data/raw_correlation_data{timeout_approach}.csv"
    #output_folder = f"scatter_plots{suffix_files}{timeout_approach}"

    #aggregate # of warnings (avg)
    #input_file = f"data/raw_correlation_data_avg{timeout_approach}.csv"
    #output_folder = f"scatter_plots_avg{suffix_files}{timeout_approach}"

    # -----------------

    # create output folder
    os.makedirs(output_folder, exist_ok=True)

    #read data
    data = pd.read_csv(input_file)

    #read data for metric types
    correlation_data = pd.read_excel(input_correlation_excel_file, sheet_name="all_tools")

    #select metrics of interest
    ds_metrics = correlation_data[["dataset_id", "metric", "metric_type", "expected_cor"]]
    #ds_metrics.rename(columns={"dataset_id": "dataset"}, inplace=True)

    #string converstion, there should be other ways to do it
    ds_metrics["dataset"] = list(map(lambda x: str(x), ds_metrics.iloc[:,0]))

    #convert to best possible datatypes
    data = data.convert_dtypes()
    ds_metrics = ds_metrics.convert_dtypes(infer_objects=False)

    #keep only DS9 with no comments
    data = data[(data.dataset != "9_gc") & (data.dataset != "9_bc")]

    #assign metric type to data
    data = data.merge(ds_metrics, on=['dataset','metric'], how='left')
    
    #----------------------
    #code that generates the scatter plots in individual files

    # #save the data, for sanitity check
    data.to_csv(output_folder + "/data2.csv", index=False)

    # #group by tool, dataset, and metric
    # data_by_tdm = data.groupby(["tool", "dataset", "metric"])

    # # for each group
    # for key, group in data_by_tdm:
    #     print(f"Processing {key!r}")

    #     # select columns of interest
    #     df = group[["metric_value", "#_of_warnings"]]
    #     df["metric_value"] = list(map(lambda x: x.item(), df.iloc[:,0]))
    #     df["#_of_warnings"] = list(map(lambda x: x.item(), df.iloc[:,1]))

    #     xlabel = f"{key[1]}_{key[2]}"
    #     ylabel = f"# of warnings ({key[0]})"
    #     file_prefix = f"{key[0]}_{key[1]}_{key[2]}"

    #     #plot_data(df, xlabel, ylabel, file_prefix)

    #     #break

    #----------------------

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

        #---------------

        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.3)

        #---------------

          
        dict_df = {
                'metric':[],
                'dataset_id':[],
                'metric_type':[],
                'expected_cor':[],
                "num_snippets_for_correlation":[],
                "kendalls_tau":[],
                "kendalls_p_value": [],
                "expected_cor?": [],
                "cor_intepretation": [],
                "stat_significant?": []
            }

        tool_cor_data = pd.DataFrame(dict_df)

        for key, group in data_by_dm:
             # select columns of interest
            df = group[["metric_value", "#_of_warnings", "metric_type", "expected_cor"]]

            df["metric_value"] = list(map(lambda x: x.item(), df.iloc[:,0]))
            df["#_of_warnings"] = list(map(lambda x: x.item(), df.iloc[:,1]))
            
            #------------------------

            if remove_outliers:
                df = df[no_outliers(df["metric_value"])]

            #------------------------

            #corr = df['metric_value'].corr(df["#_of_warnings"], method='kendall')

            corr, p_value = scpy.kendalltau(df['metric_value'], df["#_of_warnings"])

            #-----------------------
            dataset = key[0]
            metric = key[1]
            #all array values should be the same for each property
            metric_type = df.metric_type.values[0]
            expected_cor = df.expected_cor.values[0]
            expected_cor_short = "neg" if "negative" == expected_cor else "pos"
            num_snippets_for_correlation = len(df)
            kendalls_tau = corr
            kendalls_p_value = p_value
            expected_cor_test= "" if numpy.isnan(corr) else \
                               "*" if "negative" == expected_cor and corr < 0 else \
                               "*" if "positive" == expected_cor and corr > 0 else ""
            cor_intepretation= "" if numpy.isnan(corr) else \
                                "none" if abs(corr) >=0 and abs(corr) < 0.1 else \
                                "small" if abs(corr) >=0.1 and abs(corr) < 0.3 else \
                                "medium" if abs(corr) >=0.3 and abs(corr) < 0.5 else "large"
            #one star -> p_value <= 0.05, two stars -> p_value <= 0.01                    
            stat_significant = list(map(lambda x: ''.join(['*' for t in [0.01,0.05] if x<=t]), [p_value]))[0]

            record = {
                'metric':[metric],
                'dataset_id':[dataset],
                'metric_type':[metric_type],
                'expected_cor':[expected_cor],
                "num_snippets_for_correlation":[num_snippets_for_correlation],
                "kendalls_tau":[kendalls_tau],
                "kendalls_p_value": [kendalls_p_value],
                "expected_cor?": [expected_cor_test],
                "cor_intepretation": [cor_intepretation],
                "stat_significant?": [stat_significant]
            }
            df_record = pd.DataFrame(record)
            tool_cor_data = pd.concat([tool_cor_data, df_record], ignore_index=True, axis=0)  

            #------------------------

            if numpy.isnan(corr):
                continue

            #--------------------

            xlabel = f"{dataset}_{metric}"

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
            h_gap = ((right - left) / 2.4)
            ax1.text(right - h_gap, top - gap, 
                "Kendall's τ: " + format(corr, '.2f') + f"{stat_significant}", fontdict=graph_label)
            ax1.text(right - h_gap, top - gap*2, 
                f"τ's interpr: {cor_intepretation}", fontdict=graph_label)  
            ax1.text(right - h_gap, top - gap*3, 
                f"Expected cor: {expected_cor_short}{expected_cor_test}", fontdict=graph_label)    
            ax1.text(right - h_gap, top - gap*4, 
                '# of points: ' + str(num_snippets_for_correlation), fontdict=graph_label)
            ax1.text(right - h_gap, top - gap*5, 
                'r squared: ' + format(r_value ** 2, '.2f'), fontdict=graph_label)

            plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
            

            #additional cosmetic changes
            #sns.despine()
            #plt.tight_layout()

            for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                                ax1.get_xticklabels() + ax1.get_yticklabels()):
                item.set_fontsize(14)

        #save correlation values
        tool_cor_data = tool_cor_data.convert_dtypes()
        tool_cor_data.to_csv(os.path.join(output_folder, str(tool) + '_corr_data.csv'), index=False)


        #save plot
        #fig.tight_layout()
        plt.savefig(os.path.join(output_folder, str(file_prefix) + '.pdf'), dpi=300, 
        bbox_inches='tight', 
        pad_inches=0.1)
        
        #clear plot
        plt.clf()

        #break
