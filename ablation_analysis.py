import os
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

if __name__ == "__main__":

    #read data
    data = pd.read_csv(f"data/raw_correlation_data.csv")

    #--------------

    data_indiv_tools = data[data["tool"] != "all_tools"]

    #keep only DS9 with no comments
    data_indiv_tools = data_indiv_tools[(data_indiv_tools.dataset != "9_gc") & (data_indiv_tools.dataset != "9_bc")]

    data_by_dsmt = data_indiv_tools.groupby(["dataset", "snippet", "metric"])
    
    #----------
    
    dict_df = {'dataset':[],
            'snippet':[],
            'metric':[],
            "metric_value":[],
            "#_of_warnings":[],
            "tool": []
        }
    
    avg_cor_data = pd.DataFrame(dict_df)
    
    #--------------
    #data generation for average analysis

    for key, group_df in data_by_dsmt:

        avg_warnings_value = numpy.average(group_df[["#_of_warnings"]])

        if len(group_df.metric_value.unique()) != 1 :
            raise Exception("the metric values should be the same and they are not")

        record = {'dataset': [key[0]],
            'snippet': [key[1]],
            'metric': [key[2]],
            "metric_value": [group_df.metric_value.values[0]], #just get one value, cause they should be the same
            "#_of_warnings": [avg_warnings_value],
            "tool": ["all_tools_avg"]
        }
        df_record = pd.DataFrame(record)
        avg_cor_data = pd.concat([avg_cor_data, df_record], ignore_index=True, axis=0)


    avg_cor_data = avg_cor_data.convert_dtypes()
    avg_cor_data.to_csv("data/raw_correlation_data_avg.csv", index=False)

    #---------------
    #data generation for ablation analysis

    ablation_cor_data = pd.DataFrame(dict_df)
    indiv_tools = data_indiv_tools.tool.unique()

    for tool in indiv_tools:
        for key, group_df in data_by_dsmt:

            group_df_notool = group_df[group_df["tool"] != tool]
            warnings_value = numpy.sum(group_df_notool[["#_of_warnings"]]).item()

            if len(group_df.metric_value.unique()) != 1 :
                raise Exception("the metric values should be the same and they are not")

            record = {'dataset': [key[0]],
                'snippet': [key[1]],
                'metric': [key[2]],
                "metric_value": [group_df.metric_value.values[0]], #just get one value, cause they should be the same
                "#_of_warnings": [warnings_value],
                "tool": [ f"no_{tool}"]
            }
            df_record = pd.DataFrame(record)
            ablation_cor_data = pd.concat([ablation_cor_data, df_record], ignore_index=True, axis=0)
    
    ablation_cor_data = ablation_cor_data.convert_dtypes()
    ablation_cor_data.to_csv("data/raw_correlation_data_ablation.csv", index=False)