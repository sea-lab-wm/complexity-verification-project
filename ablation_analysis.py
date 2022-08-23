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

    data_by_dsmt = data_indiv_tools.groupby(["dataset", "snippet", "metric"])
    
    #----------
    
    dict = {'dataset':[],
            'snippet':[],
            'metric':[],
            "metric_value":[],
            "#_of_warnings":[],
            "tool": []
        }
    
    avg_cor_data = pd.DataFrame(dict)
    
    #--------------
    #average analysis

    for key, group_df in data_by_dsmt:

        avg_value = numpy.average(group_df[["#_of_warnings"]])

        if len(group_df.metric_value.unique()) != 1 :
            raise Exception("the metric values should be the same and they are not")

        record = {'dataset': [key[0]],
            'snippet': [key[1]],
            'metric': [key[2]],
            "metric_value": [group_df.metric_value.values[0]], #just get one value, cause they should be the same
            "#_of_warnings": [avg_value],
            "tool": ["all_tools_avg"]
        }
        df_record = pd.DataFrame(record)
        avg_cor_data = pd.concat([avg_cor_data, df_record], ignore_index=True, axis=0)


    avg_cor_data = avg_cor_data.convert_dtypes()
    avg_cor_data.to_csv("data/raw_correlation_data_avg.csv", index=False)

    #---------------
    #ablation analysis