import traceback

import pandas as pd


def process_cyclomatic_complexity_data(cyclomatic_complexity_df):
    """
    This function is used to aggregate cyclomatic complexity data of each method in a 
    file and return the sum of cyclomatic complexity of each file.
    """
    ## split the description by "/" and get the last element
    file_name = cyclomatic_complexity_df['File'].str.split('/').str[-1]
    ## replace the file name with the new file name
    cyclomatic_complexity_df['File'] = file_name
    ## split the description by "has a cyclomatic complexity of" and get cyclomatic complexity value
    cyclomatic_complexity_description = cyclomatic_complexity_df['Description'].str.split('has a cyclomatic complexity of').str[-1].str.split('.').str[0]
    ## replace the description with the new cyclomatic complexity value
    cyclomatic_complexity_df['Description'] = cyclomatic_complexity_description
    cyclomatic_complexity_df.rename(columns={"Description": "cyclomatic_complexity"}, inplace=True)
    ## convert the cyclomatic_complexity column to numeric
    cyclomatic_complexity_df['cyclomatic_complexity'] = pd.to_numeric(cyclomatic_complexity_df['cyclomatic_complexity'])
    ## merge similar File names and get the sum of cyclomatic complexity
    cyclomatic_complexity_df = cyclomatic_complexity_df.groupby(['File']).agg({'cyclomatic_complexity': 'sum'}).reset_index()
    ## remove the duplicate file names and keep the first one with the sum of cyclomatic complexity
    cyclomatic_complexity_df.drop_duplicates(subset=['File'], keep='first', inplace=True)
    return cyclomatic_complexity_df



if __name__ == "__main__":
    try:
        ## Feature data
        feature_data = pd.read_csv("ml_model/feature_data.csv")
        raw_feature_data = pd.read_csv("ml_model/raw_feature_data.csv")
        
        ## Verification Tool warning data
        metric_data = pd.read_csv("ml_model/metric_data.csv")
        
        ## LOC data
        loc_data = pd.read_csv("ml_model/loc_data.csv")
        raw_loc_data = pd.read_csv("ml_model/raw_loc_data.csv")
        
        ## Cyclomatic Complexity data
        cyclomatic_complexity_data = pd.read_csv("ml_model/cyclomatic_complexity_data.csv")
        raw_cyclomatic_complexity_data = pd.read_csv("ml_model/raw_cyclomatic_complexity_data.csv")
    except FileNotFoundError as e:
        print(traceback.format_exc())
        quit()

    feature_data.reset_index()
    raw_feature_data.reset_index()
    metric_data.reset_index()
    loc_data.reset_index()
    raw_loc_data.reset_index()
    cyclomatic_complexity_data.reset_index()
    raw_cyclomatic_complexity_data.reset_index()

    ## Do preprocessing on cyclomatic_complexity_data
    cyclomatic_complexity_data = process_cyclomatic_complexity_data(cyclomatic_complexity_data)
    ## Merge cyclomatic_complexity column in the cyclomatic_complexity_data with feature_data using file column
    feature_data = pd.merge(feature_data, cyclomatic_complexity_data[["File", "cyclomatic_complexity"]], left_on="file", right_on="File", how="left")
    ## drop the File column
    feature_data.drop(columns=["File"], inplace=True)

    ## Merge file column in feature_data with Filename column in loc_data
    feature_data = pd.merge(feature_data, loc_data[["Filename", "Code"]], left_on="file", right_on="Filename", how="left")
    feature_data.drop(columns=["Filename"], inplace=True)
    feature_data.rename(columns={"Code": "LOC"}, inplace=True)

    raw_cyclomatic_complexity_data = process_cyclomatic_complexity_data(raw_cyclomatic_complexity_data)
    raw_feature_data = pd.merge(raw_feature_data, raw_cyclomatic_complexity_data[["File", "cyclomatic_complexity"]], left_on="file", right_on="File", how="left")
    raw_feature_data.drop(columns=["File"], inplace=True)

    ## Merge file column in raw_feature_data with Filename column in raw_loc_data
    raw_feature_data = pd.merge(raw_feature_data, raw_loc_data[["Filename", "Code"]], left_on="file", right_on="Filename", how="left")
    raw_feature_data.drop(columns=["Filename"], inplace=True)
    raw_feature_data.rename(columns={"Code": "LOC"}, inplace=True)

    # Merge the two data tables
    all_df = pd.merge(metric_data, feature_data, on=["dataset_id", "snippet_id"], how="left")
    all_raw_df = pd.merge(metric_data, raw_feature_data, on=["dataset_id", "snippet_id"], how="left")

    if len(all_df.index) != 14496:
        raise Exception("join_tables: length mismatch.")
    
    if len(all_raw_df.index) != 14496:
        raise Exception("join_tables: length mismatch.")

    all_df.to_csv("ml_model/ml_table.csv", index=False)
    all_raw_df.to_csv("ml_model/ml_raw_table.csv", index=False)
