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

    ## rename the column to cyclomatic_complexity
    cyclomatic_complexity_df.rename(columns={"Description": "cyclomatic_complexity"}, inplace=True)
    ## split the cyclomatic complexity description by "has a cyclomatic complexity of" and get cyclomatic complexity value
    cyclomatic_complexity = cyclomatic_complexity_df['cyclomatic_complexity'].str.split('has a cyclomatic complexity of').str[-1].str.split('.').str[0]
    ## replace the column with the new cyclomatic complexity data
    cyclomatic_complexity_df['cyclomatic_complexity'] = cyclomatic_complexity
    ## convert the cyclomatic_complexity column to numeric
    cyclomatic_complexity_df['cyclomatic_complexity'] = pd.to_numeric(cyclomatic_complexity_df['cyclomatic_complexity'])
    
    ## merge similar File names and get the sum of cyclomatic complexity
    cyclomatic_complexity_df = cyclomatic_complexity_df.groupby(['File']).agg({'cyclomatic_complexity': 'sum'}).reset_index()
    ## remove the duplicate file names and keep the first one with the sum of cyclomatic complexity
    cyclomatic_complexity_df.drop_duplicates(subset=['File'], keep='first', inplace=True)
    return cyclomatic_complexity_df

if __name__ == "__main__":
    try:
        ## Feature data from scalabrino
        feature_data = pd.read_csv("complexity-verification-project/ml_model/feature_data.csv")
        
        ## Feature data from scalabrino
        scalabrino_data  = pd.read_csv("complexity-verification-project/ml_model/scalabrino_features_complete.csv")

        raw_cyclomatic_complexity_data = pd.read_csv("complexity-verification-project/ml_model/raw_cyclomatic_complexity_data.csv")

    except FileNotFoundError as e:
        print(traceback.format_exc())
        quit()

    scalabrino_data.reset_index()
    raw_cyclomatic_complexity_data.reset_index()

    ## Do preprocessing on cyclomatic_complexity_data
    cyclomatic_complexity_data = process_cyclomatic_complexity_data(raw_cyclomatic_complexity_data)
    
    ## Merge cyclomatic_complexity column in the cyclomatic_complexity_data with feature_data using file column
    scalabrino_data = pd.merge(scalabrino_data, cyclomatic_complexity_data[["File", "cyclomatic_complexity"]], left_on="file", right_on="File", how="left")
    ## drop the File column
    scalabrino_data.drop(columns=["File"], inplace=True)

    ## change cyclomatic_complexity column name to Cyclomatic complexity
    scalabrino_data.rename(columns={"cyclomatic_complexity": "Cyclomatic complexity"}, inplace=True)
    
    ## Merge scalabrino_data with feature_data ##
    ## Pick MIDQ (avg), MIDQ (max) and MIDQ (min) columns from feature_data
    scalabrino_data = pd.merge(scalabrino_data, feature_data[["MIDQ (avg)", "MIDQ (max)", "MIDQ (min)", "#statements", "#parameters", "#nested blocks (avg)",  "file"]], left_on="file", right_on="file", how="left")
 

    if len(scalabrino_data.index) != 231:
        raise Exception("join_tables: length mismatch.")
    

    scalabrino_data.to_csv("complexity-verification-project/ml_model/final_features.csv", index=False)
