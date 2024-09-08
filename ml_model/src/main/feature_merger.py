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

def main(ROOT_PATH):
    # ROOT_PATH="/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model"
    try:
        ## Feature data from our code ##
        feature_data = pd.read_csv(ROOT_PATH + "/feature_data.csv")
        
        # Feature data from scalabrino
        scalabrino_data  = pd.read_csv(ROOT_PATH + "/scalabrino_features_complete_classes_packages.csv")

        ## Read NMI data ##
        nmi_data = pd.read_csv(ROOT_PATH + "/nmi_data.csv")

        ## read NM data ##
        nm_data = pd.read_csv(ROOT_PATH + "/nm_data.csv")

        ## Read ITID data ##
        itid_data = pd.read_csv(ROOT_PATH + "/itid.csv")

        ## Read LOC data ##
        loc_data = pd.read_csv(ROOT_PATH + "/raw_loc_data.csv")

        raw_cyclomatic_complexity_data = pd.read_csv(ROOT_PATH + "/raw_cyclomatic_complexity_data.csv")

        ## read scalaribo_raw_data to get targets ##
        scalaribo_raw_data = pd.read_csv(ROOT_PATH + "/understandability_with_warnings_with_targets.csv")

        ## Verification Tool warning data
        metric_data = pd.read_csv(ROOT_PATH + "/metric_data.csv")


    except FileNotFoundError as e:
        print(traceback.format_exc())
        quit()

    
    
    feature_data.reset_index()
    scalabrino_data.reset_index()
    nmi_data.reset_index()
    nm_data.reset_index()
    itid_data.reset_index()
    loc_data.reset_index()
    raw_cyclomatic_complexity_data.reset_index()
    scalaribo_raw_data.reset_index()

    ## Do preprocessing on cyclomatic_complexity_data
    cyclomatic_complexity_data = process_cyclomatic_complexity_data(raw_cyclomatic_complexity_data)
    
    ## Merge cyclomatic_complexity column in the cyclomatic_complexity_data with feature_data using file column
    # scalabrino_data = pd.merge(scalabrino_data, cyclomatic_complexity_data[["File", "cyclomatic_complexity"]], left_on="file", right_on="File", how="left")
    all_data = pd.merge(feature_data, cyclomatic_complexity_data[["File", "cyclomatic_complexity"]], left_on="file", right_on="File", how="left")
    ## drop the File column
    all_data.drop(columns=["File"], inplace=True)

    ## change cyclomatic_complexity column name to Cyclomatic complexity
    all_data.rename(columns={"cyclomatic_complexity": "Cyclomatic complexity"}, inplace=True)
    
    ## Merge scalabrino_data with feature_data ##
    ## Pick MIDQ (avg), MIDQ (max) and MIDQ (min) columns from feature_data
    # all_data = pd.merge(scalabrino_data, feature_data[["MIDQ (avg)", "MIDQ (max)", "MIDQ (min)", "#statements", "#parameters", "#nested blocks (avg)",  "file"]], left_on="file", right_on="file", how="left")
    
    ## Merge NMI data with all_data ##
    all_data = pd.merge(all_data, nmi_data[["Filename", "NMI (avg)", "NMI (max)"]], left_on="file", right_on="Filename", how="left")
    all_data.drop(columns=["Filename"], inplace=True)

    ## Merge ITID data with all_data ##
    all_data = pd.merge(all_data, itid_data[["Filename", "ITID (avg)", "ITID (min)"]], left_on="file", right_on="Filename", how="left")
    all_data.drop(columns=["Filename"], inplace=True)

    ## Merge NM data with all_data ##
    all_data = pd.merge(all_data, nm_data[["Filename", "NM (avg)", "NM (max)"]], left_on="file", right_on="Filename", how="left")
    all_data.drop(columns=["Filename"], inplace=True)

    ## Merge Code column with all_data ##
    all_data = pd.merge(all_data, loc_data[["Filename", "Code"]], left_on="file", right_on="Filename", how="left")
    all_data.drop(columns=["Filename"], inplace=True)
    ## rename the column to LOC
    all_data.rename(columns={"Code": "LOC"}, inplace=True)

    # ## take #assignments (dft), #characters (max), #commas (dft), #comments (dft), Comments (Visual X),Comments (Visual Y), #comparisons (dft), #conditionals (dft), #identifiers (dft), Identifiers (Visual X), Identifiers (Visual Y), #keywords (dft), Keywords (Visual X), Keywords (Visual Y), Literals (Visual X), Literals (Visual Y), #loops (dft), #numbers (dft), Numbers (Visual X), Numbers (Visual Y), Operators (Visual X), Operators (Visual Y), #operators (dft), #parenthesis (dft), #periods (dft), #spaces (dft), Strings (Visual X), Strings (Visual Y),  Indentation length (dft), Line length (dft), #aligned blocks, Extent of aligned blocks, TC (avg), TC (min), TC (max), Readability, CR
    ## from scalabrino_data and merge it with all_data
    all_data = pd.merge(all_data, scalabrino_data[["dataset_id","snippet_id","method_name","file","#assignments (dft)","#characters (max)","#commas (dft)","#comments (dft)","Comments (Visual X)","Comments (Visual Y)","#comparisons (dft)","#conditionals (dft)","#identifiers (dft)","Identifiers (Visual X)","Identifiers (Visual Y)","#keywords (dft)", "Literals (Visual X)", "Literals (Visual Y)", "#loops (dft)", "#numbers (dft)", "Numbers (Visual X)", "Numbers (Visual Y)", "Operators (Visual X)", "Operators (Visual Y)", "#operators (dft)", "#parenthesis (dft)", "#periods (dft)", "#spaces (dft)", "Strings (Visual X)", "Strings (Visual Y)", "Indentation length (dft)", "Line length (dft)", "#aligned blocks", "Extent of aligned blocks", "TC (avg)", "TC (min)", "TC (max)", "Readability", "CR", "Keywords (Visual X)", "Keywords (Visual Y)"]], on=["dataset_id", "snippet_id","method_name","file"], how="left")

    # # Merge the features with metric data ##
    all_df = pd.merge(metric_data, all_data, on=["dataset_id", "snippet_id"], how="left")
    # all_df = all_data

    if len(all_df.index) != 14494:
        raise Exception("join_tables: length mismatch.")
    

    # all_df.to_csv("ml_model/DS_3_ml_table.csv", index=False)
    all_df.to_csv(ROOT_PATH + "/final_features.csv", index=False)
    print ("Final features saved to final_features.csv - For all the datasets")
    ## filter only the dataset_id=3 data and save it to a file
    all_df[all_df['dataset_id'] == '3'].to_csv(ROOT_PATH + "/final_features_ds3.csv", index=False)
    print ("Final features saved to final_features_ds3.csv - For dataset_id=3")
    all_df[all_df['dataset_id'] == '6'].to_csv(ROOT_PATH + "/final_features_ds6.csv", index=False)
    print ("Final features saved to final_features_ds6.csv - For dataset_id=6")

    
    data_ds6 = pd.read_csv(ROOT_PATH + "/final_features_ds6.csv")
    ## merge understandability_with_warnings_with_targets.csv with final_features_ds6.csv on system_name,developer_position,PE gen,PE spec (java)
    ## rename file_name to snipper_id in scalaribo_raw_data
    scalaribo_raw_data.rename(columns={"file_name": "snippet_id"}, inplace=True)
    data_ds6 = pd.merge(data_ds6, scalaribo_raw_data[["PBU", "ABU", "ABU50", "BD", "BD50", "AU", "snippet_id", "developer_position", "PE gen", "PE spec (java)"]], on=["snippet_id", "developer_position", "PE gen", "PE spec (java)"], how="left")
    

    # # remove duplicate rows by keeping the first one. consider rows which has same snippet_id,person_id,developer_position,PE gen,PE spec (java)
    data_ds6.drop_duplicates(subset=["person_id", "snippet_id", "developer_position", "PE gen", "PE spec (java)"], keep='first', inplace=True)
    
    ## add two specific rows to the data_ds6 ##
    ## search person_id=69 snippet_id=1-SpringBatch from the data_ds6 and add it to the data_ds6
    spring_batch = data_ds6[(data_ds6['person_id'] == 69) & (data_ds6['snippet_id'] == "1-SpringBatch") & (data_ds6['developer_position'] == 4) & (data_ds6['PE gen'] == 10) & (data_ds6['PE spec (java)'] == 2)]
    data_ds6 = pd.concat([data_ds6, spring_batch], ignore_index=True)

    ## search person_id=69 snippet_id=2-SpringBatch from the data_ds6 and add it to the data_ds6
    spring_batch = data_ds6[(data_ds6['person_id'] == 69) & (data_ds6['snippet_id'] == "2-SpringBatch") & (data_ds6['developer_position'] == 4) & (data_ds6['PE gen'] == 10) & (data_ds6['PE spec (java)'] == 2)]
    data_ds6 = pd.concat([data_ds6, spring_batch], ignore_index=True)

    k9_batch=data_ds6[(data_ds6['person_id'] == 58) & (data_ds6['snippet_id'] == "2-K9") & (data_ds6['developer_position'] == 2) & (data_ds6['PE gen'] == 4) & (data_ds6['PE spec (java)'] == 2)]
    data_ds6 = pd.concat([data_ds6, k9_batch], ignore_index=True)

    opencmscore_batch=data_ds6[(data_ds6['person_id'] == 11) & (data_ds6['snippet_id'] == "3-OpenCMSCore") & (data_ds6['developer_position'] == 1) & (data_ds6['PE gen'] == 8) & (data_ds6['PE spec (java)'] == 3)]
    data_ds6 = pd.concat([data_ds6, opencmscore_batch], ignore_index=True)

    myexpenses_batch=data_ds6[(data_ds6['person_id'] == 36) & (data_ds6['snippet_id'] == "4-MyExpenses") & (data_ds6['developer_position'] == 1) & (data_ds6['PE gen'] == 2) & (data_ds6['PE spec (java)'] == 2)]
    data_ds6 = pd.concat([data_ds6, myexpenses_batch], ignore_index=True)

    data_ds6.to_csv(ROOT_PATH + "/final_features_ds6.csv", index=False)

main("/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model")    