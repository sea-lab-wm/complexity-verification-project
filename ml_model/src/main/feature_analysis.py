"""
This is to merge final_feature_ds6.csv (our features), scalabrino_features_complete.csv (their code on our snippets) and scalabrino_raw_data.csv (Italian's provided data)
"""

import pandas as pd

def main():
    ROOT_PATH = "/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model"
    our_data = pd.read_csv(ROOT_PATH + "/final_features_ds6.csv") # extracted features from our code
    scalabrino_data = pd.read_csv(ROOT_PATH + "/scalabrino_features_complete.csv") # features extracted from Italian code
    scalabrino_raw_data = pd.read_csv(ROOT_PATH + "/scalabrino_raw_data.csv") # raw data provided by Italian
    
    ## append _our to the column names of our_data
    our_data.columns = [f"{col}_our" if col not in ['dataset_id', 'snippet_id', 'method_name', 'file'] else col for col in our_data.columns]

    ## append _code_Italian to the column names of scalabrino_data
    scalabrino_data.columns = [f"{col}_code_Italian" if col not in ['dataset_id', 'snippet_id', 'method_name', 'file'] else col for col in scalabrino_data.columns]

    ## append _data_Italian to the column names of scalabrino_raw_data
    scalabrino_raw_data.columns = [f"{col}_data_Italian" if col not in ['dataset_id', 'snippet_id', 'method_name', 'file'] else col for col in scalabrino_raw_data.columns]

    ## merge the dataframes
    merged_data1 = pd.merge(our_data, scalabrino_data, on=['dataset_id', 'snippet_id', 'method_name', 'file'], how='outer')

    ## merge the above merged dataframe with scalabrino_raw_data. Need to keep _our and _code_Italian columns with _data_Italian columns
    merged_data = pd.merge(merged_data1, scalabrino_raw_data, on=['dataset_id', 'snippet_id', 'method_name', 'file'], how='outer')

    ## Keep the order _our, _code_Italian, _data_Italian
    merged_data = merged_data.reindex(columns=sorted(merged_data.columns))

    ## take the columns 'dataset_id', 'snippet_id', 'method_name', 'file' to the front ## 
    cols = ['dataset_id', 'snippet_id', 'method_name', 'file']
    cols.extend([col for col in merged_data.columns if col not in cols])
    merged_data = merged_data[cols]

    ## drop developer_position_data_Italian	participant_id_data_Italian	snippet_signature_data_Italian	system_name_data_Italian columns
    merged_data.drop(columns=['developer_position_data_Italian', 'participant_id_data_Italian', 'snippet_signature_data_data_Italian', 'system_name_data_Italian'], inplace=True)

    # Save the merged data to a CSV file
    merged_data.to_csv(ROOT_PATH + '/merged_data.csv', index=False)

main()    



