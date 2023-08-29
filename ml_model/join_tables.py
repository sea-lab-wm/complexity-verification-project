import traceback

import pandas as pd

if __name__ == "__main__":
    try:
        feature_data = pd.read_csv("ml_model/feature_data.csv")
        raw_feature_data = pd.read_csv("ml_model/raw_feature_data.csv")
        metric_data = pd.read_csv("ml_model/metric_data.csv")
        loc_data = pd.read_csv("ml_model/loc_data.csv")
        raw_loc_data = pd.read_csv("ml_model/raw_loc_data.csv")
    except FileNotFoundError as e:
        print(traceback.format_exc())
        quit()

    feature_data.reset_index()
    raw_feature_data.reset_index()
    metric_data.reset_index()
    loc_data.reset_index()
    raw_loc_data.reset_index()

    # Merge file column in feature_data with Filename column in loc_data
    feature_data = pd.merge(feature_data, loc_data[["Filename", "Code"]], left_on="file", right_on="Filename", how="left")
    # Drop the Filename column
    feature_data.drop(columns=["Filename"], inplace=True)
    # rename Code column to loc
    feature_data.rename(columns={"Code": "LOC"}, inplace=True)

    # Merge file column in raw_feature_data with Filename column in raw_loc_data
    raw_feature_data = pd.merge(raw_feature_data, raw_loc_data[["Filename", "Code"]], left_on="file", right_on="Filename", how="left")
    # Drop the Filename column
    raw_feature_data.drop(columns=["Filename"], inplace=True)
    # rename Code column to loc
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
