import traceback

import pandas as pd

if __name__ == "__main__":
    try:
        feature_data = pd.read_csv("ml_model/feature_data.csv")
        metric_data = pd.read_csv("ml_model/metric_data.csv")
    except FileNotFoundError as e:
        print(traceback.format_exc())
        quit()

    feature_data.reset_index()
    metric_data.reset_index()

    # Merge the two data tables
    all_df = pd.merge(metric_data, feature_data, on=["dataset_id", "snippet_id"], how="left")

    if len(all_df.index) != 14496:
        raise Exception("join_tables: length mismatch.")

    all_df.to_csv("ml_model/ml_table.csv", index=False)