import pandas as pd
from utils import configs_TASK2_DS3 as configs_ds3
from utils import configs_TASK2_DS6 as configs_ds6

## read merged_ds3.csv
data_ds3 = pd.read_csv(configs_ds3.ROOT_PATH + "/" + configs_ds3.DATA_PATH)

## read merged_ds6.csv
data_ds6 = pd.read_csv(configs_ds6.ROOT_PATH + "/" + configs_ds6.DATA_PATH)

## Filter target="AU" from data_ds6
data_ds6 = data_ds6[data_ds6["target"] == "AU"]

## Combine data_ds3 and data_ds6
combined_data = pd.concat([data_ds3, data_ds6], ignore_index=True)

## rename the values in target column to "(s2>s1)relative_comprehensibility"
combined_data["target"] = "(s2>s1)relative_comprehensibility"

## Save the combined data
combined_data.to_csv(configs_ds6.ROOT_PATH + "/data/combined_ds.csv", index=False)
