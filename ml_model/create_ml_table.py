import pandas as pd

def get_num_warnings(file):
    """Returns a dataframe where each entry is a dataset, snippet, and number of warnings from a single tool."""

    dict_df = {
        "dataset_id":[],
        "snippet_id":[],
        "num_warnings":[]
    }

    data = pd.DataFrame(dict_df)

    df = pd.read_csv(file)

    num_warnings = df.sum(axis=1, numeric_only=True).tolist()

    for index, row in df.iterrows():
        dataset_id = row["Snippet"].split("--")[0].strip()
        snippet_id = row["Snippet"].split("--")[1].strip()
        num_warning = num_warnings[index]

        record = {
            "dataset_id": [dataset_id],
            "snippet_id": [snippet_id],
            "num_warnings": [num_warning]
        }
        df_record = pd.DataFrame(record)
        data = pd.concat([data, df_record], ignore_index=True, axis=0)

    return data

if __name__ == "__main__":
    warning_data_files = [
        "data/checker_framework_data.csv",
        "data/typestate_checker_data.csv",
        "data/infer_data.csv",
        "data/openjml_data.csv"
    ]

    for file in warning_data_files:
        print(get_num_warnings(file))

    # TODO: Add a set of data where all tools are combined?