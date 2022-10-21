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

def get_metrics():
    """Returns a dataframe containing metric data for each dataset."""

    dict_df = {
        "dataset_id": [],
        "snippet_id": [],
        "person_id": [],
        "metric": [],
        "metric_type": []
    }

    data = pd.DataFrame(dict_df)

    for record in read_cog_dataset_1_metrics():
        df_record = pd.DataFrame(record)
        data = pd.concat([data, df_record], ignore_index=True, axis=0)  

    return data

def read_cog_dataset_1_metrics():
    """Reads the results of the first pilot study for COG dataset 1. It contains 41 people who looked at 23 snippets.
    Metrics include time to solve (in sec.), correctness where 0 = completely wrong, 1 = in part correct, 2 = completely correct, and
    Subjective rating is on a scale of 0 through 4 where 0 = very difficult, 1 = difficult, 2 = medium, 3 = easy, 4 = very easy.
    """

    records = []
    cols = []

    df = pd.read_excel("data/cog_dataset_1.xlsx")

    cols.extend([df.columns.get_loc(f"{str(snippetNum)}::time") for snippetNum in range(1, 24)])
    cols.extend([df.columns.get_loc(f"{str(snippetNum)}::Correct") for snippetNum in range(1, 24)])
    cols.extend([df.columns.get_loc(f"{str(snippetNum)}::Difficulty") for snippetNum in range(1, 24)])
    df_cols = df.iloc[:41, cols]

    for rowIndex, row in df_cols.iterrows(): #iterate over rows
        for columnIndex, value in row.items():
            records.append(
                {
                    "dataset_id": ["1"],
                    "snippet_id": [columnIndex.split("::")[0]],
                    "person_id": [rowIndex],
                    "metric": [value],
                    "metric_type": [columnIndex.split("::")[0].lower()]
                }
            )

    print(len(records))
    return records

if __name__ == "__main__":
    warning_data_files = [
        "data/checker_framework_data.csv",
        "data/typestate_checker_data.csv",
        "data/infer_data.csv",
        "data/openjml_data.csv"
    ]

    #for file in warning_data_files:
    #    print(get_num_warnings(file))

    # TODO: Add a set of data where all tools are combined?

    data = get_metrics()
    print(data)