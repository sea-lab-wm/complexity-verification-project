import pandas as pd

def get_num_warnings(file, tool):
    """Returns a dataframe where each entry is a dataset, snippet, and number of warnings from a single tool."""

    dict_df = {
        "dataset_id":[],
        "snippet_id":[],
        "tool": [],
        "num_warnings":[]
    }

    data = pd.DataFrame(dict_df)

    df = pd.read_csv(file)

    data["num_warnings"] = df.sum(axis=1, numeric_only=True).tolist()

    new = df["Snippet"].str.split("--", expand = True)
    data["dataset_id"] = new[0]
    data["snippet_id"] = new[1]

    data = data.assign(tool=tool)

    return data

def get_metrics():
    """Returns a dataframe containing metric data for each dataset."""

    metric_data = []

    metric_data.append(read_dataset_1_metrics())
    metric_data.append(read_dataset_3_metrics())

    return pd.concat(metric_data)

def read_dataset_1_metrics():
    """Reads the results of the first pilot study for COG dataset 1. It contains 41 people who looked at 23 snippets.
    Metrics include time to solve (in sec.), correctness where 0 = completely wrong, 1 = in part correct, 2 = completely correct, and
    Subjective rating is on a scale of 0 through 4 where 0 = very difficult, 1 = difficult, 2 = medium, 3 = easy, 4 = very easy.
    """

    dict_df = {
        "dataset_id": [],
        "snippet_id": [],
        "person_id": [],
        "metric": [],
        "metric_type": []
    }
    data = pd.DataFrame(dict_df)

    cols = []

    df = pd.read_excel("data/cog_dataset_1.xlsx")

    cols.extend([df.columns.get_loc(f"{str(snippetNum)}::time") for snippetNum in range(1, 24)])
    cols.extend([df.columns.get_loc(f"{str(snippetNum)}::Correct") for snippetNum in range(1, 24)])
    cols.extend([df.columns.get_loc(f"{str(snippetNum)}::Difficulty") for snippetNum in range(1, 24)])
    df_cols = df.iloc[:41, cols]

    for rowIndex, row in df_cols.iterrows(): #iterate over rows
        for columnIndex, value in row.items():
            record = {
                    "dataset_id": ["1"],
                    "snippet_id": [columnIndex.split("::")[0]],
                    "person_id": [rowIndex],
                    "metric": [value],
                    "metric_type": [columnIndex.split("::")[1].lower()]
                }
            df_record = pd.DataFrame(record)
            data = pd.concat([data, df_record], ignore_index=True, axis=0)

    return data  

def read_dataset_3_metrics():
    """Reads the results of the cog data set 3 study. It contains 121 people who rated 100 snippets on a scale of 1-5.
    1 being less readable and 5 being more readable.
    """

    dict_df = {
        "dataset_id": [],
        "snippet_id": [],
        "person_id": [],
        "metric": [],
        "metric_type": []
    }
    data = pd.DataFrame(dict_df)

    df = pd.read_csv("data/cog_dataset_3.csv", header=None)

    df_cols = df.iloc[:, 2:102]

    for rowIndex, row in df_cols.iterrows():
        for columnIndex, value in row.items():
            record = {
                    "dataset_id": ["3"],
                    "snippet_id": [columnIndex],
                    "person_id": [rowIndex],
                    "metric": [value],
                    "metric_type": ["rating"]
                }
            df_record = pd.DataFrame(record)
            data = pd.concat([data, df_record], ignore_index=True, axis=0)

    return data

if __name__ == "__main__":
    warning_data_files = {
        "checker_framework": "data/checker_framework_data.csv",
        "typestate_checker": "data/typestate_checker_data.csv",
        "infer": "data/infer_data.csv",
        "openjml": "data/openjml_data.csv"
    }
    warning_data = []

    for name, file in warning_data_files.items():
        warning_data.append(get_num_warnings(file, name))

    # Add a set of data where all tools are combined
    warning_data.append(pd.concat(warning_data))

    metric_data = get_metrics()
    print(metric_data)