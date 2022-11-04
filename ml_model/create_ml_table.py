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

    data = get_metrics()
    print(data)