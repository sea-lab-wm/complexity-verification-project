import pandas as pd
import numpy as np

def get_num_warnings(file_path: str, tool: str) -> pd.DataFrame:
    """Returns a dataframe where each entry is a dataset, snippet, and number of warnings from a single tool."""

    dict_df = {
        "dataset_id":[],
        "snippet_id":[],
        f"warnings_{tool}":[]
    }

    data = pd.DataFrame(dict_df)

    df = pd.read_csv(file_path)

    data[f"warnings_{tool}"] = df.sum(axis=1, numeric_only=True).tolist()

    new = df["Snippet"].str.split("--", expand=True, n=1)

    data["dataset_id"] = new[0]
    data["snippet_id"] = new[1]

    data.dataset_id = [d.strip() for d in data.dataset_id]
    data.snippet_id = [s.strip() for s in data.snippet_id]

    return data

def read_dataset_1_metrics() -> pd.DataFrame:
    """Reads the results of the first pilot study for COG dataset 1. It contains 41 people who looked at 23 snippets.
    Metrics include time to solve (in sec.), correctness where 0 = completely wrong, 1 = in part correct, 2 = completely correct, and
    Subjective rating is on a scale of 0 through 4 where 0 = very difficult, 1 = difficult, 2 = medium, 3 = easy, 4 = very easy.
    """

    dict_df = {
        "dataset_id": [],
        "snippet_id": [],
        "person_id": []
    }
    data = pd.DataFrame(dict_df)

    cols = []

    df = pd.read_excel("data/cog_dataset_1.xlsx")

    cols.extend([df.columns.get_loc(f"{str(snippetNum)}::time") for snippetNum in range(1, 24)])
    cols.extend([df.columns.get_loc(f"{str(snippetNum)}::Correct") for snippetNum in range(1, 24)])
    cols.extend([df.columns.get_loc(f"{str(snippetNum)}::Difficulty") for snippetNum in range(1, 24)])
    df_cols = df.iloc[:41, cols]

    for row_index, row in df_cols.iterrows(): #iterate over rows
        for column_index, value in row.items():
            metric_type = column_index.split("::")[1].lower()
            metric_id = None

            if "time" in metric_type:
                metric_id = "time_to_give_output"
            elif "correct" in metric_type:
                metric_id = "correct_output_rating"
            elif "difficulty" in metric_type:
                metric_id = "output_difficulty"
            else:
                raise Exception("read_dataset_1_metrics: could not determine metric_id")

            record = {
                    "dataset_id": ["1"],
                    "snippet_id": [column_index.split("::")[0].strip()],
                    "person_id": [row_index + 1],
                    metric_id: [value]
                }
            df_record = pd.DataFrame(record)
            data = pd.concat([data, df_record], ignore_index=True, axis=0)

    return data  

def read_dataset_2_metrics() -> pd.DataFrame:
    """Reads the results of the study for COG dataset 2. It contains 16 people who looked at 12 snippets.
    Metrics include time to solve, as well as 3 physiological metrics: BA32, BA31post, and BA31ant.

    **Note: Dataset 2 is a subset of dataset 1. All snippets in dataset 2 are also in dataset 1.
    **Note: For the physiological metrics, only the averages across all participants are recorded and thus the average is copied for each individual participant.
    """

    dict_df = {
        "dataset_id": [],
        "snippet_id": [],
        "person_id": []
    }
    data = pd.DataFrame(dict_df)

    df_physiological = pd.read_csv("data/cog_dataset_2_physiological.csv", header=0)
    df_response_times = pd.read_csv("data/cog_dataset_2_response_times.csv", header=0)

    df_cols = df_physiological.iloc[:, [0, 5, 6, 7]]
    
    for _, row in df_cols.iterrows():
        for p in range(1, 17):
            record = {
                    "dataset_id": ["2"],
                    "snippet_id": [row[0].split("_")[0]],
                    "person_id": [p],
                    "brain_deact_31ant": [row[3]],
                    "brain_deact_31post": [row[2]],
                    "brain_deact_32": [row[1]]
                }
            df_record = pd.DataFrame(record)
            data = pd.concat([data, df_record], ignore_index=True, axis=0)

    df_cols = df_response_times.iloc[1:, 1:-1:2]

    for row_index, row in df_cols.iterrows():
        for column_index, value in row.items():
            record = {
                    "dataset_id": ["2"],
                    "snippet_id": [column_index.split("_")[0]],
                    "person_id": [row_index],
                    "time_to_understand": [value]
                }
            df_record = pd.DataFrame(record)
            data = pd.concat([data, df_record], ignore_index=True, axis=0)

    return data

def read_dataset_3_metrics() -> pd.DataFrame:
    """Reads the results of the cog data set 3 study. It contains 121 people who rated 100 snippets on a scale of 1-5.
    1 being less readable and 5 being more readable.
    """

    dict_df = {
        "dataset_id": [],
        "snippet_id": [],
        "person_id": []
    }
    data = pd.DataFrame(dict_df)

    df = pd.read_csv("data/cog_dataset_3.csv", header=None)

    # Simplifies down to only the necessary data
    df_cols = df.iloc[:, 2:102]

    for row_index, row in df_cols.iterrows():
        for column_index, value in row.items():
            record = {
                    "dataset_id": ["3"],
                    "snippet_id": [str(column_index - 1)],
                    "person_id": [row_index + 1],
                    "readability_level": [value]
                }
            df_record = pd.DataFrame(record)
            data = pd.concat([data, df_record], ignore_index=True, axis=0)

    return data

def read_dataset_6_metrics() -> pd.DataFrame:
    """Reads the results of the cog data set 6 study. It contains 63 people who looked at 50 snippets with metrics based on time, correctness, and rating.
    IMPORTANT: Each participant was assigned randomly assigned about 8 snippets out of the 50. So not every person looked at every snippet.
    The metrics were Time Needed for Perceived Understandability (TNPU), Actual Understanding (AU), and Perceived Binary Understandability (PBU).
    """
    
    dict_df = {
        "dataset_id": [],
        "snippet_id": [],
        "person_id": []
    }
    data = pd.DataFrame(dict_df)

    df = pd.read_csv("data/metric_sheets_with_snippet_names/cog_dataset_6_with_snippet_names.csv")

    # Simplifies down to only the necessary data
    df_cols = df.iloc[:, [0, 2, 124, 125, 126]]
    print(df_cols)

    tmpcount = 0
    # loops through each row of data, creating a new record for each metric in that row separatly
    for _, row in df_cols.iterrows():
        tmpcount += 1
        record = {
                "dataset_id": ["6"],
                "snippet_id": [row[1]],
                "person_id": [row[0]],
                "binary_understandability": [row[2]],
                "time_to_understand": [row[3]],
                "correct_verif_questions": [row[4]]
            }
        df_record = pd.DataFrame(record)
        data = pd.concat([data, df_record], ignore_index=True, axis=0)

    print(tmpcount)
    return data

def read_dataset_9_metrics() -> pd.DataFrame:
    """Reads the results of the cog data set 9 study. It contains 104 participants and 30 unique snippets (5 snippets each with varying quality of comments).
    Correlation data is split into 3 categories of 10 snippets each: Good comments, bad comments, and no comments. Then further split into the metrics:
    Time, correctness, and rating.
    """
    
    dict_df = {
        "dataset_id": [],
        "snippet_id": [],
        "person_id": []
    }
    data = pd.DataFrame(dict_df)

    df = pd.read_excel("data/cog_dataset_9.xlsx")

    # Simplifies down to only the necessary data
    df_cols = df.iloc[:520, [0, 18, 24, 61, 81, 82, 85]]

    j = 0
    prev = None
    # loops through each row of data, creating a new record for each metric in that row separatly
    for _, row in df_cols.iterrows():
        if prev != row[1].split(":")[0].split("_")[1]:
            j += 1
            prev = row[1].split(":")[0].split("_")[1]

        # determines if the current row represents a snippet with good comments, bad comments, or no comments.
        dataset_id = None
        if "1" in row[1].split(":")[1]:
            dataset_id = "9_gc"
        elif "2" in row[1].split(":")[1]:
            dataset_id = "9_bc"
        elif "3" in row[1].split(":")[1]:
            dataset_id = "9_nc"
        else:
            raise Exception("read_dataset_9_metrics: could not determine dataset_id")

        record = {
                "dataset_id": [dataset_id],
                "snippet_id": [str(j)],
                "person_id": [row[0]],
                "gap_accuracy": [row[6]],
                "readability_level_ba": [row[2] + row[3]],
                "readability_level_before": [row[2]],
                "time_to_read_complete": [row[4] + row[5]]
            }
        df_record = pd.DataFrame(record)
        data = pd.concat([data, df_record], ignore_index=True, axis=0)
    
    return data

def read_dataset_f_metrics() -> pd.DataFrame:
    """Reads the results of the fmri study. It contains 19 people who looked at 16 snippets.
    Correctness (in %), time to solve (in sec.), and a subjective rating were all measured.
    Subjective rating of low, medium, or high.
    """

    dict_df = {
        "dataset_id": [],
        "snippet_id": [],
        "person_id": []
    }
    data = pd.DataFrame(dict_df)

    df_physiological = pd.read_csv("data/fmri_dataset_physiological.csv")
    df_physiological = df_physiological.set_index("condition", drop=False)

    df_behavioral = pd.read_csv("data/fmri_dataset_behavioral.csv")
    df_subjective = pd.read_csv("data/fmri_dataset_subjective.csv")
    combined = pd.merge(df_behavioral, df_subjective, on=["Participant", "Snippet"], how="left")

    def convert_phsyiological(row: pd.Series, mode: str ):  # mode = "BA31" OR "BA32"
        """Gets the physiological metric that corresponds to the given snippet name in 'row'.
        Important Note: The physiological metrics are already averaged for all participants and thus the average is being used for each participants individual result.
        """

        row_given_snippet = df_physiological.loc[row[3], :]

        if "BA31" in mode:
            return row_given_snippet.loc["BA31"]
        elif "BA32" in mode:
            return row_given_snippet.loc["BA32"]
        else:
            raise Exception("convert_phsyiological: invalid mode")

    # Create columns for the physiological data applying each metric for a given snippet to every instance of that snippet.
    # i.e. same result for every participant for the given snippet.
    combined["BA31"] = combined.apply(lambda row: convert_phsyiological(row, "BA31"), axis=1)
    combined["BA32"] = combined.apply(lambda row: convert_phsyiological(row, "BA32"), axis=1)

    # Simplifies down to only the necessary data
    df_cols = combined.iloc[:, [0, 3, 7, 12, 14, 15, 16]]

    j = 0
    prev = None
    # loops through each row of data, creating a new record for each metric in that row separatly
    for _, row in df_cols.iterrows():
        if prev != row[1]:
            j += 1
            prev = row[1]

        record = {
                "dataset_id": ["f"],
                "snippet_id": [f"{j} -- {row[1]}"],
                "person_id": [row[0]],
                "perc_correct_output": [row[2]],
                "time_to_understand": [float(row[3]) / 1000],    # Converting from milliseconds to seconds
                "complexity_level": [row[4]],
                "brain_deact_31": [row[5]],
                "brain_deact_32": [row[6]]
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

    # read and combine all metric data
    metric_data = []
    metric_data.append(read_dataset_1_metrics())
    metric_data.append(read_dataset_2_metrics())
    metric_data.append(read_dataset_3_metrics())
    metric_data.append(read_dataset_6_metrics())
    metric_data.append(read_dataset_9_metrics())
    metric_data.append(read_dataset_f_metrics())

    warning_df = pd.concat(warning_data, ignore_index=True)
    warning_df.to_csv("ml_model/test_warning_df.csv")
    metric_df = pd.concat(metric_data, ignore_index=True)
    metric_df.to_csv("ml_model/test_metric_df.csv")

    # Combine rows with duplicate keys
    metric_df = metric_df.replace('', np.nan).groupby(["dataset_id", "snippet_id", "person_id"]).first().reset_index()
    warning_df = warning_df.replace('', np.nan).groupby(["dataset_id", "snippet_id"]).first().reset_index()

    # Merge the two dataframes
    all_df = pd.merge(metric_df, warning_df, on=["dataset_id", "snippet_id"], how="left")

    # Fill empty warning cells with zeros
    all_df.update(all_df[["warnings_checker_framework", "warnings_typestate_checker", "warnings_infer", "warnings_openjml"]].fillna(0))

    # Create column for the sum of warnings across all tools
    all_df["warning_sum"] = all_df["warnings_checker_framework"] + all_df["warnings_typestate_checker"] + all_df["warnings_infer"] + all_df["warnings_openjml"]

    print(all_df)

    # Number of total metrics we should be getting per dataset
    # ds 1: 23 * 41 = 943
    # ds 2: 12 * 16 = 192
    # ds 3: 100 * 121 = 12,100
    # ds 6: 444 but there are some duplicates so it merges to 437
    # ds 9: 520
    # ds f: 19 * 16 = 304
    # total: 14,496

    if len(all_df.index) != 14496:
        raise Exception("create_ml_table: length mismatch.")

    all_df.to_csv("ml_model/metric_data.csv")