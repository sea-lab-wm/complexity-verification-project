import re
import pandas as pd

def readCLOCOutput():
    datasets = {
        "dataset_1": [],
        "dataset_2": [],
        "dataset_3": [],
        "dataset_6": [],
        "dataset_9": [],
        "dataset_f": []
    }

    df = pd.read_csv("loc_per_snippet/output.csv")

    for row in df.itertuples():
        for key, value in datasets.items():
            if isinstance(row[2], str) and key in row[2]:
                value.append(row[5])

    return datasets

def computeAverage(results):
    for key, value in results.items():
        results[key] = sum(value) / len(value)

    return results

if __name__ == "__main__":
    results = readCLOCOutput()

    print(computeAverage(results))