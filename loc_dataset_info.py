import pandas as pd

def readCLOCOutput(include_comments):
    datasets = {
        "dataset_1": [],
        "dataset_2": [],
        "dataset_3": [],
        "dataset_6": [],
        "dataset_9": [],
        "dataset_f": []
    }

    df = pd.read_csv("loc_per_snippet/output_without_javadoc.csv")

    if include_comments:
        for row in df.itertuples():
            for key, value in datasets.items():
                if isinstance(row[2], str) and key in row[2]:
                    value.append(row[5] + row[4])
    else:
        for row in df.itertuples():
            for key, value in datasets.items():
                if isinstance(row[2], str) and key in row[2]:
                    value.append(row[5])

    return datasets

def compute_average(results):
    for key, value in results.items():
        results[key] = sum(value) / len(value)

    return results

def compute_min_max(results):
    for key, value in results.items():
        results[key] = {
            "min": min(value),
            "max": max(value)
        }

    return results

if __name__ == "__main__":
    print("NCLOC Average: " + str(compute_average(readCLOCOutput(False))))
    print("NCLOC Min/Max: " + str(compute_min_max(readCLOCOutput(False))))

    print("LOC Average: " + str(compute_average(readCLOCOutput(True))))
    print("LOC Min/Max: " + str(compute_min_max(readCLOCOutput(True))))