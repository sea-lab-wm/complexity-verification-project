import pandas as pd

# Retrieves the snippet name and warning message for each warning output by the Checker Framework
def parseCheckerFramework(data):
    lines = []
    with open('output.txt') as f:
        lines = f.readlines()

    # Delimeters with which to parse the warnings
    startSnippet = "\\fMRI_Study_Classes\\"
    endSnippet = ".java"
    startError = "] "

    for line in lines:
        if startSnippet in line:
            data["Snippet"].append((line.split(startSnippet))[1].split(endSnippet)[0])
            data["Error Type"].append(line.split(startError)[1].strip())

    return data

def setupExcelSheet():
    # Data output structure as a dictionary
    data = {
        "Snippet": [],
        "Error Type": []
    }

    # Creates a table from the dictionary of data
    df = pd.DataFrame(parseCheckerFramework(data))

    df["Error Type Copy"] = df["Error Type"]    # Must copy values to column with different name to use same column for "values" and "columns" when making a pivot table
    df = df.pivot_table(values="Error Type", index="Snippet", columns="Error Type Copy", aggfunc="count")

    df.to_excel('demo.xlsx', engine='xlsxwriter')

if __name__ == '__main__':
    setupExcelSheet()