import pandas as pd

# Retrieves the snippet name and warning message for each warning output by the Checker Framework
def parseCheckerFramework(data):
    lines = []
    with open('checker_framework_output.txt') as f:
        lines = f.readlines()

    # Delimeters with which to parse the warnings
    startSnippetfMRI = "\\fMRI_Study_Classes\\"
    startSnippetCOG = "\\cog_complexity_validation_datasets\\"
    endSnippet = ": warning:"
    #startError = " ["

    for line in lines:
        if startSnippetfMRI in line and endSnippet in line:
            data["Snippet"].append(startSnippetfMRI + ": " + (line.split(startSnippetfMRI))[1].split(endSnippet)[0])
            data["Error Type"].append(line.split(endSnippet)[1].strip())
        elif startSnippetCOG in line and endSnippet in line:
            data["Snippet"].append(startSnippetCOG + ": " + (line.split(startSnippetCOG))[1].split(endSnippet)[0])
            data["Error Type"].append(line.split(endSnippet)[1].strip())

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

    df.to_excel('checker_framework_data.xlsx', engine='xlsxwriter')

if __name__ == '__main__':
    setupExcelSheet()