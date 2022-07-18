import pandas as pd
import copy
import os

#########################
#   Get Snippet Names   #
#########################

# Parses a .java file containing numbered snippets to determine which snippet starts on which line.
def getSnippetNames(filePath, start, end):
    methodStartLines = []

    with open(filePath) as f:
        for lineNum, line in enumerate(f, start=1):
            if start in line:
                methodStartLines.append(lineNum + 1)
            elif end in line:
                methodStartLines.append(lineNum)

    return methodStartLines

##################################
#   Parse Analysis Tool Output   #
##################################

# Retrieves the snippet name and warning message for each warning output by the Checker Framework.
def parseCheckerFramework(data, fMRIDatasetSnippetNames, cogDataset1SnippetNums, cogDataset2SnippetNums, cogDataset3SnippetNums):
    lines = []
    with open("data/checker_framework_output.txt") as f:
        lines = f.readlines()

    data = parseAll(data, lines, fMRIDatasetSnippetNames, cogDataset1SnippetNums, cogDataset2SnippetNums, cogDataset3SnippetNums, ": warning:")
    
    return ("checker_framework_data", data)

# Retrieves the snippet name and warning message for each warning output by the Typestate Checker.
def parseTypestateChecker(data, fMRIDatasetSnippetNames, cogDataset1SnippetNums, cogDataset2SnippetNums, cogDataset3SnippetNums):
    lines = []
    files = os.listdir("data")

    filesToFind = "typestate_checker_output"    # File delimeter
    
    # The Typestate Checker runs on specific directories instead of the entire project so it produces multiple output files.
    # This finds all the relevent output files.
    for fName in files:
        if filesToFind in fName:
            with open(os.path.join("data", fName)) as f:
                lines.append(f.readlines())

    for dataset in lines:
        data = parseAll(data, dataset, fMRIDatasetSnippetNames, cogDataset1SnippetNums, cogDataset2SnippetNums, cogDataset3SnippetNums, ": warning:")

    return ("typestate_checker_data", data)

def parseInfer(data, fMRIDatasetSnippetNames, cogDataset1SnippetNums, cogDataset2SnippetNums, cogDataset3SnippetNums):
    lines = []
    with open("data/infer_output.txt") as f:
        lines = f.readlines()

    data = parseAll(data, lines, fMRIDatasetSnippetNames, cogDataset1SnippetNums, cogDataset2SnippetNums, cogDataset3SnippetNums, ": error:")
    
    return ("infer_data", data)

# Parses the analysis tool output of the Checker Framework, Typestate Checker, and Infer.
def parseAll(data, lines, fMRIDatasetSnippetNames, cogDataset1SnippetNums, cogDataset2SnippetNums, cogDataset3SnippetNums, endSnippet):
    # Delimeters with which to parse the warnings
    startSnippetfMRI = os.path.join(" ", "fMRI_Study_Classes", " ").strip()
    startSnippetCOG1 = os.path.join(" ", "cog_complexity_validation_datasets", "One", " ").strip()
    startSnippetCOG3 = os.path.join(" ", "cog_complexity_validation_datasets", "Three", " ").strip()

    for line in lines:
        if startSnippetfMRI in line and endSnippet in line:
            data["Snippet"].append(f"fMRI Dataset - {str(fMRIDatasetSnippetNames.index((line.split(startSnippetfMRI))[1].split('.java')[0]) + 1)} - {(line.split(startSnippetfMRI))[1].split('.java')[0]}")
            data["Warning Type"].append(line.split(endSnippet)[1].strip())
        elif startSnippetCOG1 in line and endSnippet in line:
            lineNum = int(line.split(".java:")[1].split(":")[0])

            for i in range(len(cogDataset1SnippetNums) - 1):
                if cogDataset1SnippetNums[i] <= lineNum and cogDataset1SnippetNums[i + 1] > lineNum:
                    data["Snippet"].append(f"COG Dataset 1 - {str(i + 1)}")
                    data["Warning Type"].append(line.split(endSnippet)[1].strip())

                    # Additional check to see if the snippet is from dataset 2 as well (which is a subset of 1)
                    for j in range(0, len(cogDataset2SnippetNums) - 1, 2):
                        if cogDataset2SnippetNums[j] <= lineNum and cogDataset2SnippetNums[j + 1] > lineNum:
                            data["Snippet"].append(f"COG Dataset 2 - {str((j + 2) // 2)}")
                            data["Warning Type"].append(line.split(endSnippet)[1].strip())
                            break
                    break
        elif startSnippetCOG3 in line and endSnippet in line:
            lineNum = int(line.split(".java:")[1].split(":")[0])

            for i in range(len(cogDataset3SnippetNums) - 1):
                if cogDataset3SnippetNums[i] <= lineNum and cogDataset3SnippetNums[i + 1] > lineNum:
                    data["Snippet"].append(f"COG Dataset 3 - {str(i + 1)}")
                    data["Warning Type"].append(line.split(endSnippet)[1].strip())
                    break

    return data

########################
#   Setup CSV Sheets   #
########################

# Creates csv files for each analysis tool"s warning output.
def setupCSVSheets(allAnalysisToolData):
    for data in allAnalysisToolData:
        # Creates a table from the dictionary of data
        df = pd.DataFrame(data[1])

        # Convert data using pivot table
        df["Warning Type Copy"] = df["Warning Type"]    # Must copy values to column with different name to use same column for "values" and "columns" when making a pivot table
        df = df.pivot_table(values="Warning Type", index="Snippet", columns="Warning Type Copy", aggfunc="count")

        df.to_csv(f"data/{data[0]}.csv")

if __name__ == "__main__":
    fMRIDatasetSnippetNames = [name.split(".")[0] for name in os.listdir("simple-datasets/src/main/java/fMRI_Study_Classes") if ".java" in name]
    fMRIDatasetSnippetNames = sorted(fMRIDatasetSnippetNames, key=str.lower)    # Keeps the order of these snippets consistent across operating systems
    cogDataset1SnippetNums = getSnippetNames("simple-datasets/src/main/java/cog_complexity_validation_datasets/One/Tasks.java", "SNIPPET_STARTS", "SNIPPETS_END")
    cogDataset2SnippetNums =  getSnippetNames("simple-datasets/src/main/java/cog_complexity_validation_datasets/One/Tasks.java", "DATASET2START", "DATASET2END")
    cogDataset3SnippetNums = getSnippetNames("simple-datasets/src/main/java/cog_complexity_validation_datasets/Three/Tasks.java", "SNIPPET_STARTS", "SNIPPETS_END")

    # Data output structure as a dictionary
    data = {
        "Snippet": [],
        "Warning Type": []
    }
    allAnalysisToolData = []

    #TODO: All analysis tools go here
    allAnalysisToolData.append(parseCheckerFramework(copy.deepcopy(data), fMRIDatasetSnippetNames, cogDataset1SnippetNums, cogDataset2SnippetNums, cogDataset3SnippetNums))
    allAnalysisToolData.append(parseTypestateChecker(copy.deepcopy(data), fMRIDatasetSnippetNames, cogDataset1SnippetNums, cogDataset2SnippetNums, cogDataset3SnippetNums))
    allAnalysisToolData.append(parseInfer(copy.deepcopy(data), fMRIDatasetSnippetNames, cogDataset1SnippetNums, cogDataset2SnippetNums, cogDataset3SnippetNums))
    setupCSVSheets(allAnalysisToolData)