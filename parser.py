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

# Get the line numbers for all snippets
def getAllSnippets():
    allSnippetNums = []

    # fMRI Dataset
    fMRIDatasetSnippetNums = []
    inOrder = [file.split(".")[0] for file in os.listdir("simple-datasets/src/main/java/fMRI_Study_Classes") if ".java" in file]
    inOrder = sorted(inOrder, key=str.lower) # Keeps the order of these snippets consistent across operating systems
    for file in inOrder:
        fMRIDatasetSnippetNums.append(getSnippetNames(f"simple-datasets/src/main/java/fMRI_Study_Classes/{file}.java", "SNIPPET_STARTS", "**NO_END**")[0])
    fMRIDatasetSnippetNames = {file:fMRIDatasetSnippetNums[i] for i, file in enumerate(inOrder)}
    allSnippetNums.append(fMRIDatasetSnippetNames)

    # COG Dataset 1
    allSnippetNums.append(getSnippetNames("simple-datasets/src/main/java/cog_complexity_validation_datasets/One/Tasks.java", "SNIPPET_STARTS", "SNIPPETS_END"))
    #COG Dataset 2
    allSnippetNums.append(getSnippetNames("simple-datasets/src/main/java/cog_complexity_validation_datasets/One/Tasks.java", "DATASET2START", "DATASET2END"))

    # COG Dataset 3
    cogDataset3SnippetNums = {}
    cogDataset3SnippetNums["Tasks_1"] = getSnippetNames("simple-datasets/src/main/java/cog_complexity_validation_datasets/Three/Tasks_1.java", "SNIPPET_STARTS", "SNIPPETS_END")
    cogDataset3SnippetNums["Tasks_2"] = getSnippetNames("simple-datasets/src/main/java/cog_complexity_validation_datasets/Three/Tasks_2.java", "SNIPPET_STARTS", "SNIPPETS_END")
    cogDataset3SnippetNums["Tasks_3"] = getSnippetNames("simple-datasets/src/main/java/cog_complexity_validation_datasets/Three/Tasks_3.java", "SNIPPET_STARTS", "SNIPPETS_END")
    allSnippetNums.append(cogDataset3SnippetNums)

    # COG Dataset 6
    # A dictionary of lists. Each inner list contains the line numbers for the snippets in a single .java file. This dataset has snippets split across several files.
    # They are in the order of how they appear in "cog_dataset_6.csv", the file containing the metric data from its prior study.
    cogDataset6SnippetNums = {}
    cogDataset6SnippetNums["K9"] = getSnippetNames("dataset6/src/main/java/K9.java", "SNIPPET_STARTS", "SNIPPETS_END")
    cogDataset6SnippetNums["Pom"] = getSnippetNames("dataset6/src/main/java/Pom.java", "SNIPPET_STARTS", "SNIPPETS_END")
    cogDataset6SnippetNums["CarReport"] = getSnippetNames("dataset6/src/main/java/CarReport.java", "SNIPPET_STARTS", "SNIPPETS_END")
    cogDataset6SnippetNums["Antlr4Master"] = getSnippetNames("dataset6/src/main/java/Antlr4Master.java", "SNIPPET_STARTS", "SNIPPETS_END")
    cogDataset6SnippetNums["Phoenix"] = getSnippetNames("dataset6/src/main/java/Phoenix.java", "SNIPPET_STARTS", "SNIPPETS_END")
    cogDataset6SnippetNums["HibernateORM"] = getSnippetNames("dataset6/src/main/java/HibernateORM.java", "SNIPPET_STARTS", "SNIPPETS_END")
    cogDataset6SnippetNums["OpenCMSCore"] = getSnippetNames("dataset6/src/main/java/OpenCMSCore.java", "SNIPPET_STARTS", "SNIPPETS_END")
    cogDataset6SnippetNums["SpringBatch"] = getSnippetNames("dataset6/src/main/java/SpringBatch.java", "SNIPPET_STARTS", "SNIPPETS_END")
    cogDataset6SnippetNums["MyExpenses"] = getSnippetNames("dataset6/src/main/java/MyExpenses.java", "SNIPPET_STARTS", "SNIPPETS_END")
    cogDataset6SnippetNums["CheckEstimator"] = getSnippetNames("dataset6/src/main/java/weka/estimators/CheckEstimator.java", "SNIPPET_STARTS", "SNIPPETS_END")
    cogDataset6SnippetNums["EstimatorUtils"] = getSnippetNames("dataset6/src/main/java/weka/estimators/EstimatorUtils.java", "SNIPPET_STARTS", "SNIPPETS_END")
    cogDataset6SnippetNums["ClassifierPerformanceEvaluatorCustomizer"] = getSnippetNames("dataset6/src/main/java/weka/gui/beans/ClassifierPerformanceEvaluatorCustomizer.java", "SNIPPET_STARTS", "SNIPPETS_END")
    cogDataset6SnippetNums["ModelPerformanceChart"] = getSnippetNames("dataset6/src/main/java/weka/gui/beans/ModelPerformanceChart.java", "SNIPPET_STARTS", "SNIPPETS_END")
    cogDataset6SnippetNums["GeneratorPropertyIteratorPanel"] = getSnippetNames("dataset6/src/main/java/weka/gui/experiment/GeneratorPropertyIteratorPanel.java", "SNIPPET_STARTS", "SNIPPETS_END")
    allSnippetNums.append(cogDataset6SnippetNums)

    # COG Dataset 9
    allSnippetNums.append(getSnippetNames("dataset9/src/main/java/CodeSnippets.java", "SNIPPET_STARTS_1", "SNIPPET_END_1"))
    allSnippetNums.append(getSnippetNames("dataset9/src/main/java/CodeSnippets.java", "SNIPPET_STARTS_2", "SNIPPET_END_2"))
    allSnippetNums.append(getSnippetNames("dataset9/src/main/java/CodeSnippets.java", "SNIPPET_STARTS_3", "SNIPPET_END_3"))

    return allSnippetNums

##################################
#   Parse Analysis Tool Output   #
##################################

# Retrieves the snippet name and warning message for each warning output by the Checker Framework.
def parseCheckerFramework(data, allSnippetNums):
    lines = []
    with open("data/checker_framework_output.txt") as f:
        lines = f.readlines()

    data = parseAll(data, lines, allSnippetNums, ": warning:")
    
    return ("checker_framework_data", data)

# Retrieves the snippet name and warning message for each warning output by the Typestate Checker.
def parseTypestateChecker(data, allSnippetNums):
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
        data = parseAll(data, dataset, allSnippetNums, ": warning:")

    return ("typestate_checker_data", data)

def parseInfer(data, allSnippetNums):
    lines = []
    with open("data/infer_output.txt") as f:
        lines = f.readlines()

    data = parseAll(data, lines, allSnippetNums, ": error:")
    
    return ("infer_data", data)

def parseOpenJML(data, allSnippetNums, timeouts):
    lines = []
    files = os.listdir("data")

    filesToFind = "openjml_output"    # File delimeter
    
    # OpenJML runs on specific directories instead of the entire project so it produces multiple output files.
    # This finds all the relevent output files.
    for fName in files:
        if filesToFind in fName:
            with open(os.path.join("data", fName)) as f:
                lines.append(f.readlines())

    # Delimeters with which to parse the warnings
    startSnippetfMRI = os.path.join("fMRI_Study_Classes", " ").strip()
    startSnippetCOG1 = os.path.join("cog_complexity_validation_datasets", "One", " ").strip()
    startSnippetCOG3 = os.path.join("cog_complexity_validation_datasets", "Three").strip()
    startSnippetCOG9 = os.path.join(".", "CodeSnippets").strip()
    endSnippet = "verify:"
    startWarning = "assertion"
    endWarning = "in method"

    for dataset in lines:
        for line in dataset:
            if "Associated declaration" in line or "Associated method exit" in line:
                continue

            if startSnippetfMRI in line.split(".java:")[0] and endSnippet in line:
                lineNum = int(line.split(".java:")[1].split(":")[0])
                if allSnippetNums[0][(line.split(startSnippetfMRI))[1].split('.java')[0]] <= lineNum:
                    openJMLWriteData(data, line.split(endSnippet)[1], f"f -- {str(list(allSnippetNums[0].keys()).index((line.split(startSnippetfMRI))[1].split('.java')[0]) + 1)} -- {(line.split(startSnippetfMRI))[1].split('.java')[0]}", line, timeouts)
            elif startSnippetCOG1 in line.split(".java:")[0] and endSnippet in line:
                lineNum = int(line.split(".java:")[1].split(":")[0])

                for i in range(len(allSnippetNums[1]) - 1):
                    if allSnippetNums[1][i] <= lineNum and allSnippetNums[1][i + 1] > lineNum:
                        openJMLWriteData(data, line.split(endSnippet)[1], f"1 -- {str(i + 1)}", line, timeouts)

                        # Additional check to see if the snippet is from dataset 2 as well (which is a subset of 1)
                        for j in range(0, len(allSnippetNums[2]) - 1, 2):
                            if allSnippetNums[2][j] <= lineNum and allSnippetNums[2][j + 1] > lineNum:
                                openJMLWriteData(data, line.split(endSnippet)[1], f"2 -- {str((j + 2) // 2)}", line, timeouts)
                                break
                        break
            elif startSnippetCOG3 in line.split(".java:")[0] and endSnippet in line:
                lineNum = int(line.split(".java:")[1].split(":")[0])
                fileName = line.split(".java:")[0].rsplit("/", 1)[1]

                if fileName not in allSnippetNums[3]:
                    continue

                snippetNums = allSnippetNums[3][fileName]

                addToI = 0
                if fileName == "Tasks_2":
                    addToI = len(allSnippetNums[3]["Tasks_1"]) - 1
                elif fileName == "Tasks_3":
                    addToI = len(allSnippetNums[3]["Tasks_1"]) - 1
                    addToI += len(allSnippetNums[3]["Tasks_2"]) - 1

                for i in range(len(snippetNums) - 1):
                    if snippetNums[i] <= lineNum and snippetNums[i + 1] > lineNum:
                        openJMLWriteData(data, line.split(endSnippet)[1], f"3 -- {str(i + 1 + addToI)}", line, timeouts)

                        break
            elif "./" in line[:2] and line.split(".java:")[0].rsplit("/", 1)[1] in allSnippetNums[4]:
                lineNum = int(line.split(".java:")[1].split(":")[0])
                fileName = line.split(".java:")[0].rsplit("/", 1)[1]

                snippetNums = allSnippetNums[4][fileName]
    
                for i in range(len(snippetNums) - 1):
                    if snippetNums[i] <= lineNum and snippetNums[i + 1] > lineNum: 
                        openJMLWriteData(data, line.split(endSnippet)[1], f"6 -- {str(i + 1)} -- {fileName}", line, timeouts)

                        break
            elif startSnippetCOG9 in line.split(".java:")[0] and endSnippet in line:
                lineNum = int(line.split(".java:")[1].split(":")[0])

                for i in range(0, len(allSnippetNums[5]) - 1, 2):
                    if allSnippetNums[5][i] <= lineNum and allSnippetNums[5][i + 1] > lineNum:
                        data["Snippet"].append(f"9_gc -- {str((i + 2) // 2)}")
                        warning = line.split(endSnippet)[1]
                    elif allSnippetNums[6][i] <= lineNum and allSnippetNums[6][i + 1] > lineNum:
                        data["Snippet"].append(f"9_bc -- {str((i + 2) // 2)}")
                        warning = line.split(endSnippet)[1]
                    elif allSnippetNums[7][i] <= lineNum and allSnippetNums[7][i + 1] > lineNum:
                        data["Snippet"].append(f"9_nc -- {str((i + 2) // 2)}")
                        warning = line.split(endSnippet)[1]
                    else:
                        continue

                    if startWarning in warning and endWarning in warning:
                        warning = warning.split(endWarning)[0].split(startWarning)[1].strip()

                    data["Warning Type"].append(warning.strip())
                    break

    return ("openjml_data", data)

def openJMLWriteData(data, warning, message, line, timeouts):
    startWarning = "assertion"
    endWarning = "in method"

    if "timeout" in line:
        timeouts.append(message)

    data["Snippet"].append(message)

    if startWarning in warning and endWarning in warning:
        warning = warning.split(endWarning)[0].split(startWarning)[1].strip()

    data["Warning Type"].append(warning.strip())

def openJMLHandleTimeouts(timeouts):
    ###SET THIS TO CHANGE HOW TIMEOUTS ARE HANDLED###
    #0 = max, 1 = zero, 2 = completely remove the snippet
    handleType = 2

    df = pd.read_csv("data/openjml_data.csv")
    df.set_index("Snippet")

    max3 = findMaxNumWarnings("3", df)
    max6 = findMaxNumWarnings("6", df)

    for col in df.columns:
        if "timeout" in col:
            df = df.rename(columns={col: "timeout"})

    if handleType == 0:
        for message in timeouts:
            df.loc[df["Snippet"] == message, "timeout"] = "MAX"

        for message in timeouts:
            if (df.loc[df["Snippet"] == message, "timeout"] == "MAX").any():
                if "3" in message.split("--")[0]:
                    df.loc[df["Snippet"] == message, "timeout"] = max3
                elif "6" in message.split("--")[0]:
                    df.loc[df["Snippet"] == message, "timeout"] = max6
                else:
                    raise Exception("Issue handling timeouts!")
    else:
        for message in timeouts:
            df.loc[df["Snippet"] == message, "timeout"] = 0

        df.reset_index()
        for message in timeouts:
            if (df.loc[df["Snippet"] == message, "timeout"] == 0).all():
                if handleType == 1:
                    if (df.loc[df["Snippet"] == message].sum(axis=1, numeric_only=True) == 0).all():
                        df = df.drop(df.loc[df["Snippet"] == message].index)
                else:
                    df = df.drop(df.loc[df["Snippet"] == message].index)

    df.to_csv("data/openjml_data.csv", index=False)

def findMaxNumWarnings(dataset, df):
    numWarnings = df.sum(axis=1, numeric_only=True).tolist()
    snippetNames = df["Snippet"].to_list()
    maxNumWarnings = 0

    for i, snippet in enumerate(snippetNames):
        if pd.isnull(snippet):
            continue

        if dataset in snippet.split("--")[0]:
            if numWarnings[i] > maxNumWarnings:
                maxNumWarnings = numWarnings[i]

    return maxNumWarnings

def getNumTimeoutsPerDataset(timeouts):
    counts = {
        "dataset_1": 0,
        "dataset_2": 0,
        "dataset_3": 0,
        "dataset_6": 0,
        "dataset_9": 0,
        "dataset_f": 0
    }

    timeouts = list(set(timeouts))

    for snippet in timeouts:
        if f"dataset_{snippet.split('--')[0].strip()}" in counts:
            counts[f"dataset_{snippet.split('--')[0].strip()}"] += 1

    print(counts)

# Parses the analysis tool output of the Checker Framework, Typestate Checker, and Infer.
def parseAll(data, lines, allSnippetNums, endSnippet):
    # Delimeters with which to parse the warnings
    startSnippetfMRI = os.path.join(" ", "fMRI_Study_Classes", " ").strip()
    startSnippetCOG1 = os.path.join(" ", "cog_complexity_validation_datasets", "One", " ").strip()
    startSnippetCOG3 = os.path.join(" ", "cog_complexity_validation_datasets", "Three", " ").strip()
    startSnippetCOG6 = "dataset6"
    startSnippetCOG9 = "dataset9"

    for line in lines:
        if startSnippetfMRI in line and endSnippet in line:
            lineNum = int(line.split(".java:")[1].split(":")[0])

            if allSnippetNums[0][(line.split(startSnippetfMRI))[1].split('.java')[0]] <= lineNum:
                data["Snippet"].append(f"f -- {str(list(allSnippetNums[0].keys()).index((line.split(startSnippetfMRI))[1].split('.java')[0]) + 1)} -- {(line.split(startSnippetfMRI))[1].split('.java')[0]}")
                data["Warning Type"].append(line.split(endSnippet)[1].strip())
        elif startSnippetCOG1 in line and endSnippet in line:
            lineNum = int(line.split(".java:")[1].split(":")[0])

            for i in range(len(allSnippetNums[1]) - 1):
                if allSnippetNums[1][i] <= lineNum and allSnippetNums[1][i + 1] > lineNum:
                    data["Snippet"].append(f"1 -- {str(i + 1)}")
                    data["Warning Type"].append(line.split(endSnippet)[1].strip())

                    # Additional check to see if the snippet is from dataset 2 as well (which is a subset of 1)
                    for j in range(0, len(allSnippetNums[2]) - 1, 2):
                        if allSnippetNums[2][j] <= lineNum and allSnippetNums[2][j + 1] > lineNum:
                            data["Snippet"].append(f"2 -- {str((j + 2) // 2)}")
                            data["Warning Type"].append(line.split(endSnippet)[1].strip())
                            break
                    break
        elif startSnippetCOG3 in line and endSnippet in line:
            lineNum = int(line.split(".java:")[1].split(":")[0])
            fileName = line.split(".java:")[0].rsplit("/", 1)[1]

            if fileName not in allSnippetNums[3]:
                continue

            snippetNums = allSnippetNums[3][fileName]

            addToI = 0
            if fileName == "Tasks_2":
                addToI = len(allSnippetNums[3]["Tasks_1"]) - 1
            elif fileName == "Tasks_3":
                addToI = len(allSnippetNums[3]["Tasks_1"]) - 1
                addToI += len(allSnippetNums[3]["Tasks_2"]) - 1

            for i in range(len(snippetNums) - 1):
                if snippetNums[i] <= lineNum and snippetNums[i + 1] > lineNum:
                    data["Snippet"].append(f"3 -- {str(i + 1 + addToI)}")
                    data["Warning Type"].append(line.split(endSnippet)[1].strip())
                    break
        elif startSnippetCOG6 in line and endSnippet in line:
            lineNum = int(line.split(".java:")[1].split(":")[0])
            fileName = line.split(".java:")[0].rsplit("/", 1)[1]
            if fileName not in allSnippetNums[4]:
                continue

            snippetNums = allSnippetNums[4][fileName]
 
            for i in range(len(snippetNums) - 1):
                if snippetNums[i] <= lineNum and snippetNums[i + 1] > lineNum:
                    data["Snippet"].append(f"6 -- {str(i + 1)} -- {fileName}")
                    data["Warning Type"].append(line.split(endSnippet)[1].strip())

                    break
        elif startSnippetCOG9 in line and endSnippet in line:
            lineNum = int(line.split(".java:")[1].split(":")[0])

            for i in range(0, len(allSnippetNums[5]) - 1, 2):
                if allSnippetNums[5][i] <= lineNum and allSnippetNums[5][i + 1] > lineNum:
                    data["Snippet"].append(f"9_gc -- {str((i + 2) // 2)}")
                elif allSnippetNums[6][i] <= lineNum and allSnippetNums[6][i + 1] > lineNum:
                    data["Snippet"].append(f"9_bc -- {str((i + 2) // 2)}")
                elif allSnippetNums[7][i] <= lineNum and allSnippetNums[7][i + 1] > lineNum:
                    data["Snippet"].append(f"9_nc -- {str((i + 2) // 2)}")
                else:
                    continue

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

def csvAllSnippetNames(allSnippetNums):
    info = {
        "dataset": [],
        "snippet": []
    }

    for dataset in allSnippetNums:
        pass

if __name__ == "__main__":
    allSnippetNums = getAllSnippets()

    #csvAllSnippetNames(allSnippetNums)

    # Data output structure as a dictionary
    data = {
        "Snippet": [],
        "Warning Type": []
    }
    allAnalysisToolData = []
    openJMLTimeouts = []

    #TODO: All analysis tools go here
    allAnalysisToolData.append(parseCheckerFramework(copy.deepcopy(data), allSnippetNums))
    allAnalysisToolData.append(parseTypestateChecker(copy.deepcopy(data), allSnippetNums))
    allAnalysisToolData.append(parseInfer(copy.deepcopy(data), allSnippetNums))
    allAnalysisToolData.append(parseOpenJML(copy.deepcopy(data), allSnippetNums, openJMLTimeouts))

    setupCSVSheets(allAnalysisToolData)

    getNumTimeoutsPerDataset(openJMLTimeouts)

    openJMLHandleTimeouts(openJMLTimeouts)