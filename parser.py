import pandas as pd
import os

# OSCAR: how are the tools run?

# list of snippets and their start lines in cog_dataset_3 Tasks.java
# OSCAR: why do we need these line numbers? my understanding is that the codes checks that the warnings fall within
# these numbers. Does it mean that the warning output file contains extra warnings we don't want to parse? I thought we would parse all the warnings in the file.
# OSCAR: update, I checked the correlation code, and I think the reason for these lines is that we need to identify the snippets for which we get warnings (i.e., get the snippet ID)
#cog_dataset_1_methods_lines = [14, 26, 47, 65, 82, 101, 110, 120, 132, 145, 159, 170, 184, 195, 206, 230, 238, 257, 281, 296, 311, 326, 341]
#cog_dataset_3_methods_lines = [212, 228, 245, 277, 289, 298, 314, 322, 332, 345, 361, 371, 384, 395, 409, 417, 425, 439, 451, 460, 468, 483, 492, 502, 513, 524, 538, 547, 558, 571, 580, 601, 614, 621, 638, 654, 666, 678, 689, 704, 714, 725, 736, 745, 753, 766, 781, 788, 803, 820, 832, 842, 852, 866, 876, 894, 901, 909, 917, 927, 934, 946, 957, 967, 976, 985, 1000, 1014, 1022, 1032, 1044, 1056, 1077, 1089, 1106, 1115, 1123, 1133, 1141, 1158, 1172, 1185, 1202, 1213, 1228, 1238, 1251, 1262, 1273, 1290, 1305, 1322, 1336, 1351, 1362, 1375, 1383, 1398, 1406, 1421, 1431]

#########################
#   Get Snippet Names   #
#########################

# Parses a .java file containing numbered snippets to determine which snippet starts on which line.
def getSnippetNames(filePath):
    methodStartLines = []

    with open(filePath) as f:
        for lineNum, line in enumerate(f, start=1):
            if 'SNIPPET_STARTS' in line:
                methodStartLines.append(lineNum + 1)
            elif 'SNIPPETS_END' in line:
                methodStartLines.append(lineNum)

    return methodStartLines

##################################
#   Parse Analysis Tool Output   #
##################################

# Retrieves the snippet name and warning message for each warning output by the Checker Framework.
def parseCheckerFramework(data, fMRIDatasetSnippetNames, cogDataset1SnippetNums, cogDataset3SnippetNums):
    lines = []
    with open('data/checker_framework_output.txt') as f:
        lines = f.readlines()

    # Delimeters with which to parse the warnings
    startSnippetfMRI = "/fMRI_Study_Classes/"
    startSnippetCOG1 = "/cog_complexity_validation_datasets/One/"
    startSnippetCOG3 = "/cog_complexity_validation_datasets/Three/"
    endSnippet = ": warning:"

    for line in lines:
        if startSnippetfMRI in line and endSnippet in line:
            data["Snippet"].append("fMRI Dataset - " + str(fMRIDatasetSnippetNames.index((line.split(startSnippetfMRI))[1].split(".java")[0]) + 1) + " - " + (line.split(startSnippetfMRI))[1].split(".java")[0])
            data["Warning Type"].append(line.split(endSnippet)[1].strip())
        elif startSnippetCOG1 in line and endSnippet in line:
            lineNum = int(line.split(".java:")[1].split(":")[0])

            for i in range(len(cogDataset1SnippetNums) - 1):
                if cogDataset1SnippetNums[i] <= lineNum and cogDataset1SnippetNums[i + 1] > lineNum:
                    data["Snippet"].append("COG Dataset 1 - " + str(i + 1))
                    data["Warning Type"].append(line.split(endSnippet)[1].strip())
                    break
        elif startSnippetCOG3 in line and endSnippet in line:
            lineNum = int(line.split(".java:")[1].split(":")[0])

            for i in range(len(cogDataset3SnippetNums) - 1):
                if cogDataset3SnippetNums[i] <= lineNum and cogDataset3SnippetNums[i + 1] > lineNum:
                    data["Snippet"].append("COG Dataset 3 - " + str(i + 1))
                    data["Warning Type"].append(line.split(endSnippet)[1].strip())
                    break
    
    return ('checker_framework_data', data)

# Retrieves the snippet name and warning message for each warning output by the Typestate Checker.
def parseTypestateChecker(data):


    pass

########################
#   Setup CSV Sheets   #
########################

# Creates csv files for each analysis tool's warning output.
def setupCSVSheets(allAnalysisToolData):
    for data in allAnalysisToolData:
        # Creates a table from the dictionary of data
        df = pd.DataFrame(data[1])

        # Convert data using pivot table
        df["Warning Type Copy"] = df["Warning Type"]    # Must copy values to column with different name to use same column for "values" and "columns" when making a pivot table
        df = df.pivot_table(values="Warning Type", index="Snippet", columns="Warning Type Copy", aggfunc="count")

        df.to_csv(f'data/{data[0]}.csv')

if __name__ == '__main__':
    fMRIDatasetSnippetNames = [name.split('.')[0] for name in os.listdir('src/main/java/edu/wm/kobifeldman/fMRI_Study_Classes')]
    fMRIDatasetSnippetNames = sorted(fMRIDatasetSnippetNames, key=str.lower)    # Keeps the order of these snippets consistent across operating systems
    cogDataset1SnippetNums = getSnippetNames('src/main/java/edu/wm/kobifeldman/cog_complexity_validation_datasets/One/Tasks.java')
    cogDataset3SnippetNums = getSnippetNames('src/main/java/edu/wm/kobifeldman/cog_complexity_validation_datasets/Three/Tasks.java')

    # Data output structure as a dictionary
    data = {
        "Snippet": [],
        "Warning Type": []
    }
    allAnalysisToolData = []

    #TODO: All analysis tools go here
    allAnalysisToolData.append(parseCheckerFramework(data, fMRIDatasetSnippetNames, cogDataset1SnippetNums, cogDataset3SnippetNums))

    setupCSVSheets(allAnalysisToolData)