from fileinput import lineno
import pandas as pd

# list of snippets and their start lines in cog_dataset_3 Tasks.java
cog_dataset_1_methods_lines = [14, 26, 47, 65, 82, 101, 110, 120, 132, 145, 159, 170, 184, 195, 206, 230, 238, 257, 281, 296, 311, 326, 341]
cog_dataset_3_methods_lines = [212, 228, 245, 277, 289, 298, 314, 322, 332, 345, 361, 371, 384, 395, 409, 417, 425, 439, 451, 460, 468, 483, 492, 502, 513, 524, 538, 547, 558, 571, 580, 601, 614, 621, 638, 654, 666, 678, 689, 704, 714, 725, 736, 745, 753, 766, 781, 788, 803, 820, 832, 842, 852, 866, 876, 894, 901, 909, 917, 927, 934, 946, 957, 967, 976, 985, 1000, 1014, 1022, 1032, 1044, 1056, 1077, 1089, 1106, 1115, 1123, 1133, 1141, 1158, 1172, 1185, 1202, 1213, 1228, 1238, 1251, 1262, 1273, 1290, 1305, 1322, 1336, 1351, 1362, 1375, 1383, 1398, 1406, 1421, 1431]

# Retrieves the snippet name and warning message for each warning output by the Checker Framework
def parseCheckerFramework(data):
    lines = []
    with open('checker_framework_output.txt') as f:
        lines = f.readlines()

    # Delimeters with which to parse the warnings
    startSnippetfMRI = "\\fMRI_Study_Classes\\"
    startSnippetCOG1 = "\\cog_complexity_validation_datasets\\One\\"
    startSnippetCOG3 = "\\cog_complexity_validation_datasets\\Three\\"
    endSnippet = ": warning:"

    for line in lines:
        if startSnippetfMRI in line and endSnippet in line:
            data["Snippet"].append("fMRI Dataset - " + (line.split(startSnippetfMRI))[1].split(".java")[0])
            data["Warning Type"].append(line.split(endSnippet)[1].strip())
        elif startSnippetCOG1 in line and endSnippet in line:
            lineNum = int(line.split(".java:")[1].split(":")[0])

            for i in range(len(cog_dataset_1_methods_lines) - 1):
                if cog_dataset_1_methods_lines[i] < lineNum and cog_dataset_1_methods_lines[i + 1] > lineNum:
                    data["Snippet"].append("COG Dataset 1 - " + str(i + 1))
                    data["Warning Type"].append(line.split(endSnippet)[1].strip())
                    break
        elif startSnippetCOG3 in line and endSnippet in line:
            lineNum = int(line.split(".java:")[1].split(":")[0])

            for i in range(len(cog_dataset_3_methods_lines) - 1):
                if cog_dataset_3_methods_lines[i] < lineNum and cog_dataset_3_methods_lines[i + 1] > lineNum:
                    data["Snippet"].append("COG Dataset 3 - " + str(i + 1))
                    data["Warning Type"].append(line.split(endSnippet)[1].strip())
                    break
    
    #print(str(len(data["Snippet"])) + str(len(data["Warning Type"])))
    return data

def setupExcelSheet():
    # Data output structure as a dictionary
    data = {
        "Snippet": [],
        "Warning Type": []
    }

    # Creates a table from the dictionary of data
    df = pd.DataFrame(parseCheckerFramework(data))

    df["Warning Type Copy"] = df["Warning Type"]    # Must copy values to column with different name to use same column for "values" and "columns" when making a pivot table
    df = df.pivot_table(values="Warning Type", index="Snippet", columns="Warning Type Copy", aggfunc="count")

    df.to_excel('data/checker_framework_data.xlsx', engine='xlsxwriter')

if __name__ == '__main__':
    setupExcelSheet()