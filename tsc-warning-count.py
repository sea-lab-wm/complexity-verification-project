import os
import sys
import re
from typing import List, Callable

class WarningCount:
    def __init__(self, warningString : str) -> None:
        self.warningString = warningString
        self.count = 1
        pass
    
    def increaseCount(self):
        self.count += 1
    
    def __eq__(self, value: object) -> bool:
        if isinstance(value, WarningCount):
            return self.warningString == value.warningString
        return False
    def __repr__(self) -> str:
        return "{} - NUMBER OF DUPLICATES: {}".format(self.warningString, self.count)



def parseWarning(lines : list[str], startIndex : int) -> str:
    #Read the next 3 lines to create a single string that represents the whole warning
    warning = "";
    for i in range(startIndex, startIndex + 3):
        warning += lines[i]
    
    return warning

def ds6IncreaseOrAppend(lines: list[str], i: int):
    warning = WarningCount(parseWarning(lines, i))
    if warning in ds6Warnings:
        index = ds6Warnings.index(warning)
        ds6Warnings[index].increaseCount()
    else:
        ds6Warnings.append(warning)

def extract_number(s: str) -> int:
    match = re.search(r'\.java:(\d+):', s)
    if match:
        return int(match.group(1))
    raise ValueError("No matching number found")

# def ds6Weka(callable: Callable[[List[str], int], None]):
def processDs6(lines: list[str], line: str, index: int):

    # Contains a list of the prefixes we should be looking at and their corresponding lines where snippets start and end
    prefixesAndLines = [
        ["/weka/estimators/CheckEstimator", [218-1, 262-1]],
        ["/weka/estimators/EstimatorUtils", [55-1, 96-1]],
        ["/weka/gui/beans/ClassifierPerformanceEvaluatorCustomizer", [52-1, 97-1]],
        ["/weka/gui/beans/ModelPerformanceChart", [73-1, 114-1]],
        ["/weka/gui/experiment/GeneratorPropertyIteratorPanel", [75-1, 111-1]],
        ["/Antlr4Master", [52-1, 299-1]],
        ["/CarReport", [50-1, 276-1]],
        ["/HibernateORM", [37-1, 271-1]],
        ["/K9", [35-1, 229-1]],
        ["/MyExpenses", [46-1, 286-1]],
        ["/OpenCMSCore", [46-1, 278-1]],
        ["/Phoenix", [64-1, 306-1]],
        ["/SpringBatch", [61-1, 249-1]],
    ]

    for pl in prefixesAndLines:
        if line.startswith(ds6Prefix + pl[0]):
            if line.startswith("dataset6/src/main/java/CarReport.java:399:"):
                pass
            codeLineNumber = extract_number(line)
            if codeLineNumber >= pl[1][0] and codeLineNumber <= pl[1][1]:
                ds6IncreaseOrAppend(lines, index)
    


def openAndRead(filePath : str):
    with open(filePath) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            # Check if it is a DS1 file
            if line.startswith(ds1Prefix):
                warning = WarningCount(parseWarning(lines, i))
                if warning in ds1Warnings:
                    index = ds1Warnings.index(warning)
                    ds1Warnings[index].increaseCount()
                else:
                    ds1Warnings.append(warning)
            # Check if it is a DS3 file
            if line.startswith(ds3Prefix):
                warning = WarningCount(parseWarning(lines, i))
                if warning in ds3Warnings:
                    index = ds3Warnings.index(warning)
                    ds3Warnings[index].increaseCount()
                else:
                    ds6Warnings.append(warning)
            # Check if it is a DS6 file
            # Only DS6 has this dedicated method since it's the only dataset with issues.
            processDs6(lines, line, i)

            # Check if it is a DS9 file
            if line.startswith(ds9Prefix):
                warning = WarningCount(parseWarning(lines, i))
                if warning in ds9Warnings:
                    index = ds9Warnings.index(warning)
                    ds9Warnings[index].increaseCount()
                else:
                    ds9Warnings.append(warning)
            # Check if it is a DSf file
            if line.startswith(dsfPrefix):
                warning = WarningCount(parseWarning(lines, i))
                if warning in dsfWarnings:
                    index = dsfWarnings.index(warning)
                    dsfWarnings[index].increaseCount()
                else:
                    dsfWarnings.append(warning)

if __name__ == "__main__":
    filePaths = [
        "data/typestate_checker_output_cog_dataset_1.txt",
        "data/typestate_checker_output_cog_dataset_3.txt",
        "data/typestate_checker_output_cog_dataset_6.txt",
        "data/typestate_checker_output_cog_dataset_9.txt",
        "data/typestate_checker_output_fMRI_dataset.txt",
    ]

    ds1Warnings: List[WarningCount] = []
    ds3Warnings: List[WarningCount] = []
    ds6Warnings: List[WarningCount] = []
    ds9Warnings: List[WarningCount] = []
    dsfWarnings: List[WarningCount] = []

    ds1Prefix = "simple-datasets/src/main/java/cog_complexity_validation_datasets/One"
    ds3Prefix = "simple-datasets/src/main/java/cog_complexity_validation_datasets/Three"
    ds6Prefix = "dataset6/src/main/java"
    ds9Prefix = "dataset9/src/main/java"
    dsfPrefix = "simple-datasets/src/main/java/fMRI_Study_Classes"


    for path in filePaths:
        openAndRead(path)
    
    # The isinstance is only to have type helping when writing. If there's a simpler way to do this please let me know
    ds1FilteredWarnings: List[WarningCount] = list(filter(lambda x: isinstance(x, WarningCount) and x.count > 1, ds1Warnings))
    ds3FilteredWarnings: List[WarningCount] = list(filter(lambda x: isinstance(x, WarningCount) and x.count > 1, ds3Warnings))
    ds6FilteredWarnings: List[WarningCount] = list(filter(lambda x: isinstance(x, WarningCount) and x.count > 1, ds6Warnings))
    ds9FilteredWarnings: List[WarningCount] = list(filter(lambda x: isinstance(x, WarningCount) and x.count > 1, ds9Warnings))
    dsfFilteredWarnings: List[WarningCount] = list(filter(lambda x: isinstance(x, WarningCount) and x.count > 1, dsfWarnings))

    print("ds1 count {}".format(len(ds1FilteredWarnings)))
    print("ds3 count {}".format(len(ds3FilteredWarnings)))
    print("ds6 count {}".format(len(ds6FilteredWarnings)))
    print("ds9 count {}".format(len(ds9FilteredWarnings)))
    print("dsf count {}".format(len(dsfFilteredWarnings)))

    with open("tsc-warning-count.txt", "w") as output:
        # Only ds6 is being saved since it is the only dataset that is filtering the warnings that are outside of the actual snippets.
        # for w in ds1FilteredWarnings:
        #     output.write(repr(w) + "\n\n")
        # for w in ds3FilteredWarnings:
        #     output.write(repr(w) + "\n\n")
        for w in ds6FilteredWarnings:
            output.write(repr(w) + "\n\n")
        # for w in ds9FilteredWarnings:
        #     output.write(repr(w) + "\n\n")
        # for w in dsfFilteredWarnings:
        #     output.write(repr(w) + "\n\n")

    # for warning in ds6FilteredWarnings:
    #     print(warning)