import pandas as pd
from openpyxl import load_workbook

#
#   Retrieve Data From Analysis Tool Output
#

# Read all the data output excel sheets from the various analysis tool 
# and create a set of all the unique snippets across all the datasets that contain warnings
def getSnippetsWithWarnings():
    df = pd.read_excel('data/checker_framework_data.xlsx', usecols=['Snippet'])
    uniqueSnippets = []

    listSnippets = df['Snippet'].to_list()
    uniqueSnippets.extend(list(set(listSnippets)))

    return uniqueSnippets

# Gets a count of the number of snippets that contain warnings for each dataset
def sortUniqueSnippetsByDataset(datasets, uniqueSnippets):
    countSnippetsPerDataset = dict([(key.split("-")[1].strip(), 0) for key in datasets])
 
    for snippet in uniqueSnippets:
        snippet = snippet.split("-")[0].strip()

        if snippet in countSnippetsPerDataset:
            countSnippetsPerDataset[snippet] += 1

    return countSnippetsPerDataset

#
#   Retrieve Data From Studies
#

# Reads the results of the cog data set 3 study. It contains 120 people who rated 100 snippets on a scale of 1-5.
# 1 being less readable and 5 being more readable.
def getAveragesCogDataset3():
    df = pd.read_csv('data/cog_dataset_3.csv')

    # Returns a list of the averages for each snippet
    return [round(sum(df[column]) / len(df[column]), 2) for column in df.columns[2:]]

#
#   Interact with datapoints.xlsx
#

# Takes the averages of the complexity metric and places them into the 
# first column of dataapoints.xlsx sheet cog_dataset_3.
def setCogDataset3ComplexityMetricColumn(complexityMetrics):
    workbook = load_workbook('data/datapoints.xlsx')
    worksheet = workbook['cog_dataset_3']

    for i, value in enumerate(complexityMetrics):
        worksheet.cell(row=i + 1, column=1, value=value)

    workbook.save('data/datapoints.xlsx')

def setCogDataset3WarningCountColumn():
    pass

#
#   Interact with correlation_analysis.xlsx
#

# Gets a list of all the datasets that snippets can come from
def getDatasets():
    df = pd.read_excel('data/correlation_analysis.xlsx', usecols=['Complexity Metric'])

    return df['Complexity Metric'].to_list()

# Sets the values of the column "# of snippets with warnings" in the correlation analysis excel sheet
def setNumSnippetsWithWarningsColumn(countSnippetsPerDataset):
    workbook = load_workbook('data/correlation_analysis.xlsx')
    worksheet = workbook.active # gets first sheet

    for i, value in enumerate(countSnippetsPerDataset.values()):
        # Writes a new value PRESERVING cell styles.
        worksheet.cell(row=i + 2, column=3, value=value)

    workbook.save('data/correlation_analysis.xlsx')

def setNumDatapointsForCorrelationColumn(averages):
    pass

if __name__ == '__main__':
    setNumSnippetsWithWarningsColumn(sortUniqueSnippetsByDataset(getDatasets(), getSnippetsWithWarnings()))

    # Already set, no need to run multiple times
    #setCogDataset3ComplexityMetricColumn(getAveragesCogDataset3())