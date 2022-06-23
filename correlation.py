import pandas as pd
from openpyxl import Workbook, load_workbook

# Read all the data output excel sheets from the various analysis tool 
# and create a set of all the unique snippets across all the datasets that contain warnings
def getSnippetsWithWarnings():
    df = pd.read_excel('checker_framework_data.xlsx', usecols=['Snippet'])
    uniqueSnippets = []

    listSnippets = df['Snippet'].to_list()
    uniqueSnippets.extend(list(set(listSnippets)))

    return uniqueSnippets

# Gets a list of all the datasets that snippets can come from
def getDatasets():
    df = pd.read_excel('correlation_analysis.xlsx', usecols=['Complexity Metric'])

    return df['Complexity Metric'].to_list()

# Gets a count of the number of snippets that contain warnings for each dataset
def sortUniqueSnippetsByDataset(datasets, uniqueSnippets):
    countSnippetsPerDataset = dict([(key.split("-")[1].strip(), 0) for key in datasets])
 
    for snippet in uniqueSnippets:
        snippet = snippet.split("-")[0].strip()

        if snippet in countSnippetsPerDataset:
            countSnippetsPerDataset[snippet] += 1

    return countSnippetsPerDataset

# Sets the values of the column "# of snippets with warnings" in the correlation analysis excel sheet
def setNumSnippetsWithWarningsColumn(countSnippetsPerDataset):
    #df = pd.read_excel('correlation_analysis.xlsx')

    workbook = load_workbook('correlation_analysis.xlsx')
    worksheet = workbook.active # gets first sheet

    for i, value in enumerate(countSnippetsPerDataset.values()):
        # Writes a new value PRESERVING cell styles.
        worksheet.cell(row=i + 2, column=3, value=value)

    workbook.save('correlation_analysis.xlsx')

if __name__ == '__main__':
    setNumSnippetsWithWarningsColumn(sortUniqueSnippetsByDataset(getDatasets(), getSnippetsWithWarnings()))