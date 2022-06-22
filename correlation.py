import pandas as pd

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

def setNumSnippetsWithWarningsColumn(countSnippetsPerDataset):
    pass

if __name__ == '__main__':
    sortUniqueSnippetsByDataset(getDatasets(), getSnippetsWithWarnings())