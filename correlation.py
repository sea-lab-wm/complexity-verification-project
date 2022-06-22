import pandas as pd

# Read all the data output excel sheets from the various analysis tool 
# and create a set of all unique snippets across all the datasets that contain warnings
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

def sortUniqueSnippetsByDataset(datasets, uniqueSnippets):
    pass

if __name__ == '__main__':
    getSnippetsWithWarnings()
    getDatasets()