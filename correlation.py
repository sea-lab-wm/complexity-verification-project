import pandas as pd
import scipy.stats as scpy
import copy

# OSCAR: I think, overall, it is easier to process data with data frames:
# read data from the excel files into DFs, then processing it to get results as more DFs, and then writing this DFs
# to either an excel or CSV file
# RESOURCES:
# https://www.geeksforgeeks.org/python-pandas-dataframe/
# https://pandas.pydata.org/docs/user_guide/indexing.html

###############################################
#   Interact with correlation_analysis.xlsx   #
###############################################

# Returns the dataframe of a specific excel sheet.
def readCorrelationAnalysis(sheetName):
    return pd.read_excel("data/correlation_analysis.xlsx", sheet_name=sheetName)

# Saves the correlation analysis dataframe to its excel sheet.
def writeCorrelationAnalysis(allCorrelationAnalysisDFS):
    #correlationAnalysisDF.to_excel("data/correlation_analysis.xlsx", engine="xlsxwriter", sheet_name=sheetName, index=False)

    with pd.ExcelWriter("data/correlation_analysis.xlsx") as writer:
        # TODO: Add more analysis tool sheets here ...
        allCorrelationAnalysisDFS[0].to_excel(writer, sheet_name="all_tools", index=False)
        allCorrelationAnalysisDFS[1].to_excel(writer, sheet_name="checker_framework", index=False)
        allCorrelationAnalysisDFS[2].to_excel(writer, sheet_name="typestate_checker", index=False)

# For Reference: The columns "Complexity Metric" and "# of snippets judged (complexity)" are added manually.

# Sets the values of the column "# of snippets with warnings" in the correlation analysis dataframe.
# Returns the modified correlation analysis dataframe.
def setNumSnippetsWithWarningsColumn(dfListAnalysisTools, correlationAnalysisDF):
    datasets = correlationAnalysisDF.iloc[:, 0]  # A list of all datasets being used

    # A count of the number of snippets that contain warnings for each dataset
    countSnippetsPerDataset = sortUniqueSnippetsByDataset(datasets, getSnippetsWithWarnings(dfListAnalysisTools))

    for i in range(len(correlationAnalysisDF.index)):
        dataset = correlationAnalysisDF.iloc[i, 0].split("-")[1].strip()

        if dataset in countSnippetsPerDataset:
            correlationAnalysisDF.iloc[i, 2] = countSnippetsPerDataset[dataset]

    return correlationAnalysisDF

# Sets the values of the column "# of warnings" in the correlation analysis dataframe.
# Returns the modified correlation analysis dataframe.
def setNumWarningsColumn(warningsPerDataset, correlationAnalysisDF):
    # Loop through each dataset
    for i, numWarnings in enumerate(warningsPerDataset):
        correlationAnalysisDF.iloc[i, 3] = numWarnings

    uniqueDatasetCount = 0

    for i in range(len(correlationAnalysisDF.index)):
        if i - 1 >= 0:
            # Condition where dataset we are looking at is the same as the previous one
            if correlationAnalysisDF.iloc[i, 0].split("-")[1].strip() == correlationAnalysisDF.iloc[i - 1, 0].split("-")[1].strip():
                correlationAnalysisDF.iloc[i, 3] = correlationAnalysisDF.iloc[i - 1, 3]
                continue
        
        correlationAnalysisDF.iloc[i, 3] = warningsPerDataset[uniqueDatasetCount]
        uniqueDatasetCount += 1

    return correlationAnalysisDF

# Sets the values of the column "# of datapoints of correlation" in the correlation analysis dataframe.
# Returns the modified correlation analysis dataframe.
def setNumDatapointsForCorrelationColumn(dfListCorrelationDatapoints, correlationAnalysisDF):
    # TODO: TEMPORARY
    correlationAnalysisDF.iloc[0, 4] = "TEMP"

    # Loop through each set of datapoints, there is one for each study/metric
    for i, df in enumerate(dfListCorrelationDatapoints):
        numDataPoints = len(df)

        # TODO: THE + 1 IS TEMPORARY
        correlationAnalysisDF.iloc[i + 1, 4] = numDataPoints

    return correlationAnalysisDF

# Sets the values of the columns "Kendall"s Tau" and "Kendall"s p-Value" in the correlation analysis dataframe.
# Returns the modified correlation analysis dataframe.
def setKendallTauColumns(kendallTauVals, correlationAnalysisDF):
    correlationAnalysisDF.iloc[:, 5] = kendallTauVals[0]
    correlationAnalysisDF.iloc[:, 6] = kendallTauVals[1]

    return correlationAnalysisDF

# Sets the values of the columns "Spearman"s Rho" and "Spearman"s p-Value" in the correlation analysis dataframe.
# Returns the modified correlation analysis dataframe.
def setSpearmanRhoColumns(spearmanRhoVals, correlationAnalysisDF):
    correlationAnalysisDF.iloc[:, 7] = spearmanRhoVals[0]
    correlationAnalysisDF.iloc[:, 8] = spearmanRhoVals[1]

    return correlationAnalysisDF

##############################
#   Setup Correlation Data   #
##############################

# Master function for compiling the datapoints for correlation.
# Returns a list of dataframes where each dataframe contains the datapoints (x = complexity metric, y = # of warnings) for a specific dataset.
def setupCorrelationData(warningsPerSnippetPerDataset):
    # Datapoint structure as a dictionary, each row represents a snippet in numerical order -> first row for first snippet etc. excluding headers
    data = {
        "Metric": [],
        "Warning Count": []
    }
    dfListCorrelationDatapoints = []

    # TODO Add more datasets here ...

    # Compile datapoints for the COG Dataset 3 Study
    dfListCorrelationDatapoints.append(setCogDataset3Datapoints(warningsPerSnippetPerDataset["COG Dataset 3"], copy.deepcopy(data)))

    # Compile datapoints for the fMRI Study
    fmriDatapoints = setFMRIStudyDatapoints(warningsPerSnippetPerDataset["fMRI Dataset"], data)
    dfListCorrelationDatapoints.append(fmriDatapoints[0])
    dfListCorrelationDatapoints.append(fmriDatapoints[1])
    dfListCorrelationDatapoints.append(fmriDatapoints[2])

    return dfListCorrelationDatapoints

# TODO: functions for future datasets go here ...

# Gets a list of complexity metrics and a list of warning counts for each snippet in COG Dataset 3.
# Adds that data to a dictionary that is then converted to a dataframe.
def setCogDataset3Datapoints(warningsPerSnippet, data):
    data["Metric"] = readCOGDataset3StudyMetrics()
    data["Warning Count"] = warningsPerSnippet

    return pd.DataFrame(data)

def setFMRIStudyDatapoints(warningsperSnippet, data):
    dataCorrectness = copy.deepcopy(data)
    dataTime = copy.deepcopy(data)
    dataSubjComplexity = copy.deepcopy(data)
    metrics = readFMRIStudyMetrics()

    dataCorrectness["Metric"] = metrics[0]
    dataCorrectness["Warning Count"] = warningsperSnippet
    dataTime["Metric"] = metrics[1]
    dataTime["Warning Count"] = warningsperSnippet
    dataSubjComplexity["Metric"] = metrics[2]
    dataSubjComplexity["Warning Count"] = warningsperSnippet

    return (pd.DataFrame(dataCorrectness), pd.DataFrame(dataTime), pd.DataFrame(dataSubjComplexity))

##################################
#   Retrieve Data From Studies   #
##################################

# Reads the results of the cog data set 3 study. It contains 120 people who rated 100 snippets on a scale of 1-5.
# 1 being less readable and 5 being more readable.
# TODO:
# OSCAR: where are we filtering out the 4 snippets that are commented out?
# OSCAR: in cog_dataset_3.csv, are the snippets identified by column index?
def readCOGDataset3StudyMetrics():
    df = pd.read_csv("data/cog_dataset_3.csv")

    # Returns a list of the averages for each snippet
    return [round(sum(df[column]) / len(df[column]), 2) for column in df.columns[2:]]

# Reads the results of the fMRI study. It contains 19 people who looked at 16 snippets.
# Correctness (in %), time to solve (in sec.), and a subjective rating were all measured.
# Subjective rating of low, medium, or high.
def readFMRIStudyMetrics():
    correctness = [0] * 16
    times = [0] * 16
    subjComplexity = [0] * 16

    dfBehavioral = pd.read_csv("data/fmri_dataset_behavioral.csv")
    dfSubjective = pd.read_csv("data/fmri_dataset_subjective.csv")

    count = 0   # Keeps track of which snippet we are on (0-15)
    numParticipants = 0
    for i in range(len(dfBehavioral.index)):
        if pd.isnull(dfBehavioral.iloc[i, 12]):
            continue
        if i - 1 >= 0:
            if dfBehavioral.iloc[i, 3] == dfBehavioral.iloc[i - 1, 3]:
                correctness[count] += dfBehavioral.iloc[i, 7]
                times[count] += dfBehavioral.iloc[i, 12] / 1000  # Converting from milliseconds to seconds
                numParticipants += 1
            else:
                numParticipants += 1
                correctness[count] = (correctness[count] / numParticipants) * 100
                times[count] = times[count] / numParticipants
                numParticipants = 0

                count += 1
                correctness[count] += dfBehavioral.iloc[i, 7]
                times[count] += dfBehavioral.iloc[i, 12] / 1000  # Converting from milliseconds to seconds
        else:
            correctness[count] += dfBehavioral.iloc[i, 7]
            times[count] += dfBehavioral.iloc[i, 12] / 1000  # Converting from milliseconds to seconds
            numParticipants += 1

    count = 0   # Keeps track of which snippet we are on (0-15)
    numParticipants = 0
    for i in range(len(dfSubjective.index)):
        if i - 1 >= 0:
            if dfSubjective.iloc[i, 1] == dfSubjective.iloc[i - 1, 1]:
                subjComplexity[count] += dfSubjective.iloc[i, 2]
                numParticipants += 1
            else:
                numParticipants += 1
                subjComplexity[count] = subjComplexity[count] / numParticipants
                numParticipants = 0

                count += 1
                subjComplexity[count] += dfSubjective.iloc[i, 2]
        else:
            subjComplexity[count] += dfSubjective.iloc[i, 2]
            numParticipants += 1

    return (correctness, times, subjComplexity)


###############################################
#   Retrieve Data From Analysis Tool Output   #
###############################################

# Reads in warning per snippet data from each analysis tool.
# There is data for one analysis tool per csv file.
# The data frames for each file are returned in a list.
def readAnalysisToolOutput():
    dfList = []

    #TODO: Add more analysis tool output here
    dfList.append(pd.read_csv("data/checker_framework_data.csv"))
    dfList.append(pd.read_csv("data/typestate_checker_data.csv"))

    return dfList

# Read all the data output data frames from the various analysis tool 
# and create a list of all the unique snippets across all the datasets that contain warnings
def getSnippetsWithWarnings(dfListAnalysisTools):
    uniqueSnippets = []

    # Case where we are only looking at a single analysis tool at a time
    if isinstance(dfListAnalysisTools, pd.core.frame.DataFrame):
        listSnippets = dfListAnalysisTools["Snippet"].to_list()
        uniqueSnippets.extend(listSnippets)
        uniqueSnippets = list(set(uniqueSnippets))
        
        return uniqueSnippets

    for df in dfListAnalysisTools:
        listSnippets = df["Snippet"].to_list()
        uniqueSnippets.extend(listSnippets)

    uniqueSnippets = list(set(uniqueSnippets))

    # Name of snippets in "uniqueSnippets" format example: COG Dataset 1 - 12
    #                                              format: Dataset Name - Snippet #
    return uniqueSnippets

# Gets a count of the number of snippets that contain warnings for each dataset
# Returns a dictionary where the keys is the names of data sets. The values are an integer count of 
# the number of snippets that contain warnings for that data set.
def sortUniqueSnippetsByDataset(datasets, uniqueSnippets):
    countSnippetsPerDataset = dict([(key.split("-")[1].strip(), 0) for key in datasets])

    for snippet in uniqueSnippets:
        snippet = snippet.split("-")[0].strip() # Name of snippets in "uniqueSnippets" format example: COG Dataset 1 - 12
                                                #                                              format: Dataset Name - Snippet #
        for key in countSnippetsPerDataset:
            if snippet in key:
                countSnippetsPerDataset[key] += 1

    return countSnippetsPerDataset

# Determines the number of warnings for each snippet, separated by the dataset the snippet is from.
# Creates a dictionary where the keys are the names of data sets. Values are a list where the size is 
# the TOTAL number of snippets in the dataset and values within the list are the number of warnings for a given snippet.
def getNumWarningsPerSnippetPerDataset(dfListAnalysisTools, correlationAnalysisDF):
    # Gets data from the dataframe corresponding to correlation_analysis.xlsx
    datasets = correlationAnalysisDF.iloc[:,0]  # A list of all datasets being used
    numSnippetsJudgedPerDataset = correlationAnalysisDF.iloc[:,1]   # A list of the number of snippets in each dataset

    # Setup the dictionary with its keys
    warningsPerSnippetPerDataset = dict([(key.split("-")[1].strip(), 0) for key in datasets])

    # Setup the dictionary with empty lists for its values
    # Size of the list corresponds to the total number of snippets in that dataset
    count = 0
    for dataset in warningsPerSnippetPerDataset:
        warningsPerSnippetPerDataset[dataset] = [0] * int(numSnippetsJudgedPerDataset[count])
        count += 1

    # Case where we are only looking at a single analysis tool at a time
    if isinstance(dfListAnalysisTools, pd.core.frame.DataFrame):
        numWarnings = dfListAnalysisTools.sum(axis=1, numeric_only=True).tolist()
        snippetNames = dfListAnalysisTools["Snippet"].to_list()

        if len(snippetNames) != len(numWarnings):
            raise Exception("Number of snippets does not match number of warnings associated with said snippets") 

        for i in range(len(snippetNames)):
            snippetDataset = snippetNames[i].split("-")[0].strip()
            snippetNumber = snippetNames[i].split("-")[1].strip()

            warningsPerSnippetPerDataset[snippetDataset][int(snippetNumber) - 1] += numWarnings[i]

        return warningsPerSnippetPerDataset

    # Loop through the analysis tool output dataframes
    for df in dfListAnalysisTools:
        numWarnings = df.sum(axis=1, numeric_only=True).tolist()
        snippetNames = df["Snippet"].to_list()

        if len(snippetNames) != len(numWarnings):
            raise Exception("Number of snippets does not match number of warnings associated with said snippets") 

        for i in range(len(snippetNames)):
            snippetDataset = snippetNames[i].split("-")[0].strip()
            snippetNumber = snippetNames[i].split("-")[1].strip()

            warningsPerSnippetPerDataset[snippetDataset][int(snippetNumber) - 1] += numWarnings[i]

    return warningsPerSnippetPerDataset

# Determines the number of warnings for each dataset
def getNumWarningsPerDataset(warningsPerSnippetPerDataset):
    return [sum(warningsPerSnippetPerDataset[dataset]) for dataset in warningsPerSnippetPerDataset]

############################
#   Perform Correlations   #
############################

# Perform Kendall"s Tau correlation on each dataset seperatly where datapoints are: x = complexity metric, y = # of warnings
# Return a list of the correlation coefficients for each dataset.
def kendallTau(dfListCorrelationDatapoints):
    kendallTauVals = ([], [])

    #TODO TEMPORARY
    kendallTauVals[0].append("TEMP")
    kendallTauVals[1].append("TEMP")

    # Loop through every datapoint dataframe (corresponding to each dataset).
    for df in dfListCorrelationDatapoints:
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]

        corr, pValue = scpy.kendalltau(x, y)

        kendallTauVals[0].append(corr)
        kendallTauVals[1].append(pValue)

    return kendallTauVals

# Perform Spearman"s Rho correlation on each dataset seperatly where datapoints are: a = complexity metric, b = # of warnings
# Return a list of the correlation coefficients for each dataset.
def spearmanRho(dfListCorrelationDatapoints):
    spearmanRhoVals = ([], [])

    #TODO TEMPORARY
    spearmanRhoVals[0].append("TEMP")
    spearmanRhoVals[1].append("TEMP")

    # Loop through every datapoint dataframe (corresponding to each dataset).
    for df in dfListCorrelationDatapoints:
        a = df.iloc[:, 0]
        b = df.iloc[:, 1]

        corr, pValue = scpy.spearmanr(a, b)

        spearmanRhoVals[0].append(corr)
        spearmanRhoVals[1].append(pValue)

    return spearmanRhoVals

###########################
#   Program Starts Here   #
###########################

if __name__ == "__main__":
    # STEP 1 is in parser.py

    # STEP 2:
    # Read in all excel and csv sheets as dataframes
    dfListAnalysisTools = readAnalysisToolOutput()
    correlationAnalysisDFAllTools = readCorrelationAnalysis(sheetName="all_tools")
    # TODO: Add more analysis tools here and after each step below ...
    correlationAnalysisDFCheckerFramework = readCorrelationAnalysis(sheetName="checker_framework")
    correlationAnalysisDFTypestateChecker = readCorrelationAnalysis(sheetName="typestate_checker")

    # STEP 3:
    # Determine the number of snippets that contain warnings within each dataset.
    correlationAnalysisDFAllTools = setNumSnippetsWithWarningsColumn(dfListAnalysisTools, correlationAnalysisDFAllTools)
    correlationAnalysisDFCheckerFramework = setNumSnippetsWithWarningsColumn(dfListAnalysisTools[0], correlationAnalysisDFCheckerFramework)
    correlationAnalysisDFTypestateChecker = setNumSnippetsWithWarningsColumn(dfListAnalysisTools[1], correlationAnalysisDFTypestateChecker)

    # STEP 4:
    # Determine the number of warnings per snippet per dataset
    warningsPerSnippetPerDatasetAllTools = getNumWarningsPerSnippetPerDataset(dfListAnalysisTools, correlationAnalysisDFAllTools)
    warningsPerSnippetPerDatasetCheckerFramework = getNumWarningsPerSnippetPerDataset(dfListAnalysisTools[0], correlationAnalysisDFCheckerFramework)
    warningsPerSnippetPerDatasetTypestateChecker = getNumWarningsPerSnippetPerDataset(dfListAnalysisTools[1], correlationAnalysisDFTypestateChecker)

    # STEP 5:
    # Determine the number of warnings per dataset
    correlationAnalysisDFAllTools = setNumWarningsColumn(getNumWarningsPerDataset(warningsPerSnippetPerDatasetAllTools), correlationAnalysisDFAllTools)
    correlationAnalysisDFCheckerFramework = setNumWarningsColumn(getNumWarningsPerDataset(warningsPerSnippetPerDatasetCheckerFramework), correlationAnalysisDFCheckerFramework)
    correlationAnalysisDFTypestateChecker = setNumWarningsColumn(getNumWarningsPerDataset(warningsPerSnippetPerDatasetTypestateChecker), correlationAnalysisDFTypestateChecker)

    # STEP 6:
    # Compile all datapoints for correlation: x = complexity metric, y = # of warnings
    dfListCorrelationDatapointsAllTools = setupCorrelationData(warningsPerSnippetPerDatasetAllTools)
    dfListCorrelationDatapointsCheckerFramework = setupCorrelationData(warningsPerSnippetPerDatasetCheckerFramework)
    dfListCorrelationDatapointsTypestateChecker = setupCorrelationData(warningsPerSnippetPerDatasetTypestateChecker)
    # Update correlation analyis data frame 
    correlationAnalysisDFAllTools = setNumDatapointsForCorrelationColumn(dfListCorrelationDatapointsAllTools, correlationAnalysisDFAllTools)
    correlationAnalysisDFCheckerFramework = setNumDatapointsForCorrelationColumn(dfListCorrelationDatapointsCheckerFramework, correlationAnalysisDFCheckerFramework)
    correlationAnalysisDFTypestateChecker = setNumDatapointsForCorrelationColumn(dfListCorrelationDatapointsTypestateChecker, correlationAnalysisDFTypestateChecker)

    # STEP 7:
    # Perform Correlations
    kendallTauValsAllTools = kendallTau(dfListCorrelationDatapointsAllTools)
    correlationAnalysisDFAllTools = setKendallTauColumns(kendallTauValsAllTools, correlationAnalysisDFAllTools)
    kendallTauValsCheckerFramework = kendallTau(dfListCorrelationDatapointsCheckerFramework)
    correlationAnalysisDFCheckerFramework = setKendallTauColumns(kendallTauValsCheckerFramework, correlationAnalysisDFCheckerFramework)
    kendallTauValsTypestateChecker = kendallTau(dfListCorrelationDatapointsTypestateChecker)
    correlationAnalysisDFTypestateChecker = setKendallTauColumns(kendallTauValsTypestateChecker, correlationAnalysisDFTypestateChecker)

    spearmanRhoValsAllTools = spearmanRho(dfListCorrelationDatapointsAllTools)
    correlationAnalysisDFAllTools = setSpearmanRhoColumns(spearmanRhoValsAllTools, correlationAnalysisDFAllTools)
    spearmanRhoValsCheckerFramework = spearmanRho(dfListCorrelationDatapointsCheckerFramework)
    correlationAnalysisDFCheckerFramework = setSpearmanRhoColumns(spearmanRhoValsCheckerFramework, correlationAnalysisDFCheckerFramework)
    spearmanRhoValsTypestateChecker = spearmanRho(dfListCorrelationDatapointsTypestateChecker)
    correlationAnalysisDFTypestateChecker = setSpearmanRhoColumns(spearmanRhoValsTypestateChecker, correlationAnalysisDFTypestateChecker)

    # Update correlation_analysis.xlsx
    allCorrelationAnalysisDFS = [correlationAnalysisDFAllTools, correlationAnalysisDFCheckerFramework, correlationAnalysisDFTypestateChecker]
    writeCorrelationAnalysis(allCorrelationAnalysisDFS)
