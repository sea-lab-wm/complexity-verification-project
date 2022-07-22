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
        allCorrelationAnalysisDFS[3].to_excel(writer, sheet_name="infer", index=False)

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
    # Loop through each set of datapoints, there is one for each study/metric
    for i, df in enumerate(dfListCorrelationDatapoints):
        numDataPoints = len(df)

        correlationAnalysisDF.iloc[i, 4] = numDataPoints

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

    # Compile datapoints for the COG Dataset 1 Study
    cogDataset1Datapoints = setCogDataset1Datapoints(warningsPerSnippetPerDataset["COG Dataset 1"], copy.deepcopy(data))
    dfListCorrelationDatapoints.append(cogDataset1Datapoints[0])
    dfListCorrelationDatapoints.append(cogDataset1Datapoints[1])
    dfListCorrelationDatapoints.append(cogDataset1Datapoints[2])

    # Compile datapoints for the COG Dataset 2 Study
    cogDataset2Datapoints = setCogDataset2Datapoints(warningsPerSnippetPerDataset["COG Dataset 2"], copy.deepcopy(data))
    dfListCorrelationDatapoints.append(cogDataset2Datapoints[0])
    dfListCorrelationDatapoints.append(cogDataset2Datapoints[1])
    dfListCorrelationDatapoints.append(cogDataset2Datapoints[2])
    dfListCorrelationDatapoints.append(cogDataset2Datapoints[3])

    # Compile datapoints for the COG Dataset 3 Study
    dfListCorrelationDatapoints.append(setCogDataset3Datapoints(warningsPerSnippetPerDataset["COG Dataset 3"], copy.deepcopy(data)))

    # Compile datapoints for the COG Dataset 6 Study
    cogDataset6Datapoints = setCogDataset6Datapoints(warningsPerSnippetPerDataset["COG Dataset 6"], copy.deepcopy(data))
    dfListCorrelationDatapoints.append(cogDataset6Datapoints[0])
    dfListCorrelationDatapoints.append(cogDataset6Datapoints[1])
    dfListCorrelationDatapoints.append(cogDataset6Datapoints[2])

    # Compile datapoints for the COG Dataset 9 Study
    cogDataset9Datapoints = setCogDataset9Datapoints(warningsPerSnippetPerDataset["COG Dataset 9"], copy.deepcopy(data))
    dfListCorrelationDatapoints.append(cogDataset9Datapoints[0])
    dfListCorrelationDatapoints.append(cogDataset9Datapoints[1])
    dfListCorrelationDatapoints.append(cogDataset9Datapoints[2])
    dfListCorrelationDatapoints.append(cogDataset9Datapoints[3])

    # Compile datapoints for the fMRI Study
    fmriDatapoints = setFMRIStudyDatapoints(warningsPerSnippetPerDataset["fMRI Dataset"], data)
    dfListCorrelationDatapoints.append(fmriDatapoints[0])
    dfListCorrelationDatapoints.append(fmriDatapoints[1])
    dfListCorrelationDatapoints.append(fmriDatapoints[2])

    return dfListCorrelationDatapoints

# TODO: functions for future datasets go here ...

def setCogDataset1Datapoints(warningsPerSnippet, data):
    dataTime = copy.deepcopy(data)
    dataCorrectness = copy.deepcopy(data)
    dataSubjComplexity = copy.deepcopy(data)
    metrics = readCOGDataset1StudyMetrics()

    dataTime["Metric"] = metrics[0]
    dataTime["Warning Count"] = warningsPerSnippet
    dataCorrectness["Metric"] = metrics[1]
    dataCorrectness["Warning Count"] = warningsPerSnippet
    dataSubjComplexity["Metric"] = metrics[2]
    dataSubjComplexity["Warning Count"] = warningsPerSnippet

    return (pd.DataFrame(dataTime), pd.DataFrame(dataCorrectness), pd.DataFrame(dataSubjComplexity))

def setCogDataset2Datapoints(warningsPerSnippet, data):
    dataTime = copy.deepcopy(data)
    dataBA32 = copy.deepcopy(data)
    dataBA31post = copy.deepcopy(data)
    dataBA31ant = copy.deepcopy(data)
    metrics = readCOGDataset2StudyMetrics()

    dataTime["Metric"] = metrics[0]
    dataTime["Warning Count"] = warningsPerSnippet
    dataBA32["Metric"] = metrics[1]
    dataBA32["Warning Count"] = warningsPerSnippet
    dataBA31post["Metric"] = metrics[2]
    dataBA31post["Warning Count"] = warningsPerSnippet
    dataBA31ant["Metric"] = metrics[3]
    dataBA31ant["Warning Count"] = warningsPerSnippet

    return (pd.DataFrame(dataTime), pd.DataFrame(dataBA32), pd.DataFrame(dataBA31post), pd.DataFrame(dataBA31ant))

# Gets a list of complexity metrics and a list of warning counts for each snippet in COG Dataset 3.
# Adds that data to a dictionary that is then converted to a dataframe.
def setCogDataset3Datapoints(warningsPerSnippet, data):
    data["Metric"] = readCOGDataset3StudyMetrics()
    data["Warning Count"] = warningsPerSnippet

    return pd.DataFrame(data)

def setCogDataset6Datapoints(warningsPerSnippet, data):
    dataTime = copy.deepcopy(data)
    dataCorrectness = copy.deepcopy(data)
    dataRating = copy.deepcopy(data)
    metrics = readCOGDataset6StudyMetrics()

    dataTime["Metric"] = metrics[0]
    dataTime["Warning Count"] = warningsPerSnippet
    dataCorrectness["Metric"] = metrics[1]
    dataCorrectness["Warning Count"] = warningsPerSnippet
    dataRating["Metric"] = metrics[2]
    dataRating["Warning Count"] = warningsPerSnippet

    return (pd.DataFrame(dataTime), pd.DataFrame(dataCorrectness), pd.DataFrame(dataRating))

def setCogDataset9Datapoints(warningsPerSnippet, data):
    dataTime = copy.deepcopy(data)
    dataCorrectness = copy.deepcopy(data)
    dataRating1 = copy.deepcopy(data)
    dataRating2 = copy.deepcopy(data)
    metrics = readCOGDataset9StudyMetrics()

    dataTime["Metric"] = metrics[0]
    dataTime["Warning Count"] = warningsPerSnippet
    dataCorrectness["Metric"] = metrics[1]
    dataCorrectness["Warning Count"] = warningsPerSnippet
    dataRating1["Metric"] = metrics[2]
    dataRating1["Warning Count"] = warningsPerSnippet
    dataRating2["Metric"] = metrics[3]
    dataRating2["Warning Count"] = warningsPerSnippet

    return (pd.DataFrame(dataTime), pd.DataFrame(dataCorrectness), pd.DataFrame(dataRating1), pd.DataFrame(dataRating2))

def setFMRIStudyDatapoints(warningsPerSnippet, data):
    dataCorrectness = copy.deepcopy(data)
    dataTime = copy.deepcopy(data)
    dataSubjComplexity = copy.deepcopy(data)
    metrics = readFMRIStudyMetrics()

    dataCorrectness["Metric"] = metrics[0]
    dataCorrectness["Warning Count"] = warningsPerSnippet
    dataTime["Metric"] = metrics[1]
    dataTime["Warning Count"] = warningsPerSnippet
    dataSubjComplexity["Metric"] = metrics[2]
    dataSubjComplexity["Warning Count"] = warningsPerSnippet

    return (pd.DataFrame(dataCorrectness), pd.DataFrame(dataTime), pd.DataFrame(dataSubjComplexity))

##################################
#   Retrieve Data From Studies   #
##################################

# Reads the results of the first pilot study for COG dataset 1. It contains 41 people who looked at 23 snippets.
# Metrics include time to solve (in sec.), correctness where 0 = completely wrong, 1 = in part correct, 2 = completely correct, and
# subjective rating is on a scale of 0 through 4 where 0 = very difficult, 1 = difficult, 2 = medium, 3 = easy, 4 = very easy.
def readCOGDataset1StudyMetrics():
    times = [0] * 23
    correctness = [0] * 23
    subjComplexity = [0] * 23

    df = pd.read_excel("data/cog_dataset_1.xlsx")

    # Get time column for each snippet
    timeCols = df.iloc[:41, [df.columns.get_loc(f"{str(snippetNum)}::time") for snippetNum in range(1, 24)]]
    # Average the values of each column
    times = [val / 41 for val in timeCols.sum(axis=0)]

    # Get correctness column for each snippet
    correctnessCols = df.iloc[:41, [df.columns.get_loc(f"{str(snippetNum)}::Correct") for snippetNum in range(1, 24)]]
    # Average the values of each column
    correctness = [val / 41 for val in correctnessCols.sum(axis=0)]

    # Get subjective rating column for each snippet ("difficulty")
    subjComplexityCols = df.iloc[:41, [df.columns.get_loc(f"{str(snippetNum)}::Difficulty") for snippetNum in range(1, 24)]]
    # Average the values of each column
    subjComplexity = [val / 41 for val in subjComplexityCols.sum(axis=0)]

    return (times, correctness, subjComplexity)

# Reads the results of the study for COG dataset 2 . It contains 16 people who looked at 12 snippets.
# Note that dataset 2 is a subset of dataset 1. All snippets in dataset 2 are also in dataset 1.
# Metrics include time to solve, as well as 3 physiological metrics: BA32, BA31post, and BA31ant.
def readCOGDataset2StudyMetrics():
    times = [0] * 12

    dfTime = pd.read_csv("data/cog_dataset_2_response_times.csv")
    dfPhysiological = pd.read_csv("data/cog_dataset_2_physiological.csv")

    # Get time column for each snippet
    timeCols = dfTime.iloc[1:18, [val for val in range(1, 25, 2)]]

    # Average the values of each column
    times = [int(val) / 16 for val in timeCols.astype(int).sum(axis=0)]

    BA32 = dfPhysiological.iloc[:, 5]
    BA31post = dfPhysiological.iloc[:, 6]
    BA31ant = dfPhysiological.iloc[:, 7]

    return (times, BA32, BA31post, BA31ant)

# Reads the results of the cog data set 3 study. It contains 120 people who rated 100 snippets on a scale of 1-5.
# 1 being less readable and 5 being more readable.
# TODO:
# OSCAR: where are we filtering out the 4 snippets that are commented out?
# OSCAR: in cog_dataset_3.csv, are the snippets identified by column index?

def readCOGDataset3StudyMetrics():
    df = pd.read_csv("data/cog_dataset_3.csv")
    
    # Returns a list of the averages for each snippet
    return [round(sum(df[column]) / len(df[column]), 2) for column in df.columns[2:]]

# Reads the results of the cog data set 6 study. It contains 120 people who looked at 63 snippets with metrics based on time, correctness, and rating.
# IMPORTANT: Each participant was assigned randomly assigned about 8 snippets out of the 50. So not every person looked at every snippet.
# The metrics were Time Needed for Perceived Understandability (TNPU), Actual Understanding (AU), and Perceived Binary Understandability (PBU).
def readCOGDataset6StudyMetrics():
    times = []
    correctness = []
    rating = []

    df = pd.read_csv("data/cog_dataset_6.csv")
    
    participantsPerSnippet = 0
    participantsPerSnippetTNPU = 0  # Separate variable to keep track of the fact that some TNPU are NULL. We don't want to average with the participant count if we skipped over the NULL values
    lastSnippet = ""
    sumTNPU = 0
    sumAU = 0
    sumPBU = 0
    for row in df.itertuples():
        if row[3] != lastSnippet and row[0] != 0:
            # Moved onto new snippet. Get averages for previous snippet.
            times.append(sumTNPU / participantsPerSnippetTNPU)
            correctness.append(sumAU / participantsPerSnippet)
            rating.append(sumPBU / participantsPerSnippet)

            sumTNPU = 0
            sumAU = 0
            sumPBU = 0
            participantsPerSnippet = 0
            participantsPerSnippetTNPU = 0
        
        # Still on same snippet, on first snippet, or starting new snippet after getting the averages for the previous one.
        participantsPerSnippet += 1
        #124 = PBU, 125 = TNPU, 126 = AU
        if not pd.isnull(row[125]):
            participantsPerSnippetTNPU += 1
            sumTNPU += row[125]

        sumAU += row[126]
        sumPBU += row[124]
        lastSnippet = row[3]

        #print(row[124], row[125], row[126])

    # Get averages for last snippet
    times.append(sumTNPU / participantsPerSnippetTNPU)
    correctness.append(sumAU / participantsPerSnippet)
    rating.append(sumPBU / participantsPerSnippet)

    if len(times) != 50 and len(correctness) != 50 and len(rating) != 50:
        raise Exception

    return (times, correctness, rating)

# Reads the results of the cog data set 9 study. It contains 104 participants and 30 unique snippets (5 snippets each with varying quality of comments).
# Time, correctness, and rating were measured.
def readCOGDataset9StudyMetrics():
    times = []
    correctness = []
    rating1 = []
    rating2 = []

    df = pd.read_excel("data/cog_dataset_9.xlsx")

    participantsPerSnippet = 0
    lastSnippet = ""
    sumTime = 0
    sumCorrectness = 0
    sumRating1 = 0
    sumRating2 = 0
    for row in df.itertuples():
        if pd.isnull(row[19]):
            break

        if row[19] != lastSnippet and row[0] != 0:
            # Moved onto new snippet. Get averages for previous snippet.
            times.append(sumTime / (participantsPerSnippet * 2))
            correctness.append(sumCorrectness / participantsPerSnippet)
            rating1.append(sumRating1 / (participantsPerSnippet * 2))
            rating2.append(sumRating2 / participantsPerSnippet)

            sumTime = 0
            sumCorrectness = 0
            sumRating = 0
            participantsPerSnippet = 0
        
        # Still on same snippet, on first snippet, or starting new snippet after getting the averages for the previous one.
        participantsPerSnippet += 1
        #25 = Score R1, 62 = Score R2, 64 = Score Difference, 82 = Time Read, 83 = Time Completion, 86 = recall accuracy (acc)
        sumTime += row[82] + row[83]
        sumCorrectness += row[86]
        sumRating1 += row[25] + row[62]
        sumRating2 += row[62]
        lastSnippet = row[19]

    # Get averages for last snippet
    times.append(sumTime / (participantsPerSnippet * 2))
    correctness.append(sumCorrectness / participantsPerSnippet)
    rating1.append(sumRating1 / (participantsPerSnippet * 2))
    rating2.append(sumRating2 / participantsPerSnippet)

    if len(times) != 30 and len(correctness) != 30 and len(rating1) != 30 and len(rating2) != 30:
        raise Exception

    return (times, correctness, rating1, rating2)

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
    dfList.append(pd.read_csv("data/infer_data.csv"))

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

    datasetsUnique = []
    numSnippetsJudgedPerDatasetUnique = []

    for i, dataset in enumerate(datasets):
        if dataset.split("-")[1].strip() not in datasetsUnique:
            datasetsUnique.append(dataset.split("-")[1].strip())
            numSnippetsJudgedPerDatasetUnique.append(numSnippetsJudgedPerDataset[i])

    # Setup the dictionary with its keys
    warningsPerSnippetPerDataset = dict([(key, 0) for key in datasetsUnique])

    # Setup the dictionary with empty lists for its values
    # Size of the list corresponds to the total number of snippets in that dataset
    count = 0
    for dataset in warningsPerSnippetPerDataset:
        warningsPerSnippetPerDataset[dataset] = [0] * int(numSnippetsJudgedPerDatasetUnique[count])
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

        #print(warningsPerSnippetPerDataset)
        #print(snippetNames)
        #print(numWarnings)
        for i in range(len(snippetNames)):
            snippetDataset = snippetNames[i].split("-")[0].strip()
            snippetNumber = snippetNames[i].split("-")[1].strip()
            #print(snippetDataset)
            #print(snippetNumber)
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
    correlationAnalysisDFInfer = readCorrelationAnalysis(sheetName="infer")

    # STEP 3:
    # Determine the number of snippets that contain warnings within each dataset.
    correlationAnalysisDFAllTools = setNumSnippetsWithWarningsColumn(dfListAnalysisTools, correlationAnalysisDFAllTools)
    correlationAnalysisDFCheckerFramework = setNumSnippetsWithWarningsColumn(dfListAnalysisTools[0], correlationAnalysisDFCheckerFramework)
    correlationAnalysisDFTypestateChecker = setNumSnippetsWithWarningsColumn(dfListAnalysisTools[1], correlationAnalysisDFTypestateChecker)
    correlationAnalysisDFInfer = setNumSnippetsWithWarningsColumn(dfListAnalysisTools[2], correlationAnalysisDFInfer)

    # STEP 4:
    # Determine the number of warnings per snippet per dataset
    warningsPerSnippetPerDatasetAllTools = getNumWarningsPerSnippetPerDataset(dfListAnalysisTools, correlationAnalysisDFAllTools)
    warningsPerSnippetPerDatasetCheckerFramework = getNumWarningsPerSnippetPerDataset(dfListAnalysisTools[0], correlationAnalysisDFCheckerFramework)
    warningsPerSnippetPerDatasetTypestateChecker = getNumWarningsPerSnippetPerDataset(dfListAnalysisTools[1], correlationAnalysisDFTypestateChecker)
    warningsPerSnippetPerDatasetInfer = getNumWarningsPerSnippetPerDataset(dfListAnalysisTools[2], correlationAnalysisDFInfer)
    #print(warningsPerSnippetPerDatasetAllTools)
    # STEP 5:
    # Determine the number of warnings per dataset
    correlationAnalysisDFAllTools = setNumWarningsColumn(getNumWarningsPerDataset(warningsPerSnippetPerDatasetAllTools), correlationAnalysisDFAllTools)
    correlationAnalysisDFCheckerFramework = setNumWarningsColumn(getNumWarningsPerDataset(warningsPerSnippetPerDatasetCheckerFramework), correlationAnalysisDFCheckerFramework)
    correlationAnalysisDFTypestateChecker = setNumWarningsColumn(getNumWarningsPerDataset(warningsPerSnippetPerDatasetTypestateChecker), correlationAnalysisDFTypestateChecker)
    correlationAnalysisDFInfer = setNumWarningsColumn(getNumWarningsPerDataset(warningsPerSnippetPerDatasetInfer), correlationAnalysisDFInfer)

    # STEP 6:
    # Compile all datapoints for correlation: x = complexity metric, y = # of warnings
    dfListCorrelationDatapointsAllTools = setupCorrelationData(warningsPerSnippetPerDatasetAllTools)
    dfListCorrelationDatapointsCheckerFramework = setupCorrelationData(warningsPerSnippetPerDatasetCheckerFramework)
    dfListCorrelationDatapointsTypestateChecker = setupCorrelationData(warningsPerSnippetPerDatasetTypestateChecker)
    dfListCorrelationDatapointsInfer = setupCorrelationData(warningsPerSnippetPerDatasetInfer)
    
    # Update correlation analyis data frame 
    correlationAnalysisDFAllTools = setNumDatapointsForCorrelationColumn(dfListCorrelationDatapointsAllTools, correlationAnalysisDFAllTools)
    correlationAnalysisDFCheckerFramework = setNumDatapointsForCorrelationColumn(dfListCorrelationDatapointsCheckerFramework, correlationAnalysisDFCheckerFramework)
    correlationAnalysisDFTypestateChecker = setNumDatapointsForCorrelationColumn(dfListCorrelationDatapointsTypestateChecker, correlationAnalysisDFTypestateChecker)
    correlationAnalysisDFInfer = setNumDatapointsForCorrelationColumn(dfListCorrelationDatapointsInfer, correlationAnalysisDFInfer)

    # STEP 7:
    # Perform Correlations
    kendallTauValsAllTools = kendallTau(dfListCorrelationDatapointsAllTools)
    correlationAnalysisDFAllTools = setKendallTauColumns(kendallTauValsAllTools, correlationAnalysisDFAllTools)
    kendallTauValsCheckerFramework = kendallTau(dfListCorrelationDatapointsCheckerFramework)
    correlationAnalysisDFCheckerFramework = setKendallTauColumns(kendallTauValsCheckerFramework, correlationAnalysisDFCheckerFramework)
    kendallTauValsTypestateChecker = kendallTau(dfListCorrelationDatapointsTypestateChecker)
    correlationAnalysisDFTypestateChecker = setKendallTauColumns(kendallTauValsTypestateChecker, correlationAnalysisDFTypestateChecker)
    kendallTauValsInfer = kendallTau(dfListCorrelationDatapointsInfer)
    correlationAnalysisDFInfer = setKendallTauColumns(kendallTauValsInfer, correlationAnalysisDFInfer)

    spearmanRhoValsAllTools = spearmanRho(dfListCorrelationDatapointsAllTools)
    correlationAnalysisDFAllTools = setSpearmanRhoColumns(spearmanRhoValsAllTools, correlationAnalysisDFAllTools)
    spearmanRhoValsCheckerFramework = spearmanRho(dfListCorrelationDatapointsCheckerFramework)
    correlationAnalysisDFCheckerFramework = setSpearmanRhoColumns(spearmanRhoValsCheckerFramework, correlationAnalysisDFCheckerFramework)
    spearmanRhoValsTypestateChecker = spearmanRho(dfListCorrelationDatapointsTypestateChecker)
    correlationAnalysisDFTypestateChecker = setSpearmanRhoColumns(spearmanRhoValsTypestateChecker, correlationAnalysisDFTypestateChecker)
    spearmanRhoValsInfer = spearmanRho(dfListCorrelationDatapointsInfer)
    correlationAnalysisDFInfer = setSpearmanRhoColumns(spearmanRhoValsInfer, correlationAnalysisDFInfer)

    # Update correlation_analysis.xlsx
    allCorrelationAnalysisDFS = [correlationAnalysisDFAllTools, correlationAnalysisDFCheckerFramework, correlationAnalysisDFTypestateChecker, correlationAnalysisDFInfer]
    writeCorrelationAnalysis(allCorrelationAnalysisDFS)
