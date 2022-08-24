import pandas as pd
import scipy.stats as scpy
import copy
import numpy as np

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
    with pd.ExcelWriter("data/correlation_analysis.xlsx") as writer:
        for correlationAnalysisDF in allCorrelationAnalysisDFS:
            correlationAnalysisDF[1].to_excel(writer, sheet_name=correlationAnalysisDF[0], index=False)

        # Auto-adjust columns' width
        for correlationAnalysisDF in allCorrelationAnalysisDFS:
            for column in correlationAnalysisDF[1]:
                column_width = max(correlationAnalysisDF[1][column].astype(str).map(len).max(), len(column))
                col_idx = correlationAnalysisDF[1].columns.get_loc(column)
                writer.sheets[correlationAnalysisDF[0]].set_column(col_idx, col_idx, column_width)

# For Reference: The columns "Complexity Metric" and "# of snippets judged (complexity)" are added manually.

# Sets the values of the column "# of snippets with warnings" in the correlation analysis dataframe.
# Returns the modified correlation analysis dataframe.
def setNumSnippetsWithWarningsColumn(dfListAnalysisTools, correlationAnalysisDF):
    datasets = correlationAnalysisDF.iloc[:, 1]  # A list of all datasets being used

    # A count of the number of snippets that contain warnings for each dataset
    countSnippetsPerDataset = sortUniqueSnippetsByDataset(datasets, getSnippetsWithWarnings(dfListAnalysisTools))

    for i in range(len(correlationAnalysisDF.index)):
        dataset = str(correlationAnalysisDF.iloc[i, 1])

        if dataset in countSnippetsPerDataset:
            correlationAnalysisDF.iloc[i, 5] = countSnippetsPerDataset[dataset]

    return correlationAnalysisDF

# Sets the values of the column "# of warnings" in the correlation analysis dataframe.
# Returns the modified correlation analysis dataframe.
def setNumWarningsColumn(warningsPerDataset, correlationAnalysisDF):
    for i in range(len(correlationAnalysisDF.index)):
        correlationAnalysisDF.iloc[i, 6] = warningsPerDataset[str(correlationAnalysisDF.iloc[i, 1])]

    return correlationAnalysisDF

# Sets the values of the column "# of datapoints of correlation" in the correlation analysis dataframe.
# Returns the modified correlation analysis dataframe.
def setNumDatapointsForCorrelationColumn(dfDictCorrelationDatapoints, correlationAnalysisDF):
    # Loop through each set of datapoints, there is one for each study/metric
    for key, df in dfDictCorrelationDatapoints.items():
        numDataPoints = len(df)
        i = correlationAnalysisDF[['metric','dataset_id']][(correlationAnalysisDF['metric'] == key[0]) & (correlationAnalysisDF['dataset_id'] == key[1])].index.to_list()

        if len(i) != 1:
            raise Exception("There are multiple rows with the same metric and dataset_id!")

        correlationAnalysisDF.iloc[i[0], 7] = numDataPoints

    return correlationAnalysisDF

# Sets the values of the columns "Kendall"s Tau" and "Kendall"s p-Value" in the correlation analysis dataframe.
# Returns the modified correlation analysis dataframe.
def setKendallTauColumns(kendallTauVals, correlationAnalysisDF):
    for key, val in kendallTauVals.items():
        i = correlationAnalysisDF[['metric','dataset_id']][(correlationAnalysisDF['metric'] == key[0]) & (correlationAnalysisDF['dataset_id'] == key[1])].index.to_list()

        if len(i) != 1:
            raise Exception("There are multiple rows with the same metric and dataset_id!")

        correlationAnalysisDF.iloc[i[0], 8] = val[0]
        correlationAnalysisDF.iloc[i[0], 9] = val[1]

    return correlationAnalysisDF

# Sets the values of the columns "Spearman"s Rho" and "Spearman"s p-Value" in the correlation analysis dataframe.
# Returns the modified correlation analysis dataframe.
def setSpearmanRhoColumns(spearmanRhoVals, correlationAnalysisDF):
    for key, val in spearmanRhoVals.items():
        i = correlationAnalysisDF[['metric','dataset_id']][(correlationAnalysisDF['metric'] == key[0]) & (correlationAnalysisDF['dataset_id'] == key[1])].index.to_list()

        if len(i) != 1:
            raise Exception("There are multiple rows with the same metric and dataset_id!")

        correlationAnalysisDF.iloc[i[0], 10] = val[0]
        correlationAnalysisDF.iloc[i[0], 11] = val[1]

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

    # Datapoint keys have the format: (metric, dataset_id)
    dfDictCorrelationDatapoints = {}

    # TODO Add more datasets here ...

    # Compile datapoints for the COG Dataset 1 Study
    cogDataset1Datapoints = setCogDataset1Datapoints(warningsPerSnippetPerDataset["1"], copy.deepcopy(data))
    dfDictCorrelationDatapoints[("correct_output_rating", 1)] = cogDataset1Datapoints[0]
    dfDictCorrelationDatapoints[("output_difficulty", 1)] = cogDataset1Datapoints[1]
    dfDictCorrelationDatapoints[("time_to_give_output", 1)] = cogDataset1Datapoints[2]

    # Compile datapoints for the COG Dataset 2 Study
    cogDataset2Datapoints = setCogDataset2Datapoints(warningsPerSnippetPerDataset["2"], copy.deepcopy(data))
    dfDictCorrelationDatapoints[("brain_deact_31ant", 2)] = cogDataset2Datapoints[0]
    dfDictCorrelationDatapoints[("brain_deact_31post", 2)] = cogDataset2Datapoints[1]
    dfDictCorrelationDatapoints[("brain_deact_32", 2)] = cogDataset2Datapoints[2]
    dfDictCorrelationDatapoints[("time_to_understand", 2)] = cogDataset2Datapoints[3]

    # Compile datapoints for the COG Dataset 3 Study
    dfDictCorrelationDatapoints[("readability_level", 3)] = setCogDataset3Datapoints(warningsPerSnippetPerDataset["3"], copy.deepcopy(data))

    # Compile datapoints for the COG Dataset 6 Study
    cogDataset6Datapoints = setCogDataset6Datapoints(warningsPerSnippetPerDataset["6"], copy.deepcopy(data))
    dfDictCorrelationDatapoints[("correct_verif_questions", 6)] = cogDataset6Datapoints[0]
    dfDictCorrelationDatapoints[("binary_understandability", 6)] = cogDataset6Datapoints[1]
    dfDictCorrelationDatapoints[("time_to_understand", 6)] = cogDataset6Datapoints[2]

    # Compile datapoints for the COG Dataset 9 Study
    cogDataset9Datapoints = setCogDataset9Datapoints(warningsPerSnippetPerDataset, copy.deepcopy(data))
    dfDictCorrelationDatapoints[("gap_accuracy", "9_gc")] = cogDataset9Datapoints["9_gc"][0]
    dfDictCorrelationDatapoints[("readability_level_before", "9_gc")] = cogDataset9Datapoints["9_gc"][1]
    dfDictCorrelationDatapoints[("readability_level_ba", "9_gc")] = cogDataset9Datapoints["9_gc"][2]
    dfDictCorrelationDatapoints[("time_to_read_complete", "9_gc")] = cogDataset9Datapoints["9_gc"][3]

    dfDictCorrelationDatapoints[("gap_accuracy", "9_bc")] = cogDataset9Datapoints["9_bc"][0]
    dfDictCorrelationDatapoints[("readability_level_before", "9_bc")] = cogDataset9Datapoints["9_bc"][1]
    dfDictCorrelationDatapoints[("readability_level_ba", "9_bc")] = cogDataset9Datapoints["9_bc"][2]
    dfDictCorrelationDatapoints[("time_to_read_complete", "9_bc")] = cogDataset9Datapoints["9_bc"][3]

    dfDictCorrelationDatapoints[("gap_accuracy", "9_nc")] = cogDataset9Datapoints["9_nc"][0]
    dfDictCorrelationDatapoints[("readability_level_before", "9_nc")] = cogDataset9Datapoints["9_nc"][1]
    dfDictCorrelationDatapoints[("readability_level_ba", "9_nc")] = cogDataset9Datapoints["9_nc"][2]
    dfDictCorrelationDatapoints[("time_to_read_complete", "9_nc")] = cogDataset9Datapoints["9_nc"][3]

    # Compile datapoints for the fMRI Study
    fmriDatapoints = setFMRIStudyDatapoints(warningsPerSnippetPerDataset["f"], data)
    dfDictCorrelationDatapoints[("perc_correct_output", "f")] = fmriDatapoints[0]
    dfDictCorrelationDatapoints[("brain_deact_31", "f")] = fmriDatapoints[1]
    dfDictCorrelationDatapoints[("brain_deact_32", "f")] = fmriDatapoints[2]
    dfDictCorrelationDatapoints[("complexity_level", "f")] = fmriDatapoints[3]
    dfDictCorrelationDatapoints[("time_to_understand", "f")] = fmriDatapoints[4]

    return dfDictCorrelationDatapoints

# TODO: functions for future datasets go here ...

def setCogDataset1Datapoints(warningsPerSnippet, data):
    dataCorrectness = copy.deepcopy(data)
    dataSubjComplexity = copy.deepcopy(data)
    dataTime = copy.deepcopy(data)
    metrics = readCOGDataset1StudyMetrics()

    dataCorrectness["Metric"] = metrics[0]
    dataCorrectness["Warning Count"] = warningsPerSnippet
    dataSubjComplexity["Metric"] = metrics[1]
    dataSubjComplexity["Warning Count"] = warningsPerSnippet
    dataTime["Metric"] = metrics[2]
    dataTime["Warning Count"] = warningsPerSnippet

    return (pd.DataFrame(dataCorrectness), pd.DataFrame(dataSubjComplexity), pd.DataFrame(dataTime))

def setCogDataset2Datapoints(warningsPerSnippet, data):
    dataBA31ant = copy.deepcopy(data)
    dataBA31post = copy.deepcopy(data)
    dataBA32 = copy.deepcopy(data)
    dataTime = copy.deepcopy(data)
    metrics = readCOGDataset2StudyMetrics()

    dataBA31ant["Metric"] = metrics[0]
    dataBA31ant["Warning Count"] = warningsPerSnippet
    dataBA31post["Metric"] = metrics[1]
    dataBA31post["Warning Count"] = warningsPerSnippet
    dataBA32["Metric"] = metrics[2]
    dataBA32["Warning Count"] = warningsPerSnippet
    dataTime["Metric"] = metrics[3]
    dataTime["Warning Count"] = warningsPerSnippet

    return (pd.DataFrame(dataBA31ant), pd.DataFrame(dataBA31post), pd.DataFrame(dataBA32), pd.DataFrame(dataTime))

# Gets a list of complexity metrics and a list of warning counts for each snippet in COG Dataset 3.
# Adds that data to a dictionary that is then converted to a dataframe.
def setCogDataset3Datapoints(warningsPerSnippet, data):
    data["Metric"] = readCOGDataset3StudyMetrics()
    data["Warning Count"] = warningsPerSnippet

    return pd.DataFrame(data)

def setCogDataset6Datapoints(warningsPerSnippet, data):
    dataCorrectness = copy.deepcopy(data)
    dataRating = copy.deepcopy(data)
    dataTime = copy.deepcopy(data)
    metrics = readCOGDataset6StudyMetrics()

    dataCorrectness["Metric"] = metrics[0]
    dataCorrectness["Warning Count"] = warningsPerSnippet
    dataRating["Metric"] = metrics[1]
    dataRating["Warning Count"] = warningsPerSnippet
    dataTime["Metric"] = metrics[2]
    dataTime["Warning Count"] = warningsPerSnippet

    return (pd.DataFrame(dataCorrectness), pd.DataFrame(dataRating), pd.DataFrame(dataTime))

def setCogDataset9Datapoints(warningsPerSnippetPerDataset, data):
    dataCorrectness_GC = copy.deepcopy(data)
    dataRatingBefore_GC = copy.deepcopy(data)
    dataRatingBA_GC = copy.deepcopy(data)
    dataTime_GC = copy.deepcopy(data)
    dataCorrectness_BC = copy.deepcopy(data)
    dataRatingBefore_BC = copy.deepcopy(data)
    dataRatingBA_BC = copy.deepcopy(data)
    dataTime_BC = copy.deepcopy(data)
    dataCorrectness_NC = copy.deepcopy(data)
    dataRatingBefore_NC = copy.deepcopy(data)
    dataRatingBA_NC = copy.deepcopy(data)
    dataTime_NC = copy.deepcopy(data)
    metrics = readCOGDataset9StudyMetrics()

    dataCorrectness_GC["Metric"] = metrics[0]
    dataCorrectness_GC["Warning Count"] = warningsPerSnippetPerDataset["9_gc"]
    dataRatingBefore_GC["Metric"] = metrics[1]
    dataRatingBefore_GC["Warning Count"] = warningsPerSnippetPerDataset["9_gc"]
    dataRatingBA_GC["Metric"] = metrics[2]
    dataRatingBA_GC["Warning Count"] = warningsPerSnippetPerDataset["9_gc"]
    dataTime_GC["Metric"] = metrics[3]
    dataTime_GC["Warning Count"] = warningsPerSnippetPerDataset["9_gc"]
    dataCorrectness_BC["Metric"] = metrics[4]
    dataCorrectness_BC["Warning Count"] = warningsPerSnippetPerDataset["9_bc"]
    dataRatingBefore_BC["Metric"] = metrics[5]
    dataRatingBefore_BC["Warning Count"] = warningsPerSnippetPerDataset["9_bc"]
    dataRatingBA_BC["Metric"] = metrics[6]
    dataRatingBA_BC["Warning Count"] = warningsPerSnippetPerDataset["9_bc"]
    dataTime_BC["Metric"] = metrics[7]
    dataTime_BC["Warning Count"] = warningsPerSnippetPerDataset["9_bc"]
    dataCorrectness_NC["Metric"] = metrics[8]
    dataCorrectness_NC["Warning Count"] = warningsPerSnippetPerDataset["9_nc"]
    dataRatingBefore_NC["Metric"] = metrics[9]
    dataRatingBefore_NC["Warning Count"] = warningsPerSnippetPerDataset["9_nc"]
    dataRatingBA_NC["Metric"] = metrics[10]
    dataRatingBA_NC["Warning Count"] = warningsPerSnippetPerDataset["9_nc"]
    dataTime_NC["Metric"] = metrics[11]
    dataTime_NC["Warning Count"] = warningsPerSnippetPerDataset["9_nc"]

    results = {
        "9_gc": (pd.DataFrame(dataCorrectness_GC), pd.DataFrame(dataRatingBefore_GC), pd.DataFrame(dataRatingBA_GC), pd.DataFrame(dataTime_GC)),
        "9_bc": (pd.DataFrame(dataCorrectness_BC), pd.DataFrame(dataRatingBefore_BC), pd.DataFrame(dataRatingBA_BC), pd.DataFrame(dataTime_BC)),
        "9_nc": (pd.DataFrame(dataCorrectness_NC), pd.DataFrame(dataRatingBefore_NC), pd.DataFrame(dataRatingBA_NC), pd.DataFrame(dataTime_NC))
    }

    return results

def setFMRIStudyDatapoints(warningsPerSnippet, data):
    dataCorrectness = copy.deepcopy(data)
    dataBA31 = copy.deepcopy(data)
    dataBA32 = copy.deepcopy(data)
    dataSubjComplexity = copy.deepcopy(data)
    dataTime = copy.deepcopy(data)
    metrics = readFMRIStudyMetrics()

    dataCorrectness["Metric"] = metrics[0]
    dataCorrectness["Warning Count"] = warningsPerSnippet
    dataBA31["Metric"] = metrics[1]
    dataBA31["Warning Count"] = warningsPerSnippet
    dataBA32["Metric"] = metrics[2]
    dataBA32["Warning Count"] = warningsPerSnippet
    dataSubjComplexity["Metric"] = metrics[3]
    dataSubjComplexity["Warning Count"] = warningsPerSnippet
    dataTime["Metric"] = metrics[4]
    dataTime["Warning Count"] = warningsPerSnippet

    return (pd.DataFrame(dataCorrectness), pd.DataFrame(dataBA31), pd.DataFrame(dataBA32), pd.DataFrame(dataSubjComplexity), pd.DataFrame(dataTime))

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

    return (correctness, subjComplexity, times)

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

    return (BA31ant, BA31post, BA32, times)

# Reads the results of the cog data set 3 study. It contains 121 people who rated 100 snippets on a scale of 1-5.
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

    # Get averages for last snippet
    times.append(sumTNPU / participantsPerSnippetTNPU)
    correctness.append(sumAU / participantsPerSnippet)
    rating.append(sumPBU / participantsPerSnippet)

    if len(times) != 50 and len(correctness) != 50 and len(rating) != 50:
        raise Exception

    return (correctness, rating, times)

# Reads the results of the cog data set 9 study. It contains 104 participants and 30 unique snippets (5 snippets each with varying quality of comments).
# Correlation data is split into 3 categories of 10 snippets each: Good comments, bad comments, and no comments. Then further split into the metrics:
# Time, correctness, and rating.
def readCOGDataset9StudyMetrics():
    times_GC = []
    correctness_GC = []
    ratingBA_GC = []
    ratingBefore_GC = []
    times_BC = []
    correctness_BC = []
    ratingBA_BC = []
    ratingBefore_BC = []
    times_NC = []
    correctness_NC = []
    ratingBA_NC = []
    ratingBefore_NC = []

    df = pd.read_excel("data/cog_dataset_9.xlsx")

    participantsPerSnippet = 0
    lastSnippet = ""
    sumTime = 0
    sumCorrectness = 0
    sumRatingBA = 0
    sumRatingBefore = 0
    for row in df.itertuples():
        if pd.isnull(row[19]):
            break

        if row[19] != lastSnippet and row[0] != 0:
            # Moved onto new snippet. Get averages for previous snippet.
            if "1" in lastSnippet.split(":")[1]:
                times_GC.append(sumTime / (participantsPerSnippet * 2))
                correctness_GC.append(sumCorrectness / participantsPerSnippet)
                ratingBA_GC.append(sumRatingBA / (participantsPerSnippet * 2))
                ratingBefore_GC.append(sumRatingBefore / participantsPerSnippet)
            elif "2" in lastSnippet.split(":")[1]:
                times_BC.append(sumTime / (participantsPerSnippet * 2))
                correctness_BC.append(sumCorrectness / participantsPerSnippet)
                ratingBA_BC.append(sumRatingBA / (participantsPerSnippet * 2))
                ratingBefore_BC.append(sumRatingBefore / participantsPerSnippet)
            elif "3" in lastSnippet.split(":")[1]:
                times_NC.append(sumTime / (participantsPerSnippet * 2))
                correctness_NC.append(sumCorrectness / participantsPerSnippet)
                ratingBA_NC.append(sumRatingBA / (participantsPerSnippet * 2))
                ratingBefore_NC.append(sumRatingBefore / participantsPerSnippet)

            sumTime = 0
            sumCorrectness = 0
            sumRatingBA = 0
            sumRatingBefore = 0
            participantsPerSnippet = 0
        
        # Still on same snippet, on first snippet, or starting new snippet after getting the averages for the previous one.
        participantsPerSnippet += 1
        #25 = Score R1, 62 = Score R2, 64 = Score Difference, 82 = Time Read, 83 = Time Completion, 86 = recall accuracy (acc)
        sumTime += row[82] + row[83]
        sumCorrectness += row[86]
        sumRatingBA += row[25] + row[62]
        sumRatingBefore += row[25]
        lastSnippet = row[19]

    # Get averages for last snippet
    times_NC.append(sumTime / (participantsPerSnippet * 2))
    correctness_NC.append(sumCorrectness / participantsPerSnippet)
    ratingBA_NC.append(sumRatingBA / (participantsPerSnippet * 2))
    ratingBefore_NC.append(sumRatingBefore / participantsPerSnippet)

    if len(times_GC) != 10 and len(correctness_GC) != 10 and len(ratingBA_GC) != 10 and len(ratingBefore_GC) != 10 and len(times_BC) != 10 and len(correctness_BC) != 10 and len(ratingBA_BC) != 10 and len(ratingBefore_BC) != 10 and len(times_NC) != 10 and len(correctness_NC) != 10 and len(ratingBA_NC) != 10 and len(ratingBefore_NC) != 10:
        raise Exception("Problem reading in DS9 metrics.")

    return (correctness_GC, ratingBefore_GC, ratingBA_GC, times_GC, correctness_BC, ratingBefore_BC, ratingBA_BC, times_BC, correctness_NC, ratingBefore_NC, ratingBA_NC, times_NC)

# Reads the results of the fMRI study. It contains 19 people who looked at 16 snippets.
# Correctness (in %), time to solve (in sec.), and a subjective rating were all measured.
# Subjective rating of low, medium, or high.
def readFMRIStudyMetrics():
    correctness = [0] * 16
    times = [0] * 16
    subjComplexity = [0] * 16
    ba31 = [0] * 16
    ba32 = [0] * 16

    dfBehavioral = pd.read_csv("data/fmri_dataset_behavioral.csv")
    dfSubjective = pd.read_csv("data/fmri_dataset_subjective.csv")
    dfPhysiological = pd.read_csv("data/fmri_dataset_physiological.csv")

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

    for i in range(len(dfPhysiological.index)):
        ba31[i] = dfPhysiological.iloc[i, 1]
        ba32[i] = dfPhysiological.iloc[i, 2]

    return (correctness, ba31, ba32, subjComplexity, times)


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
    dfList.append(pd.read_csv("data/openjml_data.csv"))

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
    countSnippetsPerDataset = dict([(str(key), 0) for key in datasets])

    for snippet in uniqueSnippets:
        snippet = snippet.split("--")[0].strip() # Name of snippets in "uniqueSnippets" format example: 1 - 12
                                                #                                             format: Dataset ID - Snippet #
        for key in countSnippetsPerDataset:
            if snippet in key:
                countSnippetsPerDataset[key] += 1

    return countSnippetsPerDataset

# Determines the number of warnings for each snippet, separated by the dataset the snippet is from.
# Creates a dictionary where the keys are the names of data sets. Values are a list where the size is 
# the TOTAL number of snippets in the dataset and values within the list are the number of warnings for a given snippet.
def getNumWarningsPerSnippetPerDataset(dfListAnalysisTools, correlationAnalysisDF):
    # Gets data from the dataframe corresponding to correlation_analysis.xlsx
    datasets = correlationAnalysisDF.iloc[:,1]  # A list of all datasets being used
    numSnippetsJudgedPerDataset = correlationAnalysisDF.iloc[:,4]   # A list of the number of snippets in each dataset

    datasetsUnique = []
    numSnippetsJudgedPerDatasetUnique = []

    for i, dataset in enumerate(datasets):
        if str(dataset) not in datasetsUnique:
            datasetsUnique.append(str(dataset))
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
            snippetDataset = snippetNames[i].split("--")[0].strip()
            snippetNumber = snippetNames[i].split("--")[1].strip()

            warningsPerSnippetPerDataset[snippetDataset][int(snippetNumber) - 1] += numWarnings[i]

        return warningsPerSnippetPerDataset

    # Loop through the analysis tool output dataframes
    for df in dfListAnalysisTools:
        numWarnings = df.sum(axis=1, numeric_only=True).tolist()
        snippetNames = df["Snippet"].to_list()

        if len(snippetNames) != len(numWarnings):
            raise Exception("Number of snippets does not match number of warnings associated with said snippets") 
            
        for i in range(len(snippetNames)):
            snippetDataset = str(snippetNames[i].split("--")[0].strip())
            snippetNumber = snippetNames[i].split("--")[1].strip()
            warningsPerSnippetPerDataset[snippetDataset][int(snippetNumber) - 1] += numWarnings[i]

    return warningsPerSnippetPerDataset

# Determines the number of warnings for each dataset
def getNumWarningsPerDataset(warningsPerSnippetPerDataset):
    return {dataset:sum(warningsPerSnippetPerDataset[dataset]) for dataset in warningsPerSnippetPerDataset}

############################
#   Perform Correlations   #
############################

# Perform Kendall's Tau correlation on each dataset seperatly where datapoints are: x = complexity metric, y = # of warnings
# Return a list of the correlation coefficients for each dataset.
def kendallTau(dfDictCorrelationDatapoints):
    kendallTauVals = {key:None for key in dfDictCorrelationDatapoints}

    # Loop through every datapoint dataframe (corresponding to each dataset).
    for key, df in dfDictCorrelationDatapoints.items():
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]

        corr, pValue = scpy.kendalltau(x, y)

        kendallTauVals[key] = (corr, pValue)

    return kendallTauVals

# Perform Spearman"s Rho correlation on each dataset seperatly where datapoints are: a = complexity metric, b = # of warnings
# Return a list of the correlation coefficients for each dataset.
def spearmanRho(dfDictCorrelationDatapoints):
    spearmanRhoVals = {key:None for key in dfDictCorrelationDatapoints}

    # Loop through every datapoint dataframe (corresponding to each dataset).
    for key, df in dfDictCorrelationDatapoints.items():
        a = df.iloc[:, 0]
        b = df.iloc[:, 1]

        corr, pValue = scpy.spearmanr(a, b)

        spearmanRhoVals[key] = (corr, pValue)

    return spearmanRhoVals

##############################################
#   Interact with raw_correlation_data.csv   #
##############################################

def writeRawCorrelationData(allCorrelationData):
    data = []

    for tool in allCorrelationData:
        for correlationDatapoints, df in tool[1].items():
            df = df.reset_index()  # make sure indexes pair with number of rows
            for index, row in df.iterrows():
                data.append({
                    "dataset": correlationDatapoints[1],
                    "snippet": index + 1,
                    "metric": correlationDatapoints[0],
                    "metric_value": row[1],
                    "#_of_warnings": row[2],
                    "tool": tool[0]
                })

    raw_correlation_data_df = pd.DataFrame(data)
    raw_correlation_data_df.to_csv("data/raw_correlation_data.csv", index=False)

###########################
#   Program Starts Here   #
###########################

if __name__ == "__main__":
    # STEP 1 is in parser.py

    # STEP 2:
    # Read in all analysis tool sheets produced by parser.py
    dfListAnalysisTools = readAnalysisToolOutput()
    correlationAnalysisDFAllTools = readCorrelationAnalysis(sheetName="all_tools")
    # TODO: Add more analysis tools here and after each step below ...
    correlationAnalysisDFCheckerFramework = readCorrelationAnalysis(sheetName="checker_framework")
    correlationAnalysisDFTypestateChecker = readCorrelationAnalysis(sheetName="typestate_checker")
    correlationAnalysisDFInfer = readCorrelationAnalysis(sheetName="infer")
    correlationAnalysisDFOpenJML = readCorrelationAnalysis(sheetName="openjml")

    # STEP 3:
    # Determine the number of snippets that contain warnings within each dataset.
    correlationAnalysisDFAllTools = setNumSnippetsWithWarningsColumn(dfListAnalysisTools, correlationAnalysisDFAllTools)
    correlationAnalysisDFCheckerFramework = setNumSnippetsWithWarningsColumn(dfListAnalysisTools[0], correlationAnalysisDFCheckerFramework)
    correlationAnalysisDFTypestateChecker = setNumSnippetsWithWarningsColumn(dfListAnalysisTools[1], correlationAnalysisDFTypestateChecker)
    correlationAnalysisDFInfer = setNumSnippetsWithWarningsColumn(dfListAnalysisTools[2], correlationAnalysisDFInfer)
    correlationAnalysisDFOpenJML = setNumSnippetsWithWarningsColumn(dfListAnalysisTools[3], correlationAnalysisDFOpenJML)

    # STEP 4:
    # Determine the number of warnings per snippet per dataset
    warningsPerSnippetPerDatasetAllTools = getNumWarningsPerSnippetPerDataset(dfListAnalysisTools, correlationAnalysisDFAllTools)
    warningsPerSnippetPerDatasetCheckerFramework = getNumWarningsPerSnippetPerDataset(dfListAnalysisTools[0], correlationAnalysisDFCheckerFramework)
    warningsPerSnippetPerDatasetTypestateChecker = getNumWarningsPerSnippetPerDataset(dfListAnalysisTools[1], correlationAnalysisDFTypestateChecker)
    warningsPerSnippetPerDatasetInfer = getNumWarningsPerSnippetPerDataset(dfListAnalysisTools[2], correlationAnalysisDFInfer)
    warningsPerSnippetPerDatasetOpenJML = getNumWarningsPerSnippetPerDataset(dfListAnalysisTools[3], correlationAnalysisDFOpenJML)

    # STEP 5:
    # Determine the number of warnings per dataset
    correlationAnalysisDFAllTools = setNumWarningsColumn(getNumWarningsPerDataset(warningsPerSnippetPerDatasetAllTools), correlationAnalysisDFAllTools)
    correlationAnalysisDFCheckerFramework = setNumWarningsColumn(getNumWarningsPerDataset(warningsPerSnippetPerDatasetCheckerFramework), correlationAnalysisDFCheckerFramework)
    correlationAnalysisDFTypestateChecker = setNumWarningsColumn(getNumWarningsPerDataset(warningsPerSnippetPerDatasetTypestateChecker), correlationAnalysisDFTypestateChecker)
    correlationAnalysisDFInfer = setNumWarningsColumn(getNumWarningsPerDataset(warningsPerSnippetPerDatasetInfer), correlationAnalysisDFInfer)
    correlationAnalysisDFOpenJML = setNumWarningsColumn(getNumWarningsPerDataset(warningsPerSnippetPerDatasetOpenJML), correlationAnalysisDFOpenJML)

    # STEP 6:
    # Compile all datapoints for correlation: x = complexity metric, y = # of warnings
    dfDictCorrelationDatapointsAllTools = setupCorrelationData(warningsPerSnippetPerDatasetAllTools)
    dfDictCorrelationDatapointsCheckerFramework = setupCorrelationData(warningsPerSnippetPerDatasetCheckerFramework)
    dfDictCorrelationDatapointsTypestateChecker = setupCorrelationData(warningsPerSnippetPerDatasetTypestateChecker)
    dfDictCorrelationDatapointsInfer = setupCorrelationData(warningsPerSnippetPerDatasetInfer)
    dfDictCorrelationDatapointsOpenJML = setupCorrelationData(warningsPerSnippetPerDatasetOpenJML)
    
    # Update correlation analyis data frame 
    correlationAnalysisDFAllTools = setNumDatapointsForCorrelationColumn(dfDictCorrelationDatapointsAllTools, correlationAnalysisDFAllTools)
    correlationAnalysisDFCheckerFramework = setNumDatapointsForCorrelationColumn(dfDictCorrelationDatapointsCheckerFramework, correlationAnalysisDFCheckerFramework)
    correlationAnalysisDFTypestateChecker = setNumDatapointsForCorrelationColumn(dfDictCorrelationDatapointsTypestateChecker, correlationAnalysisDFTypestateChecker)
    correlationAnalysisDFInfer = setNumDatapointsForCorrelationColumn(dfDictCorrelationDatapointsInfer, correlationAnalysisDFInfer)
    correlationAnalysisDFOpenJML = setNumDatapointsForCorrelationColumn(dfDictCorrelationDatapointsOpenJML, correlationAnalysisDFOpenJML)

    # Update raw_correlation_data.csv
    writeRawCorrelationData([("all_tools", dfDictCorrelationDatapointsAllTools), ("checker_framework", dfDictCorrelationDatapointsCheckerFramework), ("typestate_checker", dfDictCorrelationDatapointsTypestateChecker), ("infer", dfDictCorrelationDatapointsInfer), ("openjml", dfDictCorrelationDatapointsOpenJML)])

    # STEP 7:
    # Perform Correlations
    kendallTauValsAllTools = kendallTau(dfDictCorrelationDatapointsAllTools)
    correlationAnalysisDFAllTools = setKendallTauColumns(kendallTauValsAllTools, correlationAnalysisDFAllTools)
    kendallTauValsCheckerFramework = kendallTau(dfDictCorrelationDatapointsCheckerFramework)
    correlationAnalysisDFCheckerFramework = setKendallTauColumns(kendallTauValsCheckerFramework, correlationAnalysisDFCheckerFramework)
    kendallTauValsTypestateChecker = kendallTau(dfDictCorrelationDatapointsTypestateChecker)
    correlationAnalysisDFTypestateChecker = setKendallTauColumns(kendallTauValsTypestateChecker, correlationAnalysisDFTypestateChecker)
    kendallTauValsInfer = kendallTau(dfDictCorrelationDatapointsInfer)
    correlationAnalysisDFInfer = setKendallTauColumns(kendallTauValsInfer, correlationAnalysisDFInfer)
    kendallTauValsOpenJML = kendallTau(dfDictCorrelationDatapointsOpenJML)
    correlationAnalysisDFOpenJML = setKendallTauColumns(kendallTauValsOpenJML, correlationAnalysisDFOpenJML)

    spearmanRhoValsAllTools = spearmanRho(dfDictCorrelationDatapointsAllTools)
    correlationAnalysisDFAllTools = setSpearmanRhoColumns(spearmanRhoValsAllTools, correlationAnalysisDFAllTools)
    spearmanRhoValsCheckerFramework = spearmanRho(dfDictCorrelationDatapointsCheckerFramework)
    correlationAnalysisDFCheckerFramework = setSpearmanRhoColumns(spearmanRhoValsCheckerFramework, correlationAnalysisDFCheckerFramework)
    spearmanRhoValsTypestateChecker = spearmanRho(dfDictCorrelationDatapointsTypestateChecker)
    correlationAnalysisDFTypestateChecker = setSpearmanRhoColumns(spearmanRhoValsTypestateChecker, correlationAnalysisDFTypestateChecker)
    spearmanRhoValsInfer = spearmanRho(dfDictCorrelationDatapointsInfer)
    correlationAnalysisDFInfer = setSpearmanRhoColumns(spearmanRhoValsInfer, correlationAnalysisDFInfer)
    spearmanRhoValsOpenJML = spearmanRho(dfDictCorrelationDatapointsOpenJML)
    correlationAnalysisDFOpenJML = setSpearmanRhoColumns(spearmanRhoValsOpenJML, correlationAnalysisDFOpenJML)

    # Update correlation_analysis.xlsx
    allCorrelationAnalysisDFS = [("all_tools", correlationAnalysisDFAllTools), ("checker_framework", correlationAnalysisDFCheckerFramework), ("typestate_checker", correlationAnalysisDFTypestateChecker), ("infer", correlationAnalysisDFInfer), ("openjml", correlationAnalysisDFOpenJML)]
    writeCorrelationAnalysis(allCorrelationAnalysisDFS)
