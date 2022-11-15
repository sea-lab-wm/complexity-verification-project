import os

# Parses a .java file containing numbered snippets and grabs all the code directly inside a snippet (ignoring all non snippet code)
def isolateSnippets(filePath, start, end):
    keepSnippets = []
    foundStart = False
    count = 0

    with open(filePath) as f:
        for line in f:
            if start in line:
                foundStart = True
                keepSnippets.append([])
                count += 1

            if end in line:
                foundStart = False
                
            if foundStart is True:
                keepSnippets[count - 1].append(line)

    for i, snippet in enumerate(keepSnippets):
        for j, line in enumerate(snippet):
            if start in line or end in line:
                keepSnippets[i].pop(j)

    return keepSnippets

# Isolate the code for all snippets
def isolateAllSnippets():
    allSnippets = {}

    # fMRI Dataset
    fMRIDatasetSnippets = []
    inOrder = [file.split(".")[0] for file in os.listdir("simple-datasets/src/main/java/fMRI_Study_Classes") if ".java" in file]
    inOrder = sorted(inOrder, key=str.lower) # Keeps the order of these snippets consistent across operating systems
    for file in inOrder:
        fMRIDatasetSnippets.append(isolateSnippets(f"simple-datasets/src/main/java/fMRI_Study_Classes/{file}.java", "SNIPPET_STARTS", "**NO_END**"))

    for i, val in enumerate(fMRIDatasetSnippets):
        fMRIDatasetSnippets[i] = val[0]

    fMRIDatasetSnippetNames = {file:fMRIDatasetSnippets[i] for i, file in enumerate(inOrder)}
    allSnippets["dataset_f"] = fMRIDatasetSnippetNames

    # COG Dataset 1
    allSnippets["dataset_1"] = {f"S_{str(i)}":snippet for i, snippet in enumerate(isolateSnippets("simple-datasets/src/main/java/cog_complexity_validation_datasets/One/Tasks.java", "SNIPPET_STARTS", "SNIPPET_END"))}

    #COG Dataset 2
    allSnippets["dataset_2"] = {f"S_{str(i)}":snippet for i, snippet in enumerate(isolateSnippets("simple-datasets/src/main/java/cog_complexity_validation_datasets/One/Tasks.java", "DATASET2START", "DATASET2END"))}

    # COG Dataset 3
    cogDataset3SnippetsTasks_1 = isolateSnippets("simple-datasets/src/main/java/cog_complexity_validation_datasets/Three/Tasks_1.java", "SNIPPET_STARTS", "SNIPPET_END")
    cogDataset3SnippetsTasks_2 = isolateSnippets("simple-datasets/src/main/java/cog_complexity_validation_datasets/Three/Tasks_2.java", "SNIPPET_STARTS", "SNIPPET_END")
    cogDataset3SnippetsTasks_3 = isolateSnippets("simple-datasets/src/main/java/cog_complexity_validation_datasets/Three/Tasks_3.java", "SNIPPET_STARTS", "SNIPPET_END")
    cogDataset3Snippets = cogDataset3SnippetsTasks_1 + cogDataset3SnippetsTasks_2 + cogDataset3SnippetsTasks_3
    allSnippets["dataset_3"] = {f"S_{str(i)}":snippet for i, snippet in enumerate(cogDataset3Snippets)}

    # COG Dataset 6
    # A dictionary of lists. Each inner list contains the line numbers for the snippets in a single .java file. This dataset has snippets split across several files.
    # They are in the order of how they appear in "cog_dataset_6.csv", the file containing the metric data from its prior study.
    cogDataset6SnippetsK9 = isolateSnippets("dataset6/src/main/java/K9.java", "SNIPPET_STARTS", "SNIPPET_END")
    cogDataset6SnippetsPom = isolateSnippets("dataset6/src/main/java/Pom.java", "SNIPPET_STARTS", "SNIPPET_END")
    cogDataset6SnippetsCarReport = isolateSnippets("dataset6/src/main/java/CarReport.java", "SNIPPET_STARTS", "SNIPPET_END")
    cogDataset6SnippetsAntlr4Master = isolateSnippets("dataset6/src/main/java/Antlr4Master.java", "SNIPPET_STARTS", "SNIPPET_END")
    cogDataset6SnippetsPhoenix = isolateSnippets("dataset6/src/main/java/Phoenix.java", "SNIPPET_STARTS", "SNIPPET_END")
    cogDataset6SnippetsHibernateORM = isolateSnippets("dataset6/src/main/java/HibernateORM.java", "SNIPPET_STARTS", "SNIPPET_END")
    cogDataset6SnippetsOpenCMSCore = isolateSnippets("dataset6/src/main/java/OpenCMSCore.java", "SNIPPET_STARTS", "SNIPPET_END")
    cogDataset6SnippetsSpringBatch = isolateSnippets("dataset6/src/main/java/SpringBatch.java", "SNIPPET_STARTS", "SNIPPET_END")
    cogDataset6SnippetsMyExpenses = isolateSnippets("dataset6/src/main/java/MyExpenses.java", "SNIPPET_STARTS", "SNIPPET_END")
    cogDataset6SnippetsCheckEstimator = isolateSnippets("dataset6/src/main/java/weka/estimators/CheckEstimator.java", "SNIPPET_STARTS", "SNIPPET_END")
    cogDataset6SnippetsEstimatorUtils = isolateSnippets("dataset6/src/main/java/weka/estimators/EstimatorUtils.java", "SNIPPET_STARTS", "SNIPPET_END")
    cogDataset6SnippetsClassifierPerformanceEvaluatorCustomizer = isolateSnippets("dataset6/src/main/java/weka/gui/beans/ClassifierPerformanceEvaluatorCustomizer.java", "SNIPPET_STARTS", "SNIPPET_END")
    cogDataset6SnippetsModelPerformanceChart = isolateSnippets("dataset6/src/main/java/weka/gui/beans/ModelPerformanceChart.java", "SNIPPET_STARTS", "SNIPPET_END")
    cogDataset6SnippetsGeneratorPropertyIteratorPanel = isolateSnippets("dataset6/src/main/java/weka/gui/experiment/GeneratorPropertyIteratorPanel.java", "SNIPPET_STARTS", "SNIPPET_END")
    cogDataset6Snippets = cogDataset6SnippetsK9 + cogDataset6SnippetsPom + cogDataset6SnippetsCarReport + cogDataset6SnippetsAntlr4Master + cogDataset6SnippetsPhoenix + cogDataset6SnippetsHibernateORM + cogDataset6SnippetsOpenCMSCore + cogDataset6SnippetsSpringBatch + cogDataset6SnippetsMyExpenses + cogDataset6SnippetsCheckEstimator + cogDataset6SnippetsEstimatorUtils + cogDataset6SnippetsClassifierPerformanceEvaluatorCustomizer + cogDataset6SnippetsModelPerformanceChart + cogDataset6SnippetsGeneratorPropertyIteratorPanel
    allSnippets["dataset_6"] = {f"S_{str(i)}":snippet for i, snippet in enumerate(cogDataset6Snippets)}

    # COG Dataset 9
    allSnippets["dataset_9"] = {f"S_{str(i)}":snippet for i, snippet in enumerate(isolateSnippets("dataset9/src/main/java/CodeSnippets.java", "SNIPPET_STARTS", "SNIPPET_END"))}

    return allSnippets

def writeIsolatedSnippets(allSnippets):
    for key, value in allSnippets.items():
        path = f"loc_per_snippet/{key}/"

        for key2, value2 in value.items():
            #for snippet in value2:
            with open(f"{path}{key2}.java", "w") as file:
                file.writelines(value2)
            

if __name__ == "__main__":
    allSnippets = isolateAllSnippets()

    writeIsolatedSnippets(allSnippets)
    
