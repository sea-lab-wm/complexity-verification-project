## This scripts writes the scalabrino features to the output csv file
import csv
import os
import pandas as pd

csv_data_dict = {
    "dataset_id":"",
    "snippet_id": "", 
    "method_name": "",
    "file": "",

    "ITID (avg)": 0.0,
    "ITID (min)": 0.0,
    "NMI (avg)": 0.0,
    "NMI (max)": 0.0,
    "NMI (min)": 0.0,
    "CIC (avg)": 0.0,
    "CIC (max)": 0.0,
    "CICsyn (avg)": 0.0,
    "CICsyn (max)": 0.0,

    "New Expression complexity AVG": 0.0,
    "New Expression complexity MAX": 0.0,
    "New Expression complexity MIN": 0.0,
    "New Method chains AVG": 0.0,
    "New Method chains MAX": 0.0,
    "New Method chains MIN": 0.0,

    "CR": 0.0,
    "NM (avg)": 0.0,
    "NM (max)": 0.0,
    "NOC (standard)": 0.0,
    "NOC (normalized)": 0.0,
    "TC (avg)": 0.0,
    "TC (min)": 0.0,
    "TC (max)": 0.0,
    "#assignments (avg)": 0.0,

    "#blank lines (avg)": 0.0,
    "#commas (avg)": 0.0,
    "#comments (avg)": 0.0,
    "#comparisons (avg)": 0.0,
    "Identifiers length (avg)": 0.0,
    "#conditionals (avg)": 0.0,
    "Indentation length (avg)": 0.0,
    "#keywords (avg)": 0.0,
    "Line length (avg)": 0.0,
    "#loops (avg)": 0.0,
    "#identifiers (avg)": 0.0,
    "#numbers (avg)": 0.0,
    "#operators (avg)": 0.0,
    "#parenthesis (avg)": 0.0,
    "#periods (avg)": 0.0,
    "#spaces (avg)": 0.0,

    "Identifiers length (max)": 0.0,
    "Indentation length (max)": 0.0,
    "#keywords (max)": 0.0,
    "Line length (max)": 0.0,
    "#identifiers (max)": 0.0,
    "#numbers (max)": 0.0,
    "#characters (max)": 0.0,
    "#words (max)": 0.0,
    "Entropy": 0.0,
    "Volume": 0.0,
    "LOC": 0.0,
    
    "#assignments (dft)": 0.0,
    "#commas (dft)": 0.0,
    "#comments (dft)": 0.0,
    "#comparisons (dft)": 0.0,
    "#conditionals (dft)": 0.0,
    "Indentation length (dft)": 0.0,
    "#keywords (dft)": 0.0,
    "Line length (dft)": 0.0,
    "#loops (dft)": 0.0,
    "#identifiers (dft)": 0.0,
    "#numbers (dft)": 0.0,
    "#operators (dft)": 0.0,
    "#parenthesis (dft)": 0.0,
    "#periods (dft)": 0.0,
    "#spaces (dft)": 0.0,
    
    "Comments (Visual X)": 0.0,
    "Comments (Visual Y)": 0.0,
    "Identifiers (Visual X)": 0.0,
    "Identifiers (Visual Y)": 0.0,
    "Keywords (Visual X)": 0.0,
    "Keywords (Visual Y)": 0.0,
    "Numbers (Visual X)": 0.0,
    "Numbers (Visual Y)": 0.0,
    "Strings (Visual X)": 0.0,
    "Strings (Visual Y)": 0.0,
    "Literals (Visual X)": 0.0,
    "Literals (Visual Y)": 0.0,
    "Operators (Visual X)": 0.0,
    "Operators (Visual Y)": 0.0,
    "Comments (area)": 0.0,
    "Identifiers (area)": 0.0,

    "Keywords (area)": 0.0,
    "Numbers (area)": 0.0,
    "Strings (area)": 0.0,
    "Literals (area)": 0.0,
    "Operators (area)": 0.0,
    "Identifiers/comments (area)": 0.0,
    "Keywords/comments (area)": 0.0,
    "Numbers/comments (area)": 0.0,
    "Strings/comments (area)": 0.0,
    "Literals/comments (area)": 0.0,
    "Operators/comments (area)": 0.0,
    "Keywords/identifiers (area)": 0.0,
    "Numbers/identifiers (area)": 0.0,
    "Strings/identifiers (area)": 0.0,

    "Literals/identifiers (area)": 0.0,
    "Operators/literals (area)": 0.0,
    "Numbers/keywords (area)": 0.0,
    "Strings/keywords (area)": 0.0,
    "Literals/keywords (area)": 0.0,
    "Operators/keywords (area)": 0.0,
    "Strings/numbers (area)": 0.0,
    "Literals/numbers (area)": 0.0,
    "Operators/numbers (area)": 0.0,
    "Literals/strings (area)": 0.0,
    "Operators/strings (area)": 0.0,
    "Operators/literals (area).1": 0.0,
    "#aligned blocks": 0.0,
    "Extent of aligned blocks": 0.0
}

def dict_data_generator(dataset_id, snippet_id, method_name, file, feature_dict):
    csv_data_dict["dataset_id"] = dataset_id
    csv_data_dict["snippet_id"] = snippet_id
    csv_data_dict["method_name"] = method_name
    csv_data_dict["file"] = file

    csv_data_dict["ITID (avg)"] = feature_dict["New Identifiers words AVG"]
    csv_data_dict["ITID (min)"]= feature_dict["New Identifiers words MIN"]
    csv_data_dict["NMI (avg)"]= feature_dict["New Abstractness words AVG"]
    csv_data_dict["NMI (max)"]= feature_dict["New Abstractness words MAX"]
    csv_data_dict["NMI (min)"]= feature_dict["New Abstractness words MIN"]
    csv_data_dict["CIC (avg)"]= feature_dict["New Commented words AVG"]
    csv_data_dict["CIC (max)"]= feature_dict["New Commented words MAX"]
    csv_data_dict["CICsyn (avg)"]= feature_dict["New Synonym commented words AVG"]
    csv_data_dict["CICsyn (max)"]= feature_dict["New Synonym commented words MAX"]
    
    csv_data_dict["New Expression complexity AVG"]= feature_dict["New Expression complexity AVG"]
    csv_data_dict["New Expression complexity MAX"]= feature_dict["New Expression complexity MAX"]
    csv_data_dict["New Expression complexity MIN"]= feature_dict["New Expression complexity MIN"]
    csv_data_dict["New Method chains AVG"]= feature_dict["New Method chains AVG"]
    csv_data_dict["New Method chains MAX"]= feature_dict["New Method chains MAX"]
    csv_data_dict["New Method chains MIN"]= feature_dict["New Method chains MIN"]
    
    csv_data_dict["CR"]= feature_dict["New Comments readability"]
    csv_data_dict["NM (avg)"]= feature_dict["New Number of senses AVG"]
    csv_data_dict["NM (max)"]= feature_dict["New Number of senses MAX"]
    csv_data_dict["NOC (standard)"]= feature_dict["New Semantic Text Coherence Standard"]
    csv_data_dict["NOC (normalized)"] = feature_dict["New Semantic Text Coherence Normalized"]
    csv_data_dict["TC (avg)"] = feature_dict["New Text Coherence AVG"]
    csv_data_dict["TC (min)"] = feature_dict["New Text Coherence MIN"]
    csv_data_dict["TC (max)"] = feature_dict["New Text Coherence MAX"]
    csv_data_dict["#assignments (avg)"] = feature_dict["BW Avg Assignment"]
    
    csv_data_dict["#blank lines (avg)"] = feature_dict["BW Avg blank lines"]
    csv_data_dict["#commas (avg)"] = feature_dict["BW Avg commas"]
    csv_data_dict["#comments (avg)"] = feature_dict["BW Avg comments"]
    csv_data_dict["#comparisons (avg)"] = feature_dict["BW Avg comparisons"]
    csv_data_dict["Identifiers length (avg)"] = feature_dict["BW Avg Identifiers Length"]
    csv_data_dict["#conditionals (avg)"] = feature_dict["BW Avg conditionals"]
    csv_data_dict["Indentation length (avg)"] = feature_dict["BW Avg indentation length"]
    csv_data_dict["#keywords (avg)"] = feature_dict["BW Avg keywords"]
    csv_data_dict["Line length (avg)"] = feature_dict["BW Avg line length"]
    csv_data_dict["#loops (avg)"] = feature_dict["BW Avg loops"]
    csv_data_dict["#identifiers (avg)"] = feature_dict["BW Avg number of identifiers"]
    csv_data_dict["#numbers (avg)"] = feature_dict["BW Avg numbers"]
    csv_data_dict["#operators (avg)"] = feature_dict["BW Avg operators"]
    csv_data_dict["#parenthesis (avg)"] = feature_dict["BW Avg parenthesis"]
    csv_data_dict["#periods (avg)"] = feature_dict["BW Avg periods"]
    csv_data_dict["#spaces (avg)"] = feature_dict["BW Avg spaces"]
    
    csv_data_dict["Identifiers length (max)"] = feature_dict["BW Max Identifiers Length"]
    csv_data_dict["Indentation length (max)"] = feature_dict["BW Max indentation"]
    csv_data_dict["#keywords (max)"] = feature_dict["BW Max keywords"]
    csv_data_dict["Line length (max)"] = feature_dict["BW Max line length"]
    csv_data_dict["#identifiers (max)"] = feature_dict["BW Max number of identifiers"]
    csv_data_dict["#numbers (max)"] = feature_dict["BW Max numbers"]
    csv_data_dict["#characters (max)"] = feature_dict["BW Max char"]
    csv_data_dict["#words (max)"] = feature_dict["BW Max words"]
    csv_data_dict["Entropy"] = feature_dict["Posnett entropy"]
    csv_data_dict["Volume"] = feature_dict["Posnett volume"]
    csv_data_dict["LOC"] = feature_dict["Posnett lines"]
    
    csv_data_dict["#assignments (dft)"] = feature_dict["Dorn DFT Assignments"]
    csv_data_dict["#commas (dft)"] = feature_dict["Dorn DFT Commas"]
    csv_data_dict["#comments (dft)"] = feature_dict["Dorn DFT Comments"]
    csv_data_dict["#comparisons (dft)"] = feature_dict["Dorn DFT Comparisons"]
    csv_data_dict["#conditionals (dft)"] = feature_dict["Dorn DFT Conditionals"]
    csv_data_dict["Indentation length (dft)"] = feature_dict["Dorn DFT Indentations"]
    csv_data_dict["#keywords (dft)"] = feature_dict["Dorn DFT Keywords"]
    csv_data_dict["Line length (dft)"] = feature_dict["Dorn DFT LineLengths"]
    csv_data_dict["#loops (dft)"] = feature_dict["Dorn DFT Loops"]
    csv_data_dict["#identifiers (dft)"] = feature_dict["Dorn DFT Identifiers"]
    csv_data_dict["#numbers (dft)"] = feature_dict["Dorn DFT Numbers"]
    csv_data_dict["#operators (dft)"] = feature_dict["Dorn DFT Operators"]
    csv_data_dict["#parenthesis (dft)"] = feature_dict["Dorn DFT Parenthesis"]
    csv_data_dict["#periods (dft)"] = feature_dict["Dorn DFT Periods"]
    csv_data_dict["#spaces (dft)"] = feature_dict["Dorn DFT Spaces"]
    
    csv_data_dict["Comments (Visual X)"] = feature_dict["Dorn Visual X Comments"]
    csv_data_dict["Comments (Visual Y)"] = feature_dict["Dorn Visual Y Comments"]
    csv_data_dict["Identifiers (Visual X)"] = feature_dict["Dorn Visual X Identifiers"]
    csv_data_dict["Identifiers (Visual Y)"] = feature_dict["Dorn Visual Y Identifiers"]
    csv_data_dict["Keywords (Visual X)"] = feature_dict["Dorn Visual X Keywords"]
    csv_data_dict["Keywords (Visual Y)"] = feature_dict["Dorn Visual Y Keywords"]
    csv_data_dict["Numbers (Visual X)"] = feature_dict["Dorn Visual X Numbers"]
    csv_data_dict["Numbers (Visual Y)"] = feature_dict["Dorn Visual Y Numbers"]
    csv_data_dict["Strings (Visual X)"] = feature_dict["Dorn Visual X Strings"]
    csv_data_dict["Strings (Visual Y)"] = feature_dict["Dorn Visual Y Strings"]
    csv_data_dict["Literals (Visual X)"] = feature_dict["Dorn Visual X Literals"]
    csv_data_dict["Literals (Visual Y)"] = feature_dict["Dorn Visual Y Literals"]
    csv_data_dict["Operators (Visual X)"] = feature_dict["Dorn Visual X Operators"]
    csv_data_dict["Operators (Visual Y)"] = feature_dict["Dorn Visual Y Operators"]
    csv_data_dict["Comments (area)"] = feature_dict["Dorn Areas Comments"]
    csv_data_dict["Identifiers (area)"] = feature_dict["Dorn Areas Identifiers"]
    
    csv_data_dict["Keywords (area)"] = feature_dict["Dorn Areas Keywords"]
    csv_data_dict["Numbers (area)"] = feature_dict["Dorn Areas Numbers"]
    csv_data_dict["Strings (area)"] = feature_dict["Dorn Areas Strings"]
    csv_data_dict["Literals (area)"] = feature_dict["Dorn Areas Literals"]
    csv_data_dict["Operators (area)"] = feature_dict["Dorn Areas Operators"]
    csv_data_dict["Identifiers/comments (area)"] = feature_dict["Dorn Areas Identifiers/Comments"]
    csv_data_dict["Keywords/comments (area)"] = feature_dict["Dorn Areas Keywords/Comments"]
    csv_data_dict["Numbers/comments (area)"] = feature_dict["Dorn Areas Numbers/Comments"]
    csv_data_dict["Strings/comments (area)"] = feature_dict["Dorn Areas Strings/Comments"]
    csv_data_dict["Literals/comments (area)"] = feature_dict["Dorn Areas Literals/Comments"]
    csv_data_dict["Operators/comments (area)"] = feature_dict["Dorn Areas Operators/Comments"]
    csv_data_dict["Keywords/identifiers (area)"] = feature_dict["Dorn Areas Keywords/Identifiers"]
    csv_data_dict["Numbers/identifiers (area)"] = feature_dict["Dorn Areas Numbers/Identifiers"]
    csv_data_dict["Strings/identifiers (area)"] = feature_dict["Dorn Areas Strings/Identifiers"]
    csv_data_dict["Literals/identifiers (area)"] = feature_dict["Dorn Areas Literals/Identifiers"]
    csv_data_dict["Operators/literals (area)"] = feature_dict["Dorn Areas Operators/Identifiers"]
    csv_data_dict["Numbers/keywords (area)"] = feature_dict["Dorn Areas Numbers/Keywords"]
    csv_data_dict["Strings/keywords (area)"] = feature_dict["Dorn Areas Strings/Keywords"]
    csv_data_dict["Literals/keywords (area)"] = feature_dict["Dorn Areas Literals/Keywords"]
    csv_data_dict["Operators/keywords (area)"] = feature_dict["Dorn Areas Operators/Keywords"]
    csv_data_dict["Strings/numbers (area)"] = feature_dict["Dorn Areas Strings/Numbers"]
    csv_data_dict["Literals/numbers (area)"] = feature_dict[ "Dorn Areas Literals/Numbers"]
    csv_data_dict["Operators/numbers (area)"] = feature_dict["Dorn Areas Operators/Numbers"]
    csv_data_dict["Literals/strings (area)"] = feature_dict["Dorn Areas Literals/Strings"]
    csv_data_dict["Operators/strings (area)"] = feature_dict["Dorn Areas Operators/Strings"]
    csv_data_dict["Operators/literals (area).1"] = feature_dict["Dorn Areas Operators/Literals"]
    csv_data_dict["#aligned blocks"] = feature_dict["Dorn align blocks"]
    csv_data_dict["Extent of aligned blocks"] = feature_dict["Dorn align extent"]

    
    return csv_data_dict

csv_data_dict_readability = {
    "dataset_id":"",
    "snippet_id": "", 
    "method_name": "",
    "file": "",
    "Readability": 0.0
}

def dict_to_csv(output_file_path, dict_data):
    with open(output_file_path, "a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dict_data.keys())
        writer.writerow(dict_data)

def extract_features(file_path):

    feature_dict = {}

    ## read file line by line
    with open(file_path, "r") as file:
        lines = file.readlines()

        for i in range(1, len(lines)):
            line = lines[i].strip()
            feature_name = line.split(":")[0].strip()
            feature_value = line.split(":")[1].strip()
            feature_dict[feature_name] = feature_value


    return feature_dict    

def main():

    output_file_path = "/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model/scalabrino_features.csv"
    
    ## write header
    with open(output_file_path , "w+") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_data_dict.keys())
        writer.writeheader()

    
    files = os.listdir("/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model/src/main/resources/Scalabrino-Replication-Package/output")
    files = [file for file in files if file.endswith(".txt")]

    for file in files:
        file_path = "/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model/src/main/resources/Scalabrino-Replication-Package/output/" + file
        
        file_name = file.split("_features.txt")[0]
        dataset_id = file_name.split("_")[1]
        snippet_id = file_name.split("_")[3]

        if snippet_id.__contains__("$"):
            snippet_id = snippet_id.replace("$", "-")
        method_name = file_name.split("_")[4].split(".")[0]
        if dataset_id.__contains__("$"):
            dataset_id = dataset_id.replace("$", "_")
        if method_name.__contains__("$"):
            method_name = method_name.replace("$", "-")

        ## extract features
        feature_dict = extract_features(file_path)

        
        dict_data = dict_data_generator(dataset_id, snippet_id, method_name, file_name, feature_dict=feature_dict)
        dict_to_csv(output_file_path, dict_data)
    

    ### Readability features ###
    output_file_path = "/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model/scalabrino_readability_features.csv"
    
    ## write header
    with open(output_file_path , "w+") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_data_dict_readability.keys())
        writer.writeheader()               
            

    with open("complexity-verification-project/ml_model/src/main/resources/Scalabrino-Replication-Package/readability_features.txt", "r") as file:
        lines = file.readlines()
        for i in range(2, len(lines)):
            line = lines[i].strip()
            file_name = line.split(' ')[0].strip()
            readability_value = line.split(' ')[1].strip()

            dataset_id = file_name.split("/")[1].split("_")[1]
            if dataset_id.__contains__("$"):
                dataset_id = dataset_id.replace("$", "_")
            snippet_id = file_name.split("/")[2].split("_")[3]
            if snippet_id.__contains__("$"):
                snippet_id = snippet_id.replace("$", "-")
            method_name = file_name.split("/")[2].split("_")[4].split(".")[0]
            if method_name.__contains__("$"):
                method_name = method_name.replace("$", "-")

            file = file_name.split("/")[2]

            ## dict data generation
            csv_data_dict_readability["dataset_id"] = dataset_id
            csv_data_dict_readability["snippet_id"] = snippet_id
            csv_data_dict_readability["method_name"] = method_name
            csv_data_dict_readability["file"] = file
            csv_data_dict_readability["Readability"] = readability_value

            dict_to_csv(output_file_path, csv_data_dict_readability)

    
    ## merge two csv files based on dataset_id, snippet_id, method_name and file
    df_1 = pd.read_csv("/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model/scalabrino_features.csv")
    df_2 = pd.read_csv("/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model/scalabrino_readability_features.csv")
    
    df = pd.merge(df_1, df_2, on=["dataset_id", "snippet_id", "method_name", "file"])

    df.to_csv("/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model/scalabrino_features_complete.csv", index=False)

    ## remove the intermediate files
    os.remove("/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model/scalabrino_features.csv")
    os.remove("/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model/scalabrino_readability_features.csv")
    
if __name__ == "__main__":
    main()            