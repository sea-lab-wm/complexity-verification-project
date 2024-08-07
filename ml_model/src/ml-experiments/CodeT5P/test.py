import pandas as pd
from imblearn.over_sampling import SMOTE

RANDOM_SEED=42

def balance_dataset(csv_file, target):
    ## use imblearn SMOTE to balance the dataset

    full_df = pd.read_csv(csv_file)
    df = full_df[["Numbers/keywords (area)", "AEDQ (min)", "Operators/literals (area)", "NMI (avg)", "#characters (max)", "Strings/keywords (area)", "developer_position", "Operators (Visual X)", "Identifiers length (avg)", "Keywords (Visual X)", "#comments (avg)", "TC (avg)", "Keywords (Visual Y)", "Identifiers length (max)", "ITID (avg)", "IMSQ (avg)", "Comments (area)", "Volume", "#assignments (dft)", "Identifiers (Visual X)", "#assignments (avg)", "#comments (dft)", "#parenthesis (dft)", "Identifiers (area)", "MIDQ (avg)", "Operators (Visual Y)", "#commas (dft)", "Comments (Visual X)", "CICsyn (avg)", "TC (max)", "Line length (avg)", "Operators (area)", "#periods (avg)", "Line length (max)", "Cyclomatic complexity", "#conditionals (avg)", "AEDQ (avg)", "AEDQ (max)", "Indentation length (avg)", "#keywords (avg)", "#parenthesis (avg)", "NMI (max)", "#blank lines (avg)", "Keywords (area)", "CIC (avg)", "#numbers (max)", "#spaces (avg)", "#keywords (max)", "#aligned blocks", "Operators/keywords (area)", "PE gen", "#identifiers (dft)", "#numbers (dft)", "Indentation length (max)", "Numbers/identifiers (area)", "Numbers (Visual X)", "Strings (Visual X)", "Identifiers (Visual Y)", "#operators (avg)", "#commas (avg)", "Line length (dft)", "#operators (dft)", "#parameters", "Keywords/identifiers (area)", "EAP (min)", "Literals (Visual Y)", "LOC", "#nested blocks (avg)", "IMSQ (max)", "Entropy", "EAP (avg)", "CIC (max)", "Strings (area)", "MIDQ (min)", "#keywords (dft)", "Readability", "#identifiers (avg)", "IMSQ (min)", "#conditionals (dft)", "EAP (max)", "PE spec (java)", "#numbers (avg)", "#words (max)", "Comments (Visual Y)", "Strings/identifiers (area)", "#loops (avg)", "Extent of aligned blocks", "#loops (dft)", "#comparisons (avg)", "Literals (area)", "CR", "Numbers (area)", "Numbers (Visual Y)", "#comparisons (dft)", "Literals/keywords (area)", "#identifiers (max)", "Strings (Visual Y)", "MIDQ (max)", "NM (avg)", "CICsyn (max)", "TC (min)", "#spaces (dft)", "#statements", "Indentation length (dft)", "Literals/identifiers (area)", "NM (max)", "#periods (dft)", "Literals (Visual X)", "PBU"]]
    X = df.drop(target, axis=1) ## drop the target column
    y = df[target]

    ros = SMOTE(sampling_strategy='auto', random_state=RANDOM_SEED) ## resample all classes but the majority class
    X_sm, y_sm = ros.fit_resample(X, y)

    df_sm = pd.concat([X_sm, y_sm], axis=1)

    ## save the balanced dataset
    df_sm.to_csv('CodeT5P/data/balanced_' + csv_file.split("/")[-1], index=False)


    df = full_df.drop('file_content', axis=1)
    df = df[["Numbers/keywords (area)", "AEDQ (min)", "Operators/literals (area)", "NMI (avg)", "#characters (max)", "Strings/keywords (area)", "developer_position", "Operators (Visual X)", "Identifiers length (avg)", "Keywords (Visual X)", "#comments (avg)", "TC (avg)", "Keywords (Visual Y)", "Identifiers length (max)", "ITID (avg)", "IMSQ (avg)", "Comments (area)", "Volume", "#assignments (dft)", "Identifiers (Visual X)", "#assignments (avg)", "#comments (dft)", "#parenthesis (dft)", "Identifiers (area)", "MIDQ (avg)", "Operators (Visual Y)", "#commas (dft)", "Comments (Visual X)", "CICsyn (avg)", "TC (max)", "Line length (avg)", "Operators (area)", "#periods (avg)", "Line length (max)", "Cyclomatic complexity", "#conditionals (avg)", "AEDQ (avg)", "AEDQ (max)", "Indentation length (avg)", "#keywords (avg)", "#parenthesis (avg)", "NMI (max)", "#blank lines (avg)", "Keywords (area)", "CIC (avg)", "#numbers (max)", "#spaces (avg)", "#keywords (max)", "#aligned blocks", "Operators/keywords (area)", "PE gen", "#identifiers (dft)", "#numbers (dft)", "Indentation length (max)", "Numbers/identifiers (area)", "Numbers (Visual X)", "Strings (Visual X)", "Identifiers (Visual Y)", "#operators (avg)", "#commas (avg)", "Line length (dft)", "#operators (dft)", "#parameters", "Keywords/identifiers (area)", "EAP (min)", "Literals (Visual Y)", "LOC", "#nested blocks (avg)", "IMSQ (max)", "Entropy", "EAP (avg)", "CIC (max)", "Strings (area)", "MIDQ (min)", "#keywords (dft)", "Readability", "#identifiers (avg)", "IMSQ (min)", "#conditionals (dft)", "EAP (max)", "PE spec (java)", "#numbers (avg)", "#words (max)", "Comments (Visual Y)", "Strings/identifiers (area)", "#loops (avg)", "Extent of aligned blocks", "#loops (dft)", "#comparisons (avg)", "Literals (area)", "CR", "Numbers (area)", "Numbers (Visual Y)", "#comparisons (dft)", "Literals/keywords (area)", "#identifiers (max)", "Strings (Visual Y)", "MIDQ (max)", "NM (avg)", "CICsyn (max)", "TC (min)", "#spaces (dft)", "#statements", "Indentation length (dft)", "Literals/identifiers (area)", "NM (max)", "#periods (dft)", "Literals (Visual X)", "PBU"]]
    df.to_csv('CodeT5P/data/train_with_no_file_content.csv', index=False)

    return 'balanced_train.csv'

ROOT_PATH='/home/nadeeshan/ML-Experiments-2/complexity-verification-project/ml_model/model/NewExperiments/CodeT5P/data/' + 'train.csv'
    
balance_dataset(ROOT_PATH, 'PBU')

## synthetic data from row 357


## create box plot for real data

