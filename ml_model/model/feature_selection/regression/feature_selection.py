'''
This feature selection based on the paper https://ieeexplore.ieee.org/document/8651396

## Feature selection procedure ##
## 1. Perform Kendall's Tau correlation test between each pair of features.
## 2. Pick the pairs that show |Tau| >= 0.7 (highly correlated)
## 3. Remove one of the features which has the highest number of missing values.
## 4. If missing values are equal, remove one of the features randomly.
2. Remove Keywords/Comments (area) because it has the highest number of missing values
'''
import pandas as pd
import csv
import json
from sklearn.ensemble import RandomForestRegressor

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from scipy.stats import kendalltau
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

ROOT_PATH = '/home/nadeeshan/ML-Experiments/model/'
df = pd.read_csv(ROOT_PATH + 'data/understandability_with_warnings.csv')

## encode the developer_position
## if developer_position is "bachelor student" then 1
## elif developer_position is "master student" then 2
## elif developer_position is phd student" then 3
## elif developer_position is "professional developer" then 4
df['developer_position'] = df['developer_position'].replace(['bachelor student', 'master student', 'phd student', 'professional developer'], [1, 2, 3, 4])
df.to_csv(ROOT_PATH + 'data/understandability_with_warnings.csv', index=False)

## all code features 
features = [
    "CR",
    "CIC (avg)",	
    "CIC (max)",
    "CICsyn (avg)",
    "CICsyn (max)",
    "MIDQ (min)",	
    "MIDQ (avg)",
    "MIDQ (max)",
    "AEDQ (min)",
    "AEDQ (avg)",
    "AEDQ (max)",
    "EAP (min)",
    "EAP (avg)",
    "EAP (max)",
    "Cyclomatic complexity", 
    "IMSQ (min)", 
    "IMSQ (avg)", 
    "IMSQ (max)",
    "Readability", 
    "ITID (avg)",
    "NMI (avg)",
    "NMI (max)",
    "NM (avg)", 
    "NM (max)", 
    "TC (avg)",
    "TC (min)",
    "TC (max)",
    "#assignments (avg)",
    "#blank lines (avg)",
    "#commas (avg)",
    "#comments (avg)",
    "#comparisons (avg)", 
    "Identifiers length (avg)", 
    "#conditionals (avg)",
    "Indentation length (avg)",
    "#keywords (avg)",
    "Line length (avg)",
    "#loops (avg)", 
    "#identifiers (avg)", 
    "#numbers (avg)",
    "#operators (avg)", 
    "#parenthesis (avg)", 
    "#periods (avg)",
    "#spaces (avg)",
    "Identifiers length (max)",
    "Indentation length (max)", 
    "#keywords (max)", 
    "Line length (max)",
    "#identifiers (max)",
    "#numbers (max)",
    "#characters (max)",
    "#words (max)",
    "Entropy",
    "Volume",
    "LOC",
    "#assignments (dft)",
    "#commas (dft)",
    "#comments (dft)", 
    "#comparisons (dft)",
    "#conditionals (dft)",
    "Indentation length (dft)",
    "#keywords (dft)",
    "Line length (dft)",
    "#loops (dft)",
    "#identifiers (dft)",
    "#numbers (dft)",
    "#operators (dft)",
    "#parenthesis (dft)",
    "#periods (dft)",
    "#spaces (dft)",
    "Comments (Visual X)",
    "Comments (Visual Y)",
    "Identifiers (Visual X)",
    "Identifiers (Visual Y)",
    "Keywords (Visual X)",
    "Keywords (Visual Y)",
    "Numbers (Visual X)",
    "Numbers (Visual Y)",
    "Strings (Visual X)",
    "Strings (Visual Y)",
    "Literals (Visual X)",
    "Literals (Visual Y)",
    "Operators (Visual X)",
    "Operators (Visual Y)",
    "Comments (area)",
    "Identifiers (area)",
    "Keywords (area)",
    "Numbers (area)",
    "Strings (area)",
    "Literals (area)",
    "Operators (area)",	
    "Keywords/identifiers (area)",
    "Numbers/identifiers (area)",
    "Strings/identifiers (area)",
    "Literals/identifiers (area)",
    "Operators/literals (area)",
    "Numbers/keywords (area)",
    "Strings/keywords (area)",
    "Literals/keywords (area)",
    "Operators/keywords (area)", 
    "#aligned blocks",
    "Extent of aligned blocks",
    "#nested blocks (avg)",
    "#parameters",
    "#statements", 
    "Strings/comments (area)",
    "Keywords/comments (area)",
    "Literals/comments (area)",
    "Strings/numbers (area)",
    "Numbers/comments (area)",
    "Operators/numbers (area)",
    "Literals/numbers (area)",
    "Identifiers/comments (area)",
    "Literals/strings (area)",
    "Operators/strings (area)",
    "Operators/comments (area)",
    "Operators/literals (area).1",
    "developer_position",
    "PE gen", 
    "PE spec (java)"
]

dev_features = []

## Feature set 1: Consider only the code features. Took Highly correlated features. Removed one of highly correlated ones which has the larget missing values or feature1
## perfom SBFS to select the best features
## final_features1_bfs.txt

## Feature set 2: Consider only the code features. Took Highly correlated features. Removed one of highly correlated ones which has the larget missing values or feature2
## perfom SBFS to select the best features
## final_features2_bfs.txt

## Feeature set 3 Consider only code features. Standard Features from the paper in the replication package.
## remove below features because they are not code features.
## PBU - PE spec (java), MIDQ (max), CICsyn (max), 
## ABU50 - PE spec (java)
## BD50 - PE spec (java), PE gen, CR

## Features:
## PBU - "NMI (avg)", "TC (avg)", "TC (max)", "Line length (max)", "LOC", "Line length (dft)", "Strings (area)"
## ABU50 - "NMI (avg)", "TC (avg)", "#conditionals (avg)", "#periods (avg)", "Identifiers length (max)", "#comparisons (dft)", "Indentation length (dft)","Literals (Visual X)", "#parameters"
## BD50 - "Cyclomatic complexity", "#assignments (avg)", "#words (max)", "#conditionals (dft)", "Numbers (Visual X)", "Literals (Visual Y)" , "#nested blocks (avg)"


## Feature set 4 consider all code features. Remove below features because they have the highest number of missing values.
## "Operators/literals (area).1", "Operators/comments (area)", "Identifiers/comments (area)", "Literals/numbers (area)", "Operators/numbers (area)", Numbers/comments (area), Strings/numbers (area), Literals/comments (area), Keywords/comments (area), Strings/comments (area), Operators/strings (area),
## Literals/strings (area)

regression_target = ["TNPU", "TAU", "AU"]
X = df[features]
y = df[regression_target]

stat_header = {'feature1':'', 'feature2': '', 'tau': 0, 'p-value':0, 'missing values in feature1':0, 'missing values in feature2':0, 'feature to remove':''}

RANDOM_SEED = 1699304546

######################################
## Feature selection ##
######################################

def kendals_feature_selection(features, featureToRemove, output_file):
    ## write header
    with open(ROOT_PATH + 'feature_selection/regression/' + output_file, "w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
        writer.writeheader()

    for feature1 in features:
        for feature2 in features[features.index(feature1)+1:]:
            tau,p = kendalltau(X[feature1], X[feature2], nan_policy='omit') ## only perform pair-wise comparisions where both the data points has values (i.e not NAN)
            if abs(tau) >= 0.7 and feature1 != feature2:
                ## number of data points with missing values
                feature1_missing = X[feature1].isna().sum()
                feature2_missing = X[feature2].isna().sum()
                
                ## remove one of the features which has the highest number of missing values.
                if feature1_missing > feature2_missing:
                    feature_to_remove = feature1
                elif feature1_missing < feature2_missing:
                    feature_to_remove = feature2
                else:
                    feature_to_remove = feature1 if featureToRemove == "feature1" else feature2
                # write to csv
                with open(ROOT_PATH + 'feature_selection/regression/' + output_file, "a") as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
                    writer.writerow({'feature1':feature1, 'feature2': feature2, 'tau': tau, 'p-value': p, 'missing values in feature1':feature1_missing, 'missing values in feature2':feature2_missing, 'feature to remove': feature_to_remove})


def linear_floating_forward_feature_selection(df_features_X, df_target_y, kFold):
    '''
    Perform linear floating forward selection with wrapper strategy 
    We select RandomForest Classifier as the wrapper
    We evaluate the performance of the model using cross validation and F1ge as the metric
    Sequential Forward Floating Selection
    How Sequential Forward Selection - SFS works?
    1. Given 10 features originally
    2. Algorithm starts with an empty subset.
    3. First add one feature, train the classifier using cv and get the evaluation score F1. This is done for each feature.
    4. After this round we pick the feature that gives the best F1 score.
    5. Repeat this process upto (1, max_k_features)

    How Floating Forward Feature Selection works?
    1. Given 10 features originally.( a b c d e f g h i j)
    2. Same as SFS lets assume after round 4, we have 4 features selected (say a.b,c,d). 
    3. Floating basically removes a feature if removing improves performance.
    4. In our example Floating is doing feature removal process in the round 5.
    5. In Floating Round 5, it removes features one by one, train the classifier and get performance.
    6. Only if removing a feature improves the perfromace compared to last round, we keep the new feature subset.
    As example lets assume (a b d) improves the performace than using (a b c d). Then we keep (a b d ) for the round 5 of feature selection.
    7. Now round 5 of feature selection is done with 3 features. 
    https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/#example-2-toggling-between-sfs-sbs-sffs-and-sbfs
    '''
    print('\nSequential Forward Floating Feature Selection (for getting the best optinal subset of features k=1...max_k)')
    ## max number of features
    max_k = len(df_features_X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_features_X, df_target_y, test_size=0.2, random_state=RANDOM_SEED)
    
    param_grid = {
                "n_estimators": [5, 10, 50],
                "max_features": ["sqrt", "log2", 0.5, max_k],
                "max_depth": [2, 5, 10, 20],
                "bootstrap": [True, False],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "random_state": [RANDOM_SEED]
            }
    rf = RandomForestRegressor()
    grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kFold, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train.to_numpy().ravel())
    lr = grid.best_estimator_

    sffs = SFS(estimator=lr, forward=False, floating=True, scoring='neg_mean_squared_error', cv=kFold, n_jobs=-1, k_features=(1, max_k))
    sffs = sffs.fit(X_train, y_train.to_numpy().ravel())
        
    return list(sffs.k_feature_names_)

def linear_floating_backward_feature_selection(df_features_X, df_target_y, kFold):
    '''
    This is same as Linear Floating Forward Selection. Only difference is starting with all the features
    and remove one by one at a time. 
    '''
    print('\nSequential Backward Floating Feature Selection (for getting the best optinal subset of features k=1...max_k)')
    ## max number of features
    max_k = len(df_features_X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        df_features_X, df_target_y, test_size=0.2, random_state=RANDOM_SEED)
    
    param_grid = {
            "n_estimators": [5, 10, 50],
            "max_features": ["sqrt", "log2", 0.5, max_k],
            "max_depth": [2, 5, 10, 20],
            "bootstrap": [True, False],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "random_state": [RANDOM_SEED]
        }
    rf = RandomForestRegressor()
    grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kFold, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train.to_numpy().ravel())
    lr = grid.best_estimator_

    sffs = SFS(estimator=lr, forward=False, floating=True, scoring='neg_mean_squared_error', cv=kFold, n_jobs=-1, k_features=(1, max_k))
    sffs = sffs.fit(X_train, y_train.to_numpy().ravel())
    
    return list(sffs.k_feature_names_)


## STEP 1 ##
## Remove redundent features
## How we do it:
## 1. Perform Kendall's Tau correlation test between each pair of features.
## 2. Pick the pairs that show |Tau| >= 0.7 (highly correlated)
## 3. Remove one of the features which has the highest number of missing values.
## 4. If missing values are equal, remove one of the features randomly.
# kendals_feature_selection(features, "feature1", "kendall_highly_correlated_features_part1.csv") ## remove the first feature in the pair
# kendals_feature_selection(features, "feature2", "kendall_highly_correlated_features_part2.csv") ## remove the second feature in the pair


## STEP 2 ##
## Remove Keywords/Comments (area) because it has the highest number of missing values
## and create the final feature set
def remove_specific_feature(input_feature_file):
    df_remove_fea = pd.read_csv(ROOT_PATH + 'feature_selection/regression/' + input_feature_file)
    df_remove_fea = df_remove_fea['feature to remove']
    features_to_remove = list(set(df_remove_fea.to_list())) 
    features_to_remove.append('Keywords/comments (area)') ## append the specific feature to remove
    final_list_of_features = list(set(features) - set(features_to_remove))        

    return final_list_of_features        

features_part1 = remove_specific_feature('kendall_highly_correlated_features_part1.csv')
# write the final_list_of_features after removing redeundent and specific features.
with open(ROOT_PATH + 'feature_selection/regression/unique_feature_set1.txt', 'w+') as f:
    for feature in features_part1:
        f.write("%s\n" % feature) 
features_part2 = remove_specific_feature('kendall_highly_correlated_features_part2.csv')
with open(ROOT_PATH + 'feature_selection/regression/unique_feature_set2.txt', 'w+') as f:
    for feature in features_part2:
        f.write("%s\n" % feature) 

## STEP 3 ##
## Perform linear floating forward feature selection
## We select logistic regression as the wrapper
## We evaluate the performance of the model using 5-fold cross validation and F1 as the metric
## Sequential Forward Floating Selection
## https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/#example-2-toggling-between-sfs-sbs-sffs-and-sbfs
## StratifiedKFold
Folds = 5

kFold = KFold(n_splits=Folds, shuffle=True, random_state=RANDOM_SEED)
with open(ROOT_PATH + 'feature_selection/regression/final_features1_bfs.txt','w+') as file1:
    file1.write('')
with open(ROOT_PATH + 'feature_selection/regression/final_features2_bfs.txt','w+') as file2:
    file2.write('')
with open(ROOT_PATH + 'feature_selection/regression/final_features1_lfs.txt','w+') as file1:
    file1.write('')
with open(ROOT_PATH + 'feature_selection/regression/final_features2_lfs.txt','w+') as file2:
    file2.write('')    

for target in regression_target:

    df = pd.read_csv(ROOT_PATH + 'data/understandability_with_warnings.csv')
    df = df.dropna(subset=[target])
    X = df[features]
    y = df[regression_target]
    print('Target: ', target)

    final_features1 = linear_floating_forward_feature_selection(X[features_part1], y[target], kFold)
    final_features2 = linear_floating_forward_feature_selection(X[features_part2], y[target], kFold)
    # write to text file
    with open(ROOT_PATH + 'feature_selection/regression/final_features1_lfs.txt','a') as file1:
        ## add dev_features to final_features1 
        final_features1.extend(dev_features)
        file1.write(target + json.dumps(final_features1) + '\n')
    with open(ROOT_PATH + 'feature_selection/regression/final_features2_lfs.txt','a') as file2:
        ## add dev_features to final_features2
        final_features2.extend(dev_features)
        file2.write(target + json.dumps(final_features2) + '\n') 

    final_features1 = linear_floating_backward_feature_selection(X[features_part1], y[target], kFold) 
    final_features2 = linear_floating_backward_feature_selection(X[features_part2], y[target], kFold) 
    with open(ROOT_PATH + 'feature_selection/regression/final_features1_bfs.txt','a') as file3:
        ## add dev_features to final_features1 
        final_features1.extend(dev_features)
        file3.write(target + json.dumps(final_features1) + '\n')
    with open(ROOT_PATH + 'feature_selection/regression/final_features2_bfs.txt','a') as file4:
        ## add dev_features to final_features2
        final_features2.extend(dev_features)
        file4.write(target + json.dumps(final_features2) + '\n')      
