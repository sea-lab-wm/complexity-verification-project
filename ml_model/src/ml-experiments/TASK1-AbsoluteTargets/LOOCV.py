import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import precision_score, recall_score, confusion_matrix

from imblearn.pipeline import Pipeline
import logging

import csv

ROOT_PATH="/home/nadeeshan/ML-Experiments-2/complexity-verification-project/ml_model/model/Italian-Replication"

## Logger
_LOG_FMT = '[%(asctime)s - %(levelname)s - %(name)s]-   %(message)s'
_DATE_FMT = '%m/%d/%Y %H:%M:%S'
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
LOGGER = logging.getLogger('__main__')

RANDOM_SEED=1

## skip warnings
import warnings
warnings.filterwarnings("ignore")


from sklearn.random_projection import GaussianRandomProjection as RandomProjection
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from imblearn.over_sampling import SMOTE


targets = ["PBU", "ABU50", "BD50"]
targets = {
    "PBU": ["java_experience", "MaxInternalDocumentation", "READABILITY.New.Abstractness.words.AVG", "READABILITY.New.Synonym.commented.words.MAX", "READABILITY.New.Text.Coherence.AVG", "READABILITY.New.Text.Coherence.MAX", "READABILITY.BW.Max.line.length", "READABILITY.Posnett.lines", "READABILITY.Dorn.DFT.LineLengths", "READABILITY.Dorn.Areas.Strings"],
    "ABU50": ["java_experience", "READABILITY.New.Abstractness.words.AVG", "READABILITY.New.Text.Coherence.AVG", "READABILITY.BW.Avg.conditionals", "READABILITY.BW.Avg.periods", "READABILITY.BW.Max.Identifiers.Length", "READABILITY.Dorn.DFT.Comparisons", "READABILITY.Dorn.DFT.Indentations","READABILITY.Dorn.Visual.X.Literals", "KASTO.NumberOfParameters"],
    "BD50": ["programming_experience", "java_experience", "complexity", "READABILITY.New.Comments.readability", "READABILITY.BW.Avg.Assignment", "READABILITY.BW.Max.words", "READABILITY.Dorn.DFT.Conditionals", "READABILITY.Dorn.Visual.X.Numbers", "READABILITY.Dorn.Visual.X.Literals" , "KASTO.NestedBlocks"]
}

def pipeline_initialisation(model_name, target):
    LOGGER.info("Launching model: " + model_name + "...")

    if target == "PBU":
            smote = SMOTE(
                k_neighbors=5,            # -K 5: Number of nearest neighbors to use
                sampling_strategy='auto',   # -P 125.0: Ratio of the number of synthetic samples (125% increase)
                random_state=1            # -S 1: Random seed for reproducibility
                )
    elif target == "BD50":
            smote = SMOTE(
                k_neighbors=5,            # -K 5: Number of nearest neighbors to use
                sampling_strategy='all',   # -P 304.0: Ratio of the number of synthetic samples (304% increase)
                random_state=1            # -S 1: Random seed for reproducibility
                )

    if model_name == "logisticregression":
        
        if target == "BD50" or target == "PBU":
            best_model = LogisticRegression(
                C=1.0E8,             # -R 1.0E-8: Regularization parameter (inverse of regularization strength)
                max_iter=1000,       # -M -1: Maximum number of iterations (a high number to ensure convergence)
                solver='lbfgs',      # Solver used to optimize the logistic regression problem (default is 'lbfgs')
                multi_class='auto')   # Multi-class handling (default is 'auto' and can be adjusted as needed))
            pipeline = Pipeline(steps = [
                ('smote', smote),
                (model_name, best_model)])
        elif target == "ABU50":
            best_model = LogisticRegression(
                C=1.0E8,             # -R 1.0E-8: Regularization parameter (inverse of regularization strength)
                max_iter=1000,       # -M -1: Maximum number of iterations (a high number to ensure convergence)
                solver='lbfgs',      # Solver used to optimize the logistic regression problem (default is 'lbfgs')
                multi_class='auto')
            pipeline = Pipeline(steps = [
                (model_name, best_model)])


    elif model_name == "knn_classifier":
        if target == "PBU":
            pipeline = Pipeline([
                ('smote', smote),
                ('random_projection', RandomProjection(n_components=10, random_state=42)),
                ('knn', KNeighborsClassifier(n_neighbors=3, weights='uniform'))])

        elif target == "ABU50":
            pipeline = Pipeline([
                ('knn', KNeighborsClassifier(n_neighbors=4, weights='uniform'))])  

        elif target == "BD50":
            pipeline = Pipeline([
                ('smote', smote),
                ('random_projection', RandomProjection(n_components=10, random_state=42)),
                ('knn', KNeighborsClassifier(n_neighbors=5, weights='uniform'))])
                 
    elif model_name == "randomForest_classifier":
        if target == "PBU":
            pipeline = Pipeline([
                ('smote', smote),
                ('rf', RandomForestClassifier(
                        n_estimators=100,       # -I 100: Number of trees
                        max_features=1,         # -K 1: Number of features to consider for splitting a node
                        min_samples_leaf=1,     # -M 1.0: Minimum number of samples required to be at a leaf node
                        random_state=1))        # -S 1: Random seed
                      ])

        elif target == "ABU50":
            pipeline = Pipeline([
                ('rf', RandomForestClassifier(
                        n_estimators=100,         # -I 100: Number of trees
                        max_features=5,           # -K 5: Number of features to consider for splitting a node
                        min_samples_leaf=1,       # -M 1.0: Minimum number of samples required to be at a leaf node
                        random_state=1            # -S 1: Random seed
                    ))])

        elif target == "BD50":
            pipeline = Pipeline([
                ('smote', smote),
                ('rf', RandomForestClassifier(
                    n_estimators=100,         # -I 100: Number of trees
                    max_features=8,           # -K 8: Number of features to consider for splitting a node
                    min_samples_leaf=1,       # -M 1.0: Minimum number of samples required to be at a leaf node
                    random_state=1            # -S 1: Random seed
                    ))])
            
    elif model_name == "svc":
        if target == "PBU":
            model = SVC(
                    C=1.0,                   # -C 1.0: Regularization parameter
                    kernel='poly',           # -K: Polynomial kernel
                    degree=3,                # -K -E 3.0: Degree of the polynomial kernel
                    coef0=250007,            # -K -C 250007: Coefficient of the polynomial kernel
                    probability=True,        # Enable probability estimates (needed for calibration)
                    random_state=42
                    )
            pipeline = Pipeline([
                ('smote', smote),
                ('svc', model)])
                

        elif target == "ABU50":
            model = SVC(
                    C=1.0,                   # -C 1.0: Regularization parameter
                    kernel='poly',           # -K: Polynomial kernel
                    degree=2,                # -K -E 2.0: Degree of the polynomial kernel
                    coef0=250007,            # -K -C 250007: Coefficient for the polynomial kernel
                    probability=True,        # Enable probability estimates (needed for calibration)
                    random_state=42
                    )
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svc', model)])

        elif target == "BD50":
            model = SVC(
                    C=3.0,                   # -C 3.0: Regularization parameter
                    kernel='poly',           # -K: Polynomial kernel
                    degree=2,                # -K -E 2.0: Degree of the polynomial kernel
                    coef0=250007,            # -K -C 250007: Coefficient for the polynomial kernel
                    probability=True,        # Enable probability estimates (needed for calibration)
                    random_state=42
                    )
            pipeline = Pipeline([
                ('smote', smote),
                ('scaler', StandardScaler()),
                ('svc', model)
                ])  
    elif model_name == "mlp_classifier":
        if target == "PBU":
            pipeline = Pipeline([
                ('smote', smote),
                ('mlp', MLPClassifier(
                    hidden_layer_sizes=(50,),    # -H 50: Number of hidden units (one hidden layer with 50 units)
                    learning_rate_init=0.8,       # -L 0.8: Initial learning rate
                    momentum=0.5,                 # -M 0.5: Momentum
                    max_iter=500,                 # -N 500: Maximum number of iterations
                    random_state=0,               # -S 0: Random seed
                    early_stopping=True,          # -E 20: Enable early stopping
                    n_iter_no_change=20           # -E 20: Number of iterations with no improvement before stopping
                    ))])

        elif target == "ABU50":
            pipeline = Pipeline([
                ('mlp', MLPClassifier(
                    hidden_layer_sizes=(20,),    # -H 20: Number of hidden units (one hidden layer with 20 units)
                    learning_rate_init=0.8,       # -L 0.8: Initial learning rate
                    momentum=0.1,                 # -M 0.1: Momentum
                    max_iter=500,                 # -N 500: Maximum number of iterations
                    random_state=0,               # -S 0: Random seed
                    early_stopping=True,          # -E 20: Enable early stopping
                    n_iter_no_change=20           # -E 20: Number of iterations with no improvement before stopping
                    ))])

        elif target == "BD50":
            pipeline = Pipeline([
                ('smote', smote),
                ('mlp', MLPClassifier(
                    hidden_layer_sizes=(40,),    # -H 40: One hidden layer with 40 units
                    learning_rate_init=0.6,       # -L 0.6: Initial learning rate
                    momentum=0.1,                 # -M 0.1: Momentum
                    max_iter=500,                 # -N 500: Maximum number of iterations
                    random_state=0,               # -S 0: Random seed
                    early_stopping=True,          # -E 20: Enable early stopping
                    n_iter_no_change=20           # -E 20: Number of iterations with no improvement before stopping
                ))])
    elif model_name == "bayes_network":
        if target == "PBU" or target == "BD50":
            pipeline = Pipeline([
                ('smote', smote),
                ('bayes', GaussianNB())
            ])
        elif target == "ABU50":
            pipeline = Pipeline([
                ('bayes', GaussianNB())
            ])

    return pipeline        

## save the results to csv
csv_data_dict = {
    "model": "",
    "target": "",
    "y_actual_index": "",
    "y_actual": "",
    "y_pred": "",
    "tp": 0,
    "tn": 0,
    "fp": 0,
    "fn": 0,
    "n_instances": 0,
    "precision_weighted": 0.0,
    "recall_weighted": 0.0,
    "f1_weighted": 0.0,
    "auc_weighted": 0.0,
}        

## write header
with open(ROOT_PATH + "/results/" + "Italian_results.csv", "w+") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_data_dict.keys())
    writer.writeheader()

for target in targets.keys():

    if target == "PBU":
        df_train = pd.read_csv(ROOT_PATH + "/data/dataset_pbu.traintest.csv")
    elif target == "ABU50":
        df_train = pd.read_csv(ROOT_PATH + "/data/dataset_abu.traintest.csv")
    elif target == "BD50":
        df_train = pd.read_csv(ROOT_PATH + "/data/dataset_bd.traintest.csv")
    
    ## take the features data
    features_X = df_train[targets[target]]

    ## take the target data
    target_y = df_train[target]

    ## leave-one-out cross-validation for hyperparameter tuning
    from sklearn.model_selection import LeaveOneOut

    loo = LeaveOneOut()
    loo.get_n_splits(features_X)


    for model_name in ["logisticregression", "knn_classifier", "randomForest_classifier", "mlp_classifier", "bayes_network"]:
        LOGGER.info("##### TARGET :  " + target + "########")
        
        pipeline = pipeline_initialisation(model_name, target=target)

        # source: https://machinelearningmastery.com/loocv-for-evaluating-machine-learning-algorithms/
        # enumerate splits
        y_test_id, y_true, y_pred = list(), list(), list()
        for train_ix, test_ix in loo.split(features_X, target_y):
            # split data
            X_train, X_test = features_X.iloc[train_ix, :], features_X.iloc[test_ix, :]
            y_train, y_test = target_y.iloc[train_ix], target_y.iloc[test_ix]
            
            # fit model
            pipeline.fit(X_train, y_train)
            
            # evaluate model
            yhat = pipeline.predict(X_test)
            
            # store
            y_test_id.append(test_ix[0])
            y_true.append(y_test.values[0])
            y_pred.append(yhat[0])

        
        # calculate accuracy
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        auc = roc_auc_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        confusion_matrix1 = confusion_matrix(y_true, y_pred)

        csv_data_dict["model"] = model_name
        csv_data_dict["target"] = target
        csv_data_dict["y_actual_index"] = [int(x) for x in y_test_id]
        csv_data_dict["y_actual"] = [int(x) for x in y_true]
        csv_data_dict["y_pred"] = [int(x) for x in y_pred]
        csv_data_dict["tp"] = confusion_matrix1[1][1]
        csv_data_dict["tn"] = confusion_matrix1[0][0]
        csv_data_dict["fp"] = confusion_matrix1[0][1]
        csv_data_dict["fn"] = confusion_matrix1[1][0]
        csv_data_dict["n_instances"] = len(y_true)
        csv_data_dict["precision_weighted"] = precision
        csv_data_dict["recall_weighted"] = recall
        csv_data_dict["f1_weighted"] = f1
        csv_data_dict["auc_weighted"] = auc


        with open(ROOT_PATH + "/results/" + "Italian_results.csv", "a") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_data_dict.keys())
            writer.writerow(csv_data_dict)