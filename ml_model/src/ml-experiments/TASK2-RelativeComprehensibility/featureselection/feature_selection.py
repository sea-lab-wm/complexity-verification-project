import sys
# from utils import configs
####### 
from utils import configs_TASK2_DS3 as configs

sys.path.append(configs.ROOT_PATH)


from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

class FeatureSelection:
    def __init__(self, feature_df, y, code_comprehension_target, dynamic_epsilon):
        self.feature_df = feature_df
        self.code_comprehension_target = code_comprehension_target
        self.dynamic_epsilon = dynamic_epsilon
        self.X = self.feature_df
        # print(self.X.columns[self.X.isnull().any()]) ## print the columns with NaN values
        self.X = self.X.dropna(axis=1) ## drop columns with NaN values
        self.y = y
        
    def compute_mutual_information(self):
        """
        This function computes the mutual information of each feature with the target variable
        Parameters 
            X: input features
            y: target variable (discrete)
            k: number of best features to select
        returns: the list of the features with their mutual information in natural unit of information (nat)
        
        mi = H(X) - H(X|Y)
        where H(X) is the entropy of X, and H(X|Y) is the conditional entropy of X given Y
        https://www.kaggle.com/code/vickysen/feature-selection-using-information-gain
        """
        # mi = mutual_info_classif(self.X, self.y, discrete_features='auto', random_state=configs.RANDOM_SEED)

        ## compute the mutual information each column one by one and create a pandas series with the mi values
        ## and the column names as the index of the series
        mi_dict = {}
        for feature in self.X.columns:
            mi_info = mutual_info_classif(self.X[feature].values.reshape(-1, 1), self.y, discrete_features='auto', random_state=configs.RANDOM_SEED)
            mi_dict[feature] = mi_info[0]

        mi = mi_dict.values()
        mi = pd.Series(mi, index=mi_dict.keys()) #one dimensional array with index labels
        mi.index = self.X.columns

        # mi.sort_values(ascending=False).plot.bar(figsize=(20, 20))
        # plt.title('Mutual_Information_' + self.code_comprehension_target + '_' + self.y.columns[0] + '_' + str(self.dynamic_epsilon))
        # plt.ylabel('Mutual Information')
        
        
        # plt.savefig(configs.ROOT_PATH + '/featureselection/mutual_info_' + self.code_comprehension_target + '_' +  self.y.columns[0] + '_' + 'dynamic_epsilon_' + str(self.dynamic_epsilon) +'.png')

        ## save the mutual information to a csv file
        with open(configs.ROOT_PATH + '/' + configs.MI_OUTPUT_FILE_NAME, "a") as csv_file:
            for feature in self.X.columns: 
                csv_file.write(feature + "," + self.code_comprehension_target + ',' + str(self.dynamic_epsilon) +',' + str(mi_dict[feature]) + '\n')
    
       
        mi = pd.read_csv(configs.ROOT_PATH + '/' + configs.MI_OUTPUT_FILE_NAME)
        

        ## Filter by the target variable
        mi = mi[(mi['target'] == self.code_comprehension_target ) & (mi['dynamic_epsilon'] == self.dynamic_epsilon)]

        self.mi = mi

        return self.mi

    def compute_kendals_tau(self):
        """
        This function computes the kendal's tau correlation of each feature with the target variable
        Parameters 
            X: input features
            y: target variable (discrete)
        returns: the list of the features with their kendal's tau correlation
        """
        for feature in self.X.columns:
            tau, p_value = kendalltau(self.X[feature], self.y, nan_policy='omit')
            with open(configs.ROOT_PATH + '/' + configs.KENDALS_OUTPUT_FILE_NAME, "a") as csv_file:
                csv_file.write(feature + "," + self.code_comprehension_target + ',' + str(self.dynamic_epsilon) +',' + str(tau) + ',' + str(abs(tau)) + ',' + str(p_value) + '\n')
            

        ## Filter the features with the target variable
        kendall = pd.read_csv(configs.ROOT_PATH + '/' + configs.KENDALS_OUTPUT_FILE_NAME)
        
        kendall = kendall[(kendall['target'] == self.code_comprehension_target) & (kendall['dynamic_epsilon'] == self.dynamic_epsilon)]

        self.kendall = kendall
        return kendall

    def select_k_best(self, k, method):
        """
        This function selects the k best features based on the mutual information
        obtained from the compute_mutual_information function
        Parameters 
            k: number of best features to select
        returns: 
            the list of the features with their mutual information in natural unit of information (nat)
        """
        if method == "mi":
            ## order mi in descending order
            mi = self.mi.sort_values(by='mi', ascending=False)
            ## select the k best features
            k_best_features = mi.head(k)['feature']
        if method == "kendalltau":
            ## order |tau| in descending order
            kendall = self.kendall.sort_values(by='|tau|', ascending=False)
            ## select the k best features
            k_best_features = kendall.head(k)['feature']
        return k_best_features    