from sklearn.feature_selection import mutual_info_classif
import sys
# from utils import configs

####### Uncomment this to use config for DS3 ##
from utils import configs_TASK1_DS6 as configs
####### 
# from utils import configs_DS3 as configs

sys.path.append(configs.ROOT_PATH)



import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

class FeatureSelection:
    def __init__(self, feature_df, y, drop_duplicates, dataset_key):
        self.feature_df = feature_df
        self.drop_duplicates = drop_duplicates
        self.dataset = dataset_key
        self.X = feature_df
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
        mi = mutual_info_classif(self.X, self.y.values, discrete_features='auto', n_neighbors=3, copy=True, random_state=configs.RANDOM_SEED)

        mi = pd.Series(mi) #one dimensional array with index labels
        mi.index = self.X.columns
        mi.sort_values(ascending=False).plot.bar(figsize=(20, 20))
        plt.title('Mutual Information with respect to the '+ self.y.name)
        plt.ylabel('Mutual Information')
        drop_dup = 'drop_duplicates' if self.drop_duplicates else 'no_drop_duplicates'
        # if self.dataset == "ds_code":
        #     plt.savefig(configs.ROOT_PATH + '/NewExperiments/featureselection/rq1/mutual_info_' + self.dataset + '_' + self.y.name + '_' + drop_dup + '.png')
        # else:
        #     plt.savefig(configs.ROOT_PATH + '/NewExperiments/featureselection/rq2/mutual_info_' + self.dataset + '_' + self.y.name + '_' + drop_dup + '.png')

        ## save the mutual information to a csv file
        if self.dataset == "ds_code":
            with open(configs.ROOT_PATH + '/' + configs.MI_OUTPUT_PATH, "a") as csv_file:
                for feature in self.X.columns:   
                    csv_file.write(feature + "," + self.y.name + ',' + str(mi[feature]) + ',' + self.dataset + ',' + str(self.drop_duplicates) + '\n')
        # else:
        #     with open(configs.ROOT_PATH + '/' + configs.MI_OUTPUT_FILE_NAME_RQ2, "a") as csv_file:
        #         for feature in self.X.columns:   
        #             csv_file.write(feature + "," + self.y.name + ',' + str(mi[feature]) + ',' + self.dataset + ',' + str(self.drop_duplicates) + '\n')

        # mi = pd.read_csv(configs.ROOT_PATH + '/' + configs.MI_OUTPUT_FILE_NAME_RQ2)
        if self.dataset == "ds_code":
            mi = pd.read_csv(configs.ROOT_PATH + '/' + configs.MI_OUTPUT_FILE_NAME)
        

        ## Filter by the target variable and drop_duplicates columns
        mi = mi[(mi['target'] == self.y.name) & (mi['dataset'] == self.dataset) & (mi['drop_duplicates'] == self.drop_duplicates)]
        self.mi = mi

        return mi

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
            
            if self.dataset == "ds_code":
                with open(configs.ROOT_PATH + '/' + configs.KENDALS_OUTPUT_FILE_NAME, "a") as csv_file:
                    csv_file.write(feature + "," + self.y.name + ',' + str(tau) + ',' + str(abs(tau)) + ',' + str(p_value) + ',' + self.dataset + ',' + str(self.drop_duplicates) + '\n')
            # else:
            #     with open(configs.ROOT_PATH + '/' + configs.KENDALS_OUTPUT_FILE_NAME_RQ2, "a") as csv_file:
            #         csv_file.write(feature + "," + self.y.name + ',' + str(tau) + ',' + str(abs(tau)) + ',' + str(p_value) + ',' + self.dataset + ',' + str(self.drop_duplicates) + '\n')
            

        ## Filter the features with the target variable
        # kendall = pd.read_csv(configs.ROOT_PATH + '/' + configs.KENDALS_OUTPUT_FILE_NAME_RQ2)
        if self.dataset == "ds_code":
            kendall = pd.read_csv(configs.ROOT_PATH + '/' + configs.KENDALS_OUTPUT_FILE_NAME)
        
        kendall = kendall[(kendall['target'] == self.y.name) & (kendall['dataset'] == self.dataset) & (kendall['drop_duplicates'] == self.drop_duplicates)]

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