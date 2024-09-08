
import sys
import pandas as pd
sys.path.append('/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model/src/ml-experiments/TASK2-RelativeComprehensibility/')
from utils import configs_DS3 as configs


def add_code_features(relative_comprehensibility_file_name, output_file_name):
    ## read the syntactic data
    code_features = pd.read_csv(configs.ROOT_PATH + "/" + "data/final_features_ds3.csv")

    ## drop the columns that are not needed
    ## EAP features are dropping because, no clear explanation on the implementation of the feature 
    code_features = code_features.drop(columns=configs.NOT_USEFUL_FEATURES)

    ## relative code comprehensibility
    relative_comprehensibility = pd.read_csv(configs.RELATIVE_COMPREHENSIBILITY_ROOT_PATH + "/data/" + relative_comprehensibility_file_name)

    #################################################################
    #### S1 = Merge relative code comprehensibility with the data ####
    #################################################################
    s1_merged_df = relative_comprehensibility.merge(code_features, left_on='s1', right_on='snippet_id', how='inner')
    s1_merged_df = s1_merged_df.drop_duplicates()
    s1_merged_df = s1_merged_df.drop(columns=['snippet_id'])

    ## remove the index column
    s1_merged_df = s1_merged_df.set_index('s1')

    s1_merged_df.to_csv(configs.RELATIVE_COMPREHENSIBILITY_ROOT_PATH + "/data/s1_features_with_warnings.csv")

    ##################################################################
    #### S2 - Merge relative code comprehensibility with the data ####
    ##################################################################
    s2_merged_df = relative_comprehensibility.merge(code_features, left_on='s2', right_on='snippet_id', how='inner')
    s2_merged_df = s2_merged_df.drop_duplicates()
    s2_merged_df = s2_merged_df.drop(columns=['snippet_id'])

    ## remove the index column
    s2_merged_df = s2_merged_df.set_index('s1')

    s2_merged_df.to_csv(configs.RELATIVE_COMPREHENSIBILITY_ROOT_PATH + "/data/s2_features_with_warnings.csv")

    #############################################################
    ## train dataset ##
    #############################################################
    # train_df = # Merge dataframes based on common columns
    complete_df = pd.merge(s1_merged_df, s2_merged_df, on=["s1","s2","target","s1_comprehensibility","s2_comprehensibility","(s2-s1)diff","(s2>s1)relative_comprehensibility", "epsilon","dynamic_epsilon"])
    complete_df.to_csv(configs.RELATIVE_COMPREHENSIBILITY_ROOT_PATH + "/data/" + output_file_name)


if __name__ == "__main__":
    ## arguments from the 
    add_code_features()