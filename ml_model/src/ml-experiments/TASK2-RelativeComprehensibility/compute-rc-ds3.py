"""
Relative comprehensibility metric computation

How to compute Relative Comprehensibility metric:

Let's take a single metric from DS6. Eg. PBU(binary)

Snippet(S) 
S1 = PBU 1 (6) PBU 0 (3)  ⇒ 6/9 = 0.667 say S1 is understandability
S2 = PBU 1 (5) PBU 0 (3)  ⇒ 5/8 = 0.625 Say S2 is understandability

Now we define,
For Binary Target Variables (like PBU in DS6)
Relative Understandability (S1 vs S2) 	=   1 	if S2 > S1 
   		                                =   0	if S2 <= S1

"""
# from utils import configs_DS3 as configs
#from utils import configs
from utils import configs_DS3 as configs
import sys
sys.path.append(configs.RELATIVE_COMPREHENSIBILITY_ROOT_PATH)

## From DS3 ##
from add_code_features_DS3 import add_code_features
## From DS6 ##
#from add_code_features import add_code_features

import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import math

## read the dataset
df = pd.read_csv(configs.ROOT_PATH + "/data/final_features_ds3.csv")
#df = pd.read_csv(configs.ROOT_PATH + "/data/final_features_ds6.csv")

## unique methods 
snippet_methods = df['snippet_id'].unique()

## binary target variables
## FOR DS3 ##
binary_targets = ["readability_level"]
## FOR DS6 ##
#binary_targets = ["ABU", "PBU", "ABU50", "BD", "BD50", "AU"]

## csv
csv_data_dict = {
    "s1": "",
    "s2": "",
    "target": "",
    "s1_comprehensibility": 0,
    "s2_comprehensibility": 0,
    "(s2-s1)diff": 0,
    "(s2>s1)relative_comprehensibility": 0,
	"epsilon": 0,
	"dynamic_epsilon": False,
}


## write header
def csv_write_header(output_file, csv_data_dict):
	with open(configs.RELATIVE_COMPREHENSIBILITY_ROOT_PATH + "/data/" + output_file, "w+") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=csv_data_dict.keys())
		writer.writeheader()
	
def dict_to_csv(output_file_path, dict_data):
    with open(output_file_path, "a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dict_data.keys())
        writer.writerow(dict_data)

def create_boxplot_for_diff(df):
	## draw box plot for (s2-s1)diff per target
	df.boxplot(column=['(s2-s1)diff'], by='target', grid=False)
	plt.savefig(configs.RELATIVE_COMPREHENSIBILITY_ROOT_PATH + "/data/relative_comprehensibility_diff_boxplot_ds3.png")


def generate_ds(output_file, epsilon, dynamic_epsilon):
	for index, s1 in enumerate(snippet_methods):
			for target in binary_targets:
				method_df_s1 = df[df['snippet_id'] == s1]
				unique_target_value_counts = {0:0, 1:0}
				## unique target values
				target_values = method_df_s1[target].unique()
				for value in target_values:
					unique_target_value_counts[value] = (method_df_s1[method_df_s1[target] == value].shape[0])
					
				if len(unique_target_value_counts) == 2:
					s1_comprehensibility = unique_target_value_counts[1] / (unique_target_value_counts[0] + unique_target_value_counts[1]) 

					s1_denominator = unique_target_value_counts[0] + unique_target_value_counts[1]
				else:
					sum_values = 0
					for key in unique_target_value_counts:
						sum_values += unique_target_value_counts[key]*key
					s1_comprehensibility = sum_values / sum(unique_target_value_counts.values())
					s1_denominator = sum(unique_target_value_counts.values())
					
				for s2 in np.delete(snippet_methods, index):
					method_df_s2 = df[df['snippet_id'] == s2]

					unique_target_value_counts = {0:0, 1:0}
					## unique target values
					target_values = method_df_s2[target].unique()
					for value in target_values:
						unique_target_value_counts[value] = (method_df_s2[method_df_s2[target] == value].shape[0])
						
					if len(unique_target_value_counts) == 2:
						s2_comprehensibility = unique_target_value_counts[1] / (unique_target_value_counts[0] + unique_target_value_counts[1])
						s2_denominator = unique_target_value_counts[0] + unique_target_value_counts[1]
					else:
						sum_values = 0
						for key in unique_target_value_counts:
							sum_values += unique_target_value_counts[key]*key
						s2_comprehensibility = sum_values / sum(unique_target_value_counts.values())
						s2_denominator = sum(unique_target_value_counts.values())
					


					diff = s2_comprehensibility - s1_comprehensibility
					e = epsilon ## this is the threshold for the difference

					## dynamic epsilon
					if dynamic_epsilon:
						max_denominator = max(s1_denominator, s2_denominator)
						e = 1 / max_denominator


					############################################
					#### compute relative comprehensibility ####
					############################################
					# 0 = S1 is more comprensible than S2 #
					# 1 = S2 is more comprensible than S1 #
					# 2 = S1 and S2 has same comprehensibility #

					# 0 = S1 is more comprensible than S2
					# 1 = S2 is more comprensible than S1
					# 2 = S1 and S2 has same comprehensibility

					# Relative_Comprehensibility (S2>S1) 		=  0	if S1 - ε > S2
					# 											=  1	if S2 - ε > S1
					# 											=  2    if | S2  - S1 | <= ε

					relative_comprehensibility = 0
					if target == "PBU" or target == "ABU" or target == "ABU50" or target == "readability_level"  or target == "AU":
						if s1_comprehensibility - e > s2_comprehensibility:
							relative_comprehensibility = 0
						elif s2_comprehensibility - e > s1_comprehensibility:
							relative_comprehensibility = 1
						elif abs(s2_comprehensibility-s1_comprehensibility) <= e:	
							relative_comprehensibility = 2
					if target == "BD" or target == "BD50":
						# 0 = S1 is more comprensible than S2
						# 1 = S2 is more comprensible than S1
						# 2 = S1 and S2 has same comprehensibility

						# Relative_Comprehensibility (S2 > S1) 	=  0	if S2 - ε > S1
						# 										=  1	if S1 - ε > S2
						# 										=  2    if | S2  - S1 | <= ε
						if s2_comprehensibility - e > s1_comprehensibility:
							relative_comprehensibility = 0
						elif s1_comprehensibility - e > s2_comprehensibility:
							relative_comprehensibility = 1
						elif abs(s2_comprehensibility - s1_comprehensibility) <= e:
							relative_comprehensibility = 2		

					

					# relative_comprehensibility = 0
					# ## if target == BD or BD50, relative_comprehensibility = 1
					# if target == "BD" or target == "BD50":
					# 	relative_comprehensibility = 1

					# if s2_comprehensibility - s1_comprehensibility > e:
					# 	relative_comprehensibility = 1
					# 	## if target == BD or BD50, relative_comprehensibility = 0
					# 	if target == "BD" or target == "BD50":
					# 		relative_comprehensibility = 0
			
					csv_data_dict["s1"] = s1
					csv_data_dict["s2"] = s2
					csv_data_dict["target"] = target
					csv_data_dict["s1_comprehensibility"] = s1_comprehensibility
					csv_data_dict["s2_comprehensibility"] = s2_comprehensibility
					csv_data_dict["(s2-s1)diff"] = diff
					csv_data_dict["(s2>s1)relative_comprehensibility"] = relative_comprehensibility
					csv_data_dict["epsilon"] = e
					csv_data_dict["dynamic_epsilon"] = dynamic_epsilon
					

					dict_to_csv(configs.RELATIVE_COMPREHENSIBILITY_ROOT_PATH + "/data/" + output_file, csv_data_dict)
	
	## add (s1, s1) pairs and relative comprehensibility to the dataset
	for index, s1 in enumerate(snippet_methods):
		for target in binary_targets:
			method_df_s1 = df[df['snippet_id'] == s1]
			unique_target_value_counts = {0:0, 1:0}
			## unique target values
			target_values = method_df_s1[target].unique()
			for value in target_values:
				unique_target_value_counts[value] = (method_df_s1[method_df_s1[target] == value].shape[0])
					
			if len(unique_target_value_counts) == 2:
				s1_comprehensibility = unique_target_value_counts[1] / (unique_target_value_counts[0] + unique_target_value_counts[1]) 
				s2_comprehensibility = s1_comprehensibility
				s1_denominator = unique_target_value_counts[0] + unique_target_value_counts[1]
				s2_denominator = s1_denominator
			else:
				sum_values = 0
				for key in unique_target_value_counts:
					sum_values += unique_target_value_counts[key]*key
				s1_comprehensibility = sum_values / sum(unique_target_value_counts.values())
				s2_comprehensibility = s1_comprehensibility
				s1_denominator = sum(unique_target_value_counts.values())
				s2_denominator  = s1_comprehensibility

			diff = s2_comprehensibility - s1_comprehensibility
			e = epsilon ## this is the threshold for the difference

			## dynamic epsilon
			if dynamic_epsilon:
				max_denominator = max(s1_denominator, s2_denominator)
				e = 1 / max_denominator

			if abs(s2_comprehensibility - s1_comprehensibility) <= e:
				relative_comprehensibility = 2

			csv_data_dict["s1"] = s1
			csv_data_dict["s2"] = s1
			csv_data_dict["target"] = target
			csv_data_dict["s1_comprehensibility"] = s1_comprehensibility
			csv_data_dict["s2_comprehensibility"] = s2_comprehensibility
			csv_data_dict["(s2-s1)diff"] = diff
			csv_data_dict["(s2>s1)relative_comprehensibility"] = relative_comprehensibility
			csv_data_dict["epsilon"] = e
			csv_data_dict["dynamic_epsilon"] = dynamic_epsilon
			

			dict_to_csv(configs.RELATIVE_COMPREHENSIBILITY_ROOT_PATH + "/data/" + output_file, csv_data_dict)			

	# df1 = pd.read_csv(configs.ROOT_PATH + "/NewExperiments/results/epsilon" + "_" + str(epsilon) + "_" + output_file)
	# ## draw box plot for (s2-s1)diff per target
	# create_boxplot_for_diff(df1)



def main():

	output_file = "DS3_relative_comprehensibility_e_0_static.csv"
	csv_write_header(output_file, csv_data_dict)
	generate_ds(output_file, 0, False)

	output_file = "DS3_relative_comprehensibility_e_dynamic.csv"
	csv_write_header(output_file, csv_data_dict)
	generate_ds(output_file, 0, True)

	## run the add_code_features.py
	add_code_features("DS3_relative_comprehensibility_e_dynamic.csv", "DS3_train_date_with_warnings_e_dynamic.csv")
	add_code_features("DS3_relative_comprehensibility_e_0_static.csv", "DS3_train_date_with_warnings_e_0_static.csv")

	## read the train data and merge to single file
	train_df_static = pd.read_csv(configs.RELATIVE_COMPREHENSIBILITY_ROOT_PATH + "/data/DS3_train_date_with_warnings_e_dynamic.csv")
	train_df_dynamic = pd.read_csv(configs.RELATIVE_COMPREHENSIBILITY_ROOT_PATH + "/data/DS3_train_date_with_warnings_e_0_static.csv")
	complete_df = pd.concat([train_df_static, train_df_dynamic], axis=0)
	complete_df.to_csv(configs.RELATIVE_COMPREHENSIBILITY_ROOT_PATH + "/data/merged_ds3.csv", index=False)
	
if __name__ == "__main__":
	main()