""" This scripts merges understandability.csv with test_warning_df.csv"""
import pandas as pd

df1 = pd.read_csv('ml_model/src/main/model/data/understandability.csv')
df2 = pd.read_csv('ml_model/src/main/model/data/test_warning_df.csv')

## merge warning columns
df2 = df2.groupby(['dataset_id','file_name']).agg({'warnings_checker_framework': 'first','warnings_typestate_checker': 'first','warnings_infer':'first','warnings_openjml':'first'}).reset_index()
## add 0 to empty cells of warnings_checker_framework, warnings_typestate_checker, warnings_infer, warnings_openjml
df2['warnings_checker_framework'] = df2['warnings_checker_framework'].fillna(0)
df2['warnings_typestate_checker'] = df2['warnings_typestate_checker'].fillna(0)
df2['warnings_infer'] = df2['warnings_infer'].fillna(0)
df2['warnings_openjml'] = df2['warnings_openjml'].fillna(0)

## add new column to df2 sum of warnings
df2['warning_sum'] = df2['warnings_checker_framework'] + df2['warnings_typestate_checker'] + df2['warnings_infer'] + df2['warnings_openjml']

df1['file_name'] = df1['file_name'].str.replace(" -- ", "-")

df3 = pd.merge(df1, df2[['file_name','warnings_checker_framework','warnings_typestate_checker','warnings_infer','warnings_openjml','warning_sum']], on=['file_name',], how='left')

df3['warnings_checker_framework'] = df3['warnings_checker_framework'].fillna(0)
df3['warnings_typestate_checker'] = df3['warnings_typestate_checker'].fillna(0)
df3['warnings_infer'] = df3['warnings_infer'].fillna(0)
df3['warnings_openjml'] = df3['warnings_openjml'].fillna(0)
df3['warning_sum'] = df3['warnings_checker_framework'] + df3['warnings_typestate_checker'] + df3['warnings_infer'] + df3['warnings_openjml']


df3.to_csv('ml_model/src/main/model/data/understandability_with_warnings.csv', index=False)

