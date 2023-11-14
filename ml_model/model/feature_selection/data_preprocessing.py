import pandas as pd

ROOT_PATH = "/Users/nadeeshan/Documents/Spring2023/ML-Experiments/complexity-verification-project/ml_model/model/"
df = pd.read_csv(ROOT_PATH + 'data/understandability_with_warnings.csv')

## These are the columns where the values should be unique
filtering_columns = ['snippet_signature', 'developer_position', 'PE gen', 'PE spec (java)']


def get_duplicate_rows(df, filtering_columns):
    mask = df.duplicated(subset=filtering_columns, keep=False)
    return df[mask]


## This will return a all the duplicate rows considering snippet_signature, developer_position, PE gen, PE spec (java)
## including the overlapping rows
all_duplciates_including_overalaps = get_duplicate_rows(df, filtering_columns)

## write to csv - duplicate rows considering snippet_signature, developer_position, PE gen, PE spec (java)
# all_duplciates_including_overalaps.to_csv(ROOT_PATH + 'data/duplicate_rows.csv', index=False)


columns_to_check = ['snippet_signature', 'developer_position', 'PE gen', 'PE spec (java)', 'ABU50']
mask = df.duplicated(subset=columns_to_check, keep=False)
duplicate_rows_2 = df[mask]

## remove the duplicate rows 
# duplicate_rows_2.to_csv(ROOT_PATH + 'data/duplicate_rows_with_ABU50.csv', index=False)

#merge two DataFrames and create indicator column
df_all = all_duplciates_including_overalaps.merge(duplicate_rows_2, on=filtering_columns,
                   how='left', indicator=True,)

#create DataFrame with rows that exist in first DataFrame only. Use only the original column names
df1_only = df_all[df_all['_merge']=='left_only']

## remove _y columns
df1_only = df1_only[df1_only.columns.drop(list(df1_only.filter(regex='_y')))]

## rename all the columns ends with _x without _x
df1_only = df1_only.rename(columns=lambda x: x.replace('_x', ''))
## remove _merge column
df1_only = df1_only.drop(columns=['_merge'])


#write DataFrame to csv
# df1_only.to_csv(ROOT_PATH + 'data/duplicate_rows_overlapping.csv', index=False)


df_non_overlapping = df.merge(df1_only, on=filtering_columns, how='left', indicator=True)
df_non_overlapping = df_non_overlapping[df_non_overlapping['_merge']=='left_only']
df_non_overlapping = df_non_overlapping[df_non_overlapping.columns.drop(list(df_non_overlapping.filter(regex='_y')))]
df_non_overlapping = df_non_overlapping.rename(columns=lambda x: x.replace('_x', ''))
df_non_overlapping = df_non_overlapping.drop(columns=['_merge'])



## remove the duplicate rows considering columns_to_check
df_non_overlapping.drop_duplicates(subset=columns_to_check, keep=False, inplace=True)

## encode the developer_position
## if developer_position is "bachelor student" then 1
## elif developer_position is "master student" then 2
## elif developer_position is phd student" then 3
## elif developer_position is "professional developer" then 4

df_non_overlapping['developer_position'] = df_non_overlapping['developer_position'].replace(['bachelor student', 'master student', 'phd student', 'professional developer'], [1, 2, 3, 4])
df_non_overlapping.to_csv(ROOT_PATH + 'data/understandability_with_warnings_ABU50.csv', index=False)
