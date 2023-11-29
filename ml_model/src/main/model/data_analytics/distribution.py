import pandas as pd
import numpy as np
import math
import csv

import matplotlib.pyplot as plt

df = pd.read_csv("ml_model/src/main/model/data/understandability_with_warnings.csv")

feature_list=['PE gen', 'PE spec (java)', 'Cyclomatic complexity',
              	'IMSQ (min)','MIDQ (min)','AEDQ (min)','EAP (min)','IMSQ (avg)',
                'MIDQ (avg)','AEDQ (avg)','EAP (avg)','IMSQ (max)','MIDQ (max)','AEDQ (max)',	
                'EAP (max)','Readability', 'ITID (avg)','NMI (avg)','NMI (max)','CIC (avg)','CIC (max)',
                'CICsyn (avg)', 'CICsyn (max)', 'CR', 'NM (avg)', 'NM (max)', 'TC (avg)','TC (min)',
                'TC (max)','#assignments (avg)','#blank lines (avg)','#commas (avg)','#comments (avg)',
                '#comparisons (avg)', 'Identifiers length (avg)', '#conditionals (avg)','Indentation length (avg)',
                '#keywords (avg)','Line length (avg)', '#loops (avg)', '#identifiers (avg)', '#numbers (avg)',
                '#operators (avg)', '#parenthesis (avg)', '#periods (avg)','#spaces (avg)','Identifiers length (max)',
                'Indentation length (max)', '#keywords (max)', 'Line length (max)',	'#identifiers (max)','#numbers (max)',
                '#characters (max)','#words (max)','Entropy','Volume','LOC','#assignments (dft)','#commas (dft)',
                '#comments (dft)', '#comparisons (dft)','#conditionals (dft)','Indentation length (dft)',
                '#keywords (dft)','Line length (dft)','#loops (dft)','#identifiers (dft)','#numbers (dft)',
                '#operators (dft)','#parenthesis (dft)','#periods (dft)','#spaces (dft)','Comments (Visual X)',
                'Comments (Visual Y)','Identifiers (Visual X)','Identifiers (Visual Y)','Keywords (Visual X)',
                'Keywords (Visual Y)','Numbers (Visual X)','Numbers (Visual Y)','Strings (Visual X)',
                'Strings (Visual Y)','Literals (Visual X)','Literals (Visual Y)','Operators (Visual X)','Operators (Visual Y)',
                'Comments (area)','Identifiers (area)','Keywords (area)','Numbers (area)','Strings (area)','Literals (area)',
                'Operators (area)','Identifiers/comments (area)','Keywords/comments (area)','Numbers/comments (area)',	
                'Strings/comments (area)','Literals/comments (area)','Operators/comments (area)','Keywords/identifiers (area)',
                'Numbers/identifiers (area)','Strings/identifiers (area)','Literals/identifiers (area)','Operators/literals (area)',
                'Numbers/keywords (area)','Strings/keywords (area)','Literals/keywords (area)','Operators/keywords (area)',
                'Strings/numbers (area)','Literals/numbers (area)','Operators/numbers (area)','Literals/strings (area)',
                'Operators/strings (area)','Operators/literals (area)',	'#aligned blocks','Extent of aligned blocks',
                '#nested blocks (avg)','#parameters','#statements','PBU','TNPU','AU','TAU','ABU50','BD50']

dict_data = {'feature':'', 'mean': 0, 'std': 0, 'min': 0, '25%': 0, '50%': 0, '75%': 0, 'max': 0}

with open('ml_model/src/main/model/data_analytics/distribution.csv', 'w') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=dict_data.keys()) 
    writer.writeheader()

for feature in feature_list:
    full_dataset = df.dropna(subset=[feature])
    fea = full_dataset[feature]

    dict_data['feature'] = feature
    dict_data['mean'] = fea.describe()['mean']
    dict_data['std'] = fea.describe()['std']
    dict_data['min'] = fea.describe()['min']
    dict_data['25%'] = fea.describe()['25%']
    dict_data['50%'] = fea.describe()['50%']
    dict_data['75%'] = fea.describe()['75%']
    dict_data['max'] = fea.describe()['max']

    with open('ml_model/src/main/model/data_analytics/distribution.csv', 'a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dict_data.keys()) 
        writer.writerow(dict_data)
    
    ## use rule of thumb to determine the number of bins
    bins = math.ceil(math.sqrt(len(fea)))

    ## plot the histogram
    plt.hist(fea, bins=bins)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title("Histogram of " + feature)

    ## remove white spaces and / from the feature name
    feature = feature.replace(" ", "")
    feature = feature.replace("/", "_")

    plt.savefig("ml_model/src/main/model/data_analytics/histograms/" + feature + ".png")
    plt.clf()