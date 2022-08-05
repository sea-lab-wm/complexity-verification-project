import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_warning_data(tools):
    warning_data = {}

    for tool in tools:
        warning_data[tool] = pd.read_csv(f"data/{tool}_data.csv")

    return warning_data


def get_warning_data_by_type(data, warning_types):

    # determine the dataset
    warning_data_by_type = pd.DataFrame()
    warning_data_by_type["dataset"] = data["dataset"]

    # ----------------

    # for each warning type and its columns
    for w_type, w_columns in warning_types.items():
        # s um the # of warnings
        warning_data_by_type[w_type] = data[list(w_columns)].sum(axis=1)

    return warning_data_by_type


def get_warnings(data):
    warnings = list(data.columns)
    warnings = list(set(warnings) - set(["dataset", "Snippet"]))
    warnings.sort()
    return warnings


def process_typestate_checker_data(data):
    warnings = get_warnings(data)

    warning_types = {}

    # warning patterns
    patterns = {
        "prefixes": ["Cannot access ", "Cannot assign because ", "Cannot assign: cannot cast", "Cannot call ",
                     "Incompatible parameter: cannot cast from", "Incompatible return value: cannot cast from", "Unsafe cast"],
        "substrings": ["did not complete its protocol"]
    }

    # loop through all the warnings
    for warning in warnings:

        matched_warning_types = []

        # process prefixes first
        for prefix in patterns["prefixes"]:
            if warning.startswith(prefix):
                matched_warning_types.append(prefix)

        # process substrings
        for substring in patterns["substrings"]:
            if substring in warning:
                matched_warning_types.append(substring)

        # ------------------

        # checks for no or multiple warning types
        if not matched_warning_types:
            raise Exception(f"No warning type found for {warning}")

        if len(matched_warning_types) > 1:
            raise Exception(
                f"Multiple warning types found for {warning}: {warning_types}")

        # -------------------

        # select warning type
        warning_type = matched_warning_types[0]

        # create an empty set of columns/warning for this type
        if warning_type not in warning_types:
            warning_types[warning_type] = set()

        # add the warning/column to the warning type
        warning_types[warning_type].add(warning)

    # --------------

    return get_warning_data_by_type(data, warning_types)


def process_checker_framework_data(data):

    warnings = get_warnings(data)
    warning_types = {}

    # loop through all the warnings
    for warning in warnings:

        # prefix [...] is the  warning type
        warning_type = warning[:warning.index(']') + 1]

        # create an empty set of columns/warning for this type
        if warning_type not in warning_types:
            warning_types[warning_type] = set()

        # add the warning/column to the warning type
        warning_types[warning_type].add(warning)

    # --------------

    return get_warning_data_by_type(data, warning_types)


def process_infer_data(data):
    warnings = get_warnings(data)
    warning_types = {}

    # loop through all the warnings
    for warning in warnings:

        warning_type = warning

        # create an empty set of columns/warning for this type
        if warning_type not in warning_types:
            warning_types[warning_type] = set()

        # add the warning/column to the warning type
        warning_types[warning_type].add(warning)

    # --------------

    return get_warning_data_by_type(data, warning_types)


def print_warning_distribution_by_wtype_dataset(data, tool_name):
    # sum by dataset
    grouped_data = data.groupby("dataset").sum()

    # plot the distribution and save it to a file
    plot = grouped_data.plot(kind='bar', stacked=True, title='# of warnings by type', colormap="tab20"
                             # , edgecolor = "black"
                             ).legend(
        loc='center left', bbox_to_anchor=(1.0, 0.4))
    fig = plot.get_figure()
    fig.tight_layout()
    fig.savefig(f"statistics/{tool_name}_distro_warning_types.png", dpi=100)


def calculate_snippet_distribution_by_dataset(data, tools):

    #df that will store the # of snippets for which each tool produces a warning
    distro = pd.DataFrame(columns=["dataset"])

    # ------------
    #get all tools
    
    tool_names = list(tools.keys())
    tool_names.sort()

    # ------------
    #get all the datasets and sort them alphabetically

    datasets = set()
    for tool in tool_names:
        datasets.update(list(pd.unique(data[tool]["dataset"])))

    datasets = list(datasets)
    datasets.sort()
    distro["dataset"] = datasets

    # --------------

    # unique_snippets = pd.DataFrame(columns=["dataset", "Snippet"])

    #for each tool, count the snippets by dataset and add these to the distro DF
    for tool in tool_names:
        tool_data = data[tool]
        counts_by_dataset = tool_data.groupby("dataset")["Snippet"].count()
        # unique_snippets = pd.concat([unique_snippets, tool_data.loc[:, [
                                    # "dataset", "Snippet"]]], ignore_index=True, sort=False)

        #do a left join on the dataset
        distro = distro.merge(counts_by_dataset, on='dataset', how='left')
        distro.rename(columns={"Snippet": tool}, inplace=True)

    # -----------------

    # print(unique_snippets)

    # unique_snippets.drop_duplicates(inplace=True)

    # print(unique_snippets)
    # count_unique_snippets = unique_snippets.groupby("dataset")[
    #     "Snippet"].count()
    # distro = distro.merge(count_unique_snippets, on='dataset', how='left')
    # distro.rename(columns={"Snippet": "any_tool"}, inplace=True)

    #
    #
    #distro['no_tool'] = distro.sum(axis=1)

    # obtain the total number of snippets and compute the number of snippets that didn't have any wanings by any tool
    correlation_data = pd.read_excel(
        "meta-analysis/correlation_analysis_for_meta_analysis.xlsx", sheet_name="all_tools")
    correlation_data = correlation_data.loc[:, [
        "dataset_id", "num_snippets_judged", "num_snippets_warnings"]].drop_duplicates()
    correlation_data["no_tool"] = correlation_data["num_snippets_judged"] - \
        correlation_data["num_snippets_warnings"]

    # map between datasets and their ids
    dataset_ids = pd.DataFrame({
                               "dataset_id": [1,
                                              2,
                                              3,
                                              6,
                                              9,
                                              "f"],
                               "dataset": ["COG Dataset 1",
                                           "COG Dataset 2",
                                           "COG Dataset 3",
                                           "COG Dataset 6",
                                           "COG Dataset 9",
                                           "fMRI Dataset"]
                               })

    #left join by dataset_id
    correlation_data = correlation_data.merge(
        dataset_ids, on='dataset_id', how='left')

    #add the no_tool count to the distro DF
    distro = distro.merge(correlation_data.loc[:,["dataset", "no_tool"]], on='dataset', how='left')
    #print(distro)


    #print(distro2)
    #print(a)
    #print(a.to_numpy())
    #print (distro2/a.to_numpy())
    
    # ------------
    # plot the distribution and save it to a file
    
    plot = distro.plot(kind='bar', x='dataset',  title='# of snippets by tool'
    #, colormap="tab20"
                       # , edgecolor = "black"
                       ).legend(
        loc='center left', bbox_to_anchor=(1.0, 0.4))
    fig = plot.get_figure()
    fig.tight_layout()
    fig.savefig(f"statistics/distro_snippets.png", dpi=100)

    #---------------------

    distro_percentage = distro.loc[:,tool_names + ["no_tool"]] / correlation_data.loc[:, ["num_snippets_judged"]].to_numpy()
    distro_percentage["dataset"] = datasets
    
    plot = distro_percentage.plot(kind='bar', x='dataset', title='% of snippets by tool', ylim=(0,1)
    #, colormap="tab20"
                       # , edgecolor = "black"
                       ).legend(
        loc='center left', bbox_to_anchor=(1.0, 0.4))
    fig = plot.get_figure()
    fig.tight_layout()
    fig.savefig(f"statistics/distro_snippets_percentage.png", dpi=100)

# -------------------
if __name__ == "__main__":

    tools = {"checker_framework":  process_checker_framework_data,
             "typestate_checker": process_typestate_checker_data,
             "infer": process_infer_data}

    # load the data
    warning_data = read_warning_data(tools.keys())

    # -----------------

    # create output folder
    os.makedirs("statistics", exist_ok=True)

    # -----------------

    # for each tool, print the distribution of warnings by type and dataset
    for tool, tool_function in tools.items():
        # add the dataset column
        tool_warning_data = warning_data[tool]
        tool_warning_data["dataset"] = list(
            map(lambda header: header[:header.index('-')-1], tool_warning_data.iloc[:, 0]))
        tool_data = tool_function(warning_data[tool])
        print_warning_distribution_by_wtype_dataset(tool_data, tool)

    # -----------------
    print()
    calculate_snippet_distribution_by_dataset(warning_data, tools)
