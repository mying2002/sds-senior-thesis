import numpy as np
import pandas as pd
import os

import my_distance_metrics as my_dist


# ---------- helper functions to process data ---------- #


# ----- reads in data and concatenates dataframes ----- #
# data_directory: str for directory relative to current file
# set_name: name used for new files to save
# save_directory: directory to save processed files
# file_filter_condition: helps extract relevant files from data_directory
def extract_data(data_directory, set_name=None, save_directory=None, file_filter_condition=None, keep_columns=None, column_filters=None):
    # ----- get list of file names ----- #
    directory = data_directory
    directory_contents = os.listdir(directory)

    # ----- filter for data files ----- #
    if file_filter_condition != None:
        file_names = [file for file in directory_contents if file_filter_condition(file)]
    else:
        # default
        # file_names = directory_contents
        file_names = [file for file in directory_contents if "clone-pass.tsv" in file] # avoid .DS_Store and other IGNORE files

    # ----- read in and process data into master pd dataframe ----- #
    full_data = pd.DataFrame()

    for file in file_names:
        file_data = pd.read_csv(directory + '/' + file, sep = '\t')

        # select columns - WARNING: may give errors for new data sets with different column names; can pass into func later
        if keep_columns is not None:
            file_data = file_data[keep_columns]
        # file_data = file_data[["sequence_id", "sample_id", "subject_id", "clone_id", "v_call", "j_call", "junction_aa", "junction_length", "locus"]] # manual

        # print(np.unique(file_data["study_group_description"])) # all control data points have "Control" in the name

        # filter based on a column + specified value
        if column_filters is not None:
            for col, value, type in column_filters: # str_contains
                assert col in keep_columns

                if type == "str_contains":
                    file_data = file_data[file_data[col].str.contains(value)]

        # concatenate to full data DF
        full_data = pd.concat([full_data, file_data])


    # ----- save file if argument is provided ----- #
    if save_directory != None:
        if set_name != None:
            new_file_name = save_directory + '/full_' + set_name + "_data.csv"
        else:
            new_file_name = save_directory + '/full_data.csv'

        full_data.to_csv(new_file_name, index=False)
    

    return full_data


# ----- processes full data ----- #
# data_filter_tuple_list: list of tuples of (col_name, value) to filter data by. 
def process_data(full_data, set_name=None, save_directory=None, data_filter_tuple_list=None):
    # ----- TO IMPLEMENT IF NEEDED: general filter data ----- #
    if data_filter_tuple_list != None:
        for (col_name, values) in data_filter_tuple_list:
            # TO IMPLEMENT IF NEEDED
            pass

    print(f"raw size: {full_data.shape}")

    # temp manual
    # - filter junction length; locus (keep heavy chains only)
    filtered_data = full_data.loc[
                                #   (full_data["junction_length"] == target_junction_length) & 
                                (full_data["locus"] == "IGH") &
                                (full_data["v_call"].str.contains("IGHV1-58"))
                                ].copy()
    
    print(f"shape after filter IGH locus and IGHV1-58 v-call: {filtered_data.shape}")

    # - v_call normalize
    # NOTE: currently using all IGHV1-58; in the future, can change this to be similar to j_call methodology
    filtered_data["v_call"] = "IGHV1-58"

    # - j_call normalize
    # TO IMPLEMENT: can consider copying the row, once for each different j_call, when they can't identify. for now, just grab first.
    # filtered_data["j_call"] = filtered_data["j_call"].str.split(",").str[0]
    filtered_data["j_call"] = filtered_data["j_call"].str.split("*").str[0]

    # - CDHR3 without start/end string
    filtered_data["junction_aa_clipped"] = filtered_data["junction_aa"].str.slice(1,-1) # for matching with annotated covid data

    # - remove duplicates, if any
    # filtered_data = filtered_data.drop_duplicates(["subject_id",  "junction_aa"]) 
    filtered_data = filtered_data.drop_duplicates(["subject_id", "junction_aa_clipped"]) 

    print(f"shape after removing (subject, junction_aa_clipped) dupes: {filtered_data.shape}")


    # - remove rows from subjects with < 10 sequences
    filtered_data = filtered_data.groupby("subject_id").filter(lambda x: len(x) > 10)

    print(f"shape after removing subjects with <10 (unique) sequences: {filtered_data.shape}")


    # - select certain list of subjects, etc.
    # filter_list = []

    # - reset index
    filtered_data = filtered_data.reset_index(drop=True)

    # - node labels [SHOULD MATCH INDEX]
    filtered_data["node_label"] = list(range(0,filtered_data.shape[0],1))

    # ----- save file if argument is provided ----- #
    if save_directory != None:
        if set_name != None:
            new_file_name = save_directory + '/filtered_' + set_name + "_data.csv"
        else:
            new_file_name = save_directory + '/filtered_data.csv'

        filtered_data.to_csv(new_file_name, index=False)

    # ----- return object(s) ----- #
    return filtered_data



# returns a set of annotated covid data points
# need custom since data format is different (public)
def process_annotated_data(file_path):
    # ----- read in data ----- #
    # https://opig.stats.ox.ac.uk/webapps/covabdab/
    cov_ab_data = pd.read_csv(file_path)

    # ----- introduce new columns ----- #
    # cov_ab_data['heavy_v_gene'] = cov_ab_data['Heavy V Gene'].str.split(" ").str[0]
    # cov_ab_data['heavy_j_gene'] = cov_ab_data['Heavy J Gene'].str.split(" ").str[0]
    # cov_ab_data['light_v_gene'] = cov_ab_data['Light V Gene'].str.split(" ").str[0]
    # cov_ab_data['light_j_gene'] = cov_ab_data['Light J Gene'].str.split(" ").str[0]

    cov_ab_data['Heavy V Gene'] = cov_ab_data['Heavy V Gene'].str.split(" ").str[0]
    cov_ab_data['Heavy J Gene'] = cov_ab_data['Heavy J Gene'].str.split(" ").str[0]
    cov_ab_data['Light V Gene'] = cov_ab_data['Light V Gene'].str.split(" ").str[0]
    cov_ab_data['Light J Gene'] = cov_ab_data['Light J Gene'].str.split(" ").str[0]

    cov_ab_data['CDRH3_length'] = cov_ab_data['CDRH3'].str.len()

    # ----- distill data set ----- #
    # select relevant columns
    filtered_cov_data = cov_ab_data[['Name', 'Ab or Nb', 'Heavy V Gene', 'Heavy J Gene', 'Light V Gene', 'Light J Gene', 'CDRH3', 'CDRH3_length']]

    # filtering
    filtered_cov_data = filtered_cov_data.loc[(filtered_cov_data['Ab or Nb'] == 'Ab') &                 # antibody or nanobody?
                                            # (filtered_cov_data['Heavy V Gene'] == "IGHV1-58") #&       # V gene to match our data set
                                            (filtered_cov_data['Heavy V Gene'].str.contains("IGHV1-58"))
                                            # (filtered_cov_data['CDRH3_length'] == 16.0)               # length = 16 (doesn't include conserved start/end AA)
                                            ]

    # delete duplicates?
    # filtered_cov_data = filtered_cov_data.drop_duplicates(["Name", "CDRH3"]) # we don't lose any data points
    # filtered_cov_data = filtered_cov_data.drop_duplicates(["CDRH3"]) # 117 --> 99, implies different patients had matching sequences
    filtered_cov_data = filtered_cov_data.drop_duplicates(['Heavy V Gene', 'Heavy J Gene', 'CDRH3']) # 100 sequences

    # set of sequences
    # cov_CDRH3 = set(filtered_cov_data["CDRH3"]) # unused for now

    return filtered_cov_data





