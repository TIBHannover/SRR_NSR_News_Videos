import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from train_utils import *

def main():
    parser = argparse.ArgumentParser(description='Correlation Matrix over Features and Labels')
    parser.add_argument('--srr', action='store_true', help="Speaker Role Recognition")
    parser.add_argument('--nsr', action='store_true', help="News Situation Recognition")
    args = parser.parse_args()

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    ## Fixed stuff
    task = "srr" if args.srr else "nsr"
    data_feature_dir = config["speaker_feature_dir"] if task=="srr" else config["situation_feature_dir"]
    feature_names = config["feature_names_speaker"] if task=="srr" else config["feature_names_situations"]

    _, all_data = load_pickle_data(data_feature_dir, "segmentbased", config)

    all_features = [all_data[key]["feature"] for key in all_data]
    all_labels = [all_data[key]["label_1"] for key in all_data] if task=="srr" else [all_data[key]["label"] for key in all_data]

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    features_with_labels = pd.DataFrame(np.column_stack((all_features, all_labels)), columns=feature_names + ['predicted_label'])
    label_map = config['groundtruth_numerical_speaker'] if task=="srr" else config['groundtruth_numerical_situations']

    label_feature_corrs = pd.DataFrame()
    for label_numerical in label_map.values():
        # extract entries that match the label
        df_label = features_with_labels[features_with_labels['predicted_label'] == label_numerical]
        # set all predicted_label entries to 1 (indicating that the label is existent in these samples)
        df_label.loc[:, 'predicted_label'] = 1
        # extract all other entries that were not extracted in the step before
        df_without_label = features_with_labels[~features_with_labels.index.isin(df_label.index)]
        # set all predicted_label entries to 0 (indicating that the label is not existent in these samples)
        df_without_label.loc[:, 'predicted_label'] = 0
        # merge both dataframes again, so we have the distinction between samples where the label matches vs. non-matching samples
        merged_df = pd.concat([df_label, df_without_label])
        # create correlation matrix
        corr_matrix = merged_df.corr()
        # extract only the row which represents the feature-label correlation
        label_feature_corr = corr_matrix['predicted_label']

        # label numerical as string instead
        label_string = list(label_map.keys())[label_numerical]
        # convert pandas series to dictionary
        label_feature_corr_dict = label_feature_corr.to_dict()
        # set label as string into the dictionary
        label_feature_corr_dict['label'] = label_string
        # convert the dictionary to a dataframe
        df_row = pd.DataFrame.from_records([label_feature_corr_dict])
        # add the correlation between features and labels for the current label to the dataframe
        label_feature_corrs = pd.concat([label_feature_corrs, df_row], ignore_index=True)

    label_feature_corrs.set_index('label', inplace=True)
    label_feature_corrs = label_feature_corrs.drop('predicted_label', axis=1)

    # split dataframe, else heatmap is too wide
    split_index = label_feature_corrs.columns.get_loc('NumberOfShots')
    df1 = label_feature_corrs.iloc[:, :split_index + 1]
    df2 = label_feature_corrs.iloc[:, split_index + 1:]

    # split 1 plot
    plt.figure(figsize=(36,12))
    plt.rcParams['font.size'] = 24
    cmap = sns.diverging_palette(15, 120, as_cmap=True)
    sns.heatmap(df1, cmap=cmap, linewidths=0.5, square=True, cbar=False, annot=True, fmt=".2f")
    plt.xlabel(' ')
    plt.ylabel(' ')
    plot_path = f'analysis/{task}/correlation_matrix_1'
    plt.savefig(plot_path+".png", bbox_inches='tight')
    print(f'Correlation Matrix 1: {plot_path}')

    # split 2 plot
    plt.clf()
    plt.figure(figsize=(36,12))
    sns.heatmap(df2, cmap=cmap, linewidths=0.5, square=True, cbar=False, annot=True, fmt=".2f")
    plot_path = f'analysis/{task}/correlation_matrix_2'
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.savefig(plot_path+".png", bbox_inches='tight')
    print(f'Correlation Matrix 2: {plot_path}')


if __name__ == "__main__":
    main()
