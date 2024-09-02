import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

situation_label_map =   {0: "talking-head", 1: "voiceover", 2: "interview", 3: "commenting", 4: "speech"}
speaker_label_map = {0: "anchor", 1: "reporter", 2: "other", 3: "expert", 4: "layperson", 5: "politician"}


def plot_feature_importances(feature_importances, feature_names, model_type, 
                             task, aggregation, evaluation, args):
    avg_feature_importances = np.mean(feature_importances, axis=0) if len(feature_importances.shape) > 1 else feature_importances

    plt.figure(figsize=(26,20))
    plt.rcParams['font.size'] = 18
    if task == "srr":
        plt.title("Speaker Role Recognition - Feature Importances", fontweight='bold')
        basename = f"{aggregation}_l{args.hierarchy}_{args.test}" if evaluation == "cross" else f"{aggregation}_l{args.hierarchy}"
    else:
        plt.title("News Situation Recognition - Feature Importances", fontweight='bold')
        basename = f"{aggregation}_{args.test}" if evaluation == "cross" else f"{aggregation}"

    indices = np.argsort(avg_feature_importances)[::-1]  # Sort feature importances in descending order

    # Feature names from the configuration file
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_importances = [float(imp) for imp in avg_feature_importances[indices]]

    # Creating the horizontal bar plot
    plt.barh(range(len(avg_feature_importances)), sorted_importances, align='center')
    plt.yticks(range(len(avg_feature_importances)), sorted_feature_names)

    # X-label based on the model
    if model_type == "rf":
        plt.xlabel('Mean Decrease in Impurity')
    else:
        plt.xlabel('Relative Information Gain')

    # Invert y-axis for better readability
    plt.gca().invert_yaxis()

    # Saving the plot
    plot_dir = f"analysis/{task}/{model_type}/{evaluation}/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(plot_dir, f'feature_importances_{basename}.png')
    plt.savefig(plot_path, bbox_inches='tight')
    print(f'Feature Importances saved as png: {plot_path}')

    # Create a dictionary of feature names and their importances
    feature_importance_dict = {sorted_feature_names[i]: sorted_importances[i] for i in range(len(feature_names))}

    # Save the dictionary to a JSON file
    json_file_path = os.path.join(plot_dir, f'feature_importances_{basename}.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(feature_importance_dict, json_file, indent=4)
    print(f'Feature importances saved as JSON: {json_file_path}')


def plot_confusion_matrices(conf_matrices, model_type, task, aggregation, 
                            evaluation, args, ncls):
    avg_cm = np.mean(conf_matrices, axis=0) if len(conf_matrices.shape) > 2 else conf_matrices

    # Normalize the confusion matrix for better readability
    normalized_cm = avg_cm / avg_cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 10))

    if task == "srr":
        plt.title("Average Confusion Matrix for Speaker Role Recognition") if evaluation == "cross" else plt.title("Confusion Matrix for Speaker Role Recognition")
        class_names = [speaker_label_map[i] for i in range(ncls)]
        basename = f"{aggregation}_l{args.hierarchy}_{args.test}" if evaluation == "cross" else f"{aggregation}_l{args.hierarchy}"
    elif task == "nsr":
        plt.title("Average Confusion Matrix for News Situation Recognition") if evaluation == "cross" else plt.title("Confusion Matrix for News Situation Recognition")
        class_names = [situation_label_map[i] for i in range(ncls)]
        basename = f"{aggregation}_{args.test}" if evaluation == "cross" else f"{aggregation}"

    # Saving the plot
    plot_dir = f"analysis/{task}/{model_type}/{evaluation}/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    sns.heatmap(normalized_cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Saving the confusion matrix plot
    cm_plot_path = os.path.join(plot_dir, f'confusion_matrix_{basename}.png')
    plt.savefig(cm_plot_path, bbox_inches='tight')
    print(f'Confusion Matrix plot saved as png: {cm_plot_path}')

    # Optionally, save the average confusion matrix as a CSV or JSON
    cm_file_path = os.path.join(plot_dir, f'confusion_matrix_{basename}.csv')
    np.savetxt(cm_file_path, normalized_cm, delimiter=",")
    print(f'Confusion Matrix data saved as csv: {cm_file_path}')