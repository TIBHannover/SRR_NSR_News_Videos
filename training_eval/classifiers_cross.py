import argparse
import numpy as np
import pandas as pd
import random
import yaml
import warnings
from train_utils import *
from analysis_utils import *
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

warnings.filterwarnings("ignore", category=UserWarning)

def train(train_x, train_y, val_x, val_y, test_x, test_y, model, param_grid, args):
    train_x, val_x, test_x = np.array(train_x), np.array(val_x), np.array(test_x)

    if args.test == "bild" and args.nsr: ## To handle missing label for Bild
        train_x = np.concatenate((train_x, np.zeros((1, train_x.shape[1]))), axis=0)
        train_y = np.concatenate((train_y, [3]), axis=0)

    print("\tTrain X shape:", train_x.shape, ", Val X shape:", val_x.shape, ", Test X shape:", test_x.shape)

    X = np.concatenate((train_x, val_x), axis=0)
    Y = np.concatenate((train_y, val_y), axis=0)
    split_index = [-1] * len(train_x) + [0] * len(val_x)
    ps = PredefinedSplit(test_fold=split_index)

    grid_search = GridSearchCV(model, param_grid, cv=ps, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, Y)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(test_x)

    test_f1_macro = precision_recall_fscore_support(test_y, y_pred, average='macro')[2]*100
    test_acc = accuracy_score(test_y, y_pred)*100

    print(f"\tTesting domain: {args.test}")
    print(f"\tTest F1-macro: {test_f1_macro:.2f}, Test Accuracy: {test_acc:.2f}")
    print(f"\tBest parameters: {best_params}")

    return best_model, best_params, test_f1_macro, test_acc


def test(best_model, test_x, test_y):
    y_pred = best_model.predict(test_x)

    test_f1_macro = precision_recall_fscore_support(test_y, y_pred, average='macro')[2]*100
    test_acc = accuracy_score(test_y, y_pred)*100

    print(f"\tTest F1-macro: {test_f1_macro:.2f}, Test Accuracy: {test_acc:.2f}")

    return test_f1_macro, test_acc


def get_feature_importances(best_model):
    if hasattr(best_model, 'feature_importances_'):
        # This works for models like RandomForestClassifier
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'get_booster'):
        # This works for XGBoost models
        importances = best_model.get_booster().get_score(importance_type='weight')
        importances = np.array([importances.get(f, 0.) for f in best_model.get_booster().feature_names])
    else:
        raise ValueError("Model does not have feature importances")
    
    return importances

def main():
    parser = argparse.ArgumentParser(description='Training and evaluation of classifiers for K-fold splits')
    parser.add_argument('--rf', action='store_true', help="Random Forest")
    parser.add_argument('--xgb', action='store_true', help="XGBoost")

    parser.add_argument('--srr', action='store_true', help="Speaker Role Recognition")
    parser.add_argument('--nsr', action='store_true', help="News Situation Recognition")
    parser.add_argument('--seg', action='store_true', help="Training data based on speaker segments")
    parser.add_argument('--sw', action='store_true', help="Training data based on sliding windows")

    parser.add_argument('--hierarchy', action='store', type=int, choices=[0, 1], default=0, help="Speaker mapping hierarchy level")
    parser.add_argument('--test', default="tag", choices=['tag', 'bild', 'com'], help="Split dataset by news sources")
    parser.add_argument('--eval_only', action='store_true', help="Only Evaluate the saved model")
    parser.add_argument('--confusion', action='store_true', help="Plot confusion matrix")
    parser.add_argument('--importance', action='store_true', help="Plot feature importance")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    ## Fixed stuff
    task = "srr" if args.srr else "nsr"
    data_feature_dir = config["speaker_feature_dir"] if task=="srr" else config["situation_feature_dir"]
    data_split_dir = config["speaker_split_dir"] if task=="srr" else config["situation_split_dir"]
    aggregation = "windowbased" if args.sw else "segmentbased"
    model_type = "rf" if args.rf else "xgb"
    ncls = len(config["labels_%d"%(args.hierarchy)]) if task=="srr" else len(config["labels_situations"])

    print("\n----- Task: %s, Aggregation: %s, Hierarchy: %d, Evaluation: Cross-Domain, Number of Classes: %d -----\n"%(task, aggregation, args.hierarchy, ncls))

    ## Load splits
    if task == "srr":
        train_x, train_y, val_x, val_y, test_x, test_y = load_speaker_cross_splits(data_feature_dir, data_split_dir, aggregation, args.hierarchy, config, args.test)
    else:
        train_x, train_y, val_x, val_y, test_x, test_y = load_siutations_cross_splits(data_feature_dir, data_split_dir, aggregation, config, args.test)
    
    ## Model
    model_dir = f"models/{task}/{model_type}/cross/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if task == "srr":
        model_name = os.path.join(model_dir, f"{aggregation}_l{args.hierarchy}_{args.test}.pkl")
    else:
        model_name = os.path.join(model_dir, f"{aggregation}_{args.test}.pkl")
    
    if not args.eval_only:
        ## Training and Evaluation
        model, param_grid = initialize_model(model_type, "cross")

        best_model, best_params, test_f1_macro, test_acc = train(train_x, train_y, val_x, val_y, test_x, test_y, model, param_grid, args)

        ## Save the model
        model_dict = {"model": best_model, "params": best_params, "test_f1_macro": test_f1_macro, "test_acc": test_acc}
        with open(model_name, 'wb') as model_file:
            pickle.dump(model_dict, model_file)
            print(f"\tModel saved at: {model_name}")

        if args.importance and args.seg: ## Only for segment based data
            ## Feature Importances
            importance_vector = get_feature_importances(best_model)
            ## Plot
            feature_names = config[f'feature_names_speaker'] if task=="srr" else config[f'feature_names_situations']
            plot_feature_importances(importance_vector, feature_names, model_type, task, aggregation, "cross", args)
        if args.confusion:
            ## Confusion Matrix
            conf_matrix = confusion_matrix(test_y, best_model.predict(test_x))
            plot_confusion_matrices(conf_matrix, model_type, task, aggregation, "cross", args, ncls)
    else:
        ## Testing
        if not os.path.exists(model_name):
            print(f"\tModel not found: {model_name}, please train the model first !!")
            return 0
        
        with open(model_name, 'rb') as model_file:
            model_dict = pickle.load(model_file)
        best_model = model_dict["model"]
        test(best_model, test_x, test_y)


if __name__ == "__main__":
    main()