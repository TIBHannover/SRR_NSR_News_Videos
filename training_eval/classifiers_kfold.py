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
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore", category=UserWarning)

def train(train_xs, train_ys, val_xs, val_ys, test_xs, test_ys, 
          model, param_grid, model_type, ncls):
    test_f1_macros, test_accs = [], []
    best_params = []
    best_models = []
    feature_importances = []
    conf_matrices = []

    for i in range(len(train_xs)):
        train_x, val_x, test_x = train_xs[i], val_xs[i], test_xs[i]
        train_y, val_y, test_y = train_ys[i], val_ys[i], test_ys[i]

        train_x, val_x, test_x = np.array(train_x), np.array(val_x), np.array(test_x)
        print("\t\tTrain X shape:", train_x.shape, ", Val X shape:", val_x.shape, ", Test X shape:", test_x.shape)
        print("\t\tTrain Y shape:", len(train_y), ", Val Y shape:", len(val_y), ", Test Y shape:", len(test_y))

        X = np.concatenate((train_x, val_x), axis=0)
        Y = np.concatenate((train_y, val_y), axis=0)
        split_index = [-1] * len(train_x) + [0] * len(val_x)
        ps = PredefinedSplit(test_fold=split_index)

        grid_search = GridSearchCV(model, param_grid, cv=ps, scoring='accuracy', n_jobs=-1)

        ## Compute weights if xgb
        if model_type == 'xgb':
            classes_weights = compute_sample_weight(
                                class_weight='balanced',
                                y=Y
                            )
            grid_search.fit(X, Y, sample_weight=classes_weights)
        else:
            grid_search.fit(X, Y)

        best_params.append(grid_search.best_params_)
        best_model = grid_search.best_estimator_
        best_models.append(best_model)

        y_pred = best_model.predict(test_x)

        test_f1_macro = precision_recall_fscore_support(test_y, y_pred, average='macro')[2]*100
        test_f1_macros.append(test_f1_macro)
        test_acc = accuracy_score(test_y, y_pred)*100
        test_accs.append(test_acc)
        print(f"\tSplit-{i+1} Test F1-macro: {test_f1_macro:.2f}, Test Accuracy: {test_acc:.2f}")

        # Get feature importances and append to list
        if hasattr(best_model, 'feature_importances_'):
            # This works for models like RandomForestClassifier
            importances = best_model.feature_importances_
        elif hasattr(best_model, 'get_booster'):
            # This works for XGBoost models
            importances = best_model.get_booster().get_score(importance_type='weight')
            importances = np.array([importances.get(f, 0.) for f in best_model.get_booster().feature_names])
        else:
            raise ValueError("Model does not have feature importances")
        
        feature_importances.append(importances)

        ## Confusion matrix
        cm = confusion_matrix(test_y, y_pred)
        if len(cm) == ncls:
            conf_matrices.append(cm)

    return best_models, best_params, test_f1_macros, test_accs, np.array(feature_importances), np.array(conf_matrices)

def test(test_xs, test_ys, best_models):
    test_f1_macros, test_accs = [], []
    for i in range(len(test_xs)):
        test_x, test_y = test_xs[i], test_ys[i]
        best_model = best_models[i]
        y_pred = best_model.predict(test_x)

        test_f1_macro = precision_recall_fscore_support(test_y, y_pred, average='macro')[2]*100
        test_f1_macros.append(test_f1_macro)
        test_acc = accuracy_score(test_y, y_pred)*100
        test_accs.append(test_acc)
        print(f"\tSplit-{i+1} Test F1-macro: {test_f1_macro:.2f}, Test Accuracy: {test_acc:.2f}")
    
    ## Average and std of 10-fold evaluation
    print("Average F1-macro: %.2f, std: %.2f"%(np.mean(test_f1_macros), np.std(test_f1_macros)))
    print("Average Accuracy: %.2f, std: %.2f"%(np.mean(test_accs), np.std(test_accs)))

    return test_f1_macros, test_accs

def main():
    parser = argparse.ArgumentParser(description='Training and evaluation of classifiers for K-fold splits')
    parser.add_argument('--rf', action='store_true', help="Random Forest")
    parser.add_argument('--xgb', action='store_true', help="XGBoost")

    parser.add_argument('--srr', action='store_true', help="Speaker Role Recognition")
    parser.add_argument('--nsr', action='store_true', help="News Situation Recognition")
    parser.add_argument('--seg', action='store_true', help="Training data based on speaker segments")
    parser.add_argument('--sw', action='store_true', help="Training data based on sliding windows")

    parser.add_argument('--hierarchy', action='store', type=int, choices=[0, 1], default=0, help="Speaker mapping hierarchy level")
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

    print("\n----- Task: %s, Aggregation: %s, Hierarchy: %d, Evaluation: K-Fold, Number of Classes: %d -----\n"%(task, aggregation, args.hierarchy, ncls))

    ## Load splits
    if task == "srr":
        train_xs, train_ys, val_xs, val_ys, test_xs, test_ys = load_speaker_kfold_splits(data_feature_dir, data_split_dir, aggregation, args.hierarchy, config)
    else:
        train_xs, train_ys, val_xs, val_ys, test_xs, test_ys = load_siutations_kfold_splits(data_feature_dir, data_split_dir, aggregation, config)
    
    ## Model
    model_dir = f"models/{task}/{model_type}/10fold/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if task == "srr":
        model_name = os.path.join(model_dir, f"{aggregation}_l{args.hierarchy}.pkl")
    else:
        model_name = os.path.join(model_dir, f"{aggregation}.pkl")

    if not args.eval_only:
        ## Training and Evaluation
        model, param_grid = initialize_model(model_type, "10fold")

        best_models, best_params, test_f1_macros, test_accs, \
            feature_importances, conf_matrices = train(train_xs, train_ys, val_xs, val_ys, 
                                                       test_xs, test_ys, model, param_grid, model_type, ncls)
        ## Average and std of 10-fold evaluation
        print("Average F1-macro: %.2f, std: %.2f"%(np.mean(test_f1_macros), np.std(test_f1_macros)))
        print("Average Accuracy: %.2f, std: %.2f"%(np.mean(test_accs), np.std(test_accs)))

        ## Save the model
        model_dict = {"models": best_models, "params": best_params, "test_f1_macros": test_f1_macros, "test_accs": test_accs}
        with open(model_name, 'wb') as file:
            pickle.dump(model_dict, file)
            print(f"\tAll split models saved to {model_name}")
    
        if args.importance and args.seg: ## Only for segment based data
            ## Plot feature importance
            feature_names = config[f'feature_names_speaker'] if task=="srr" else config[f'feature_names_situations']            
            plot_feature_importances(feature_importances, feature_names, model_type, task, aggregation, "10fold", args)
        if args.confusion:
            ## Plot confusion matrix
            plot_confusion_matrices(conf_matrices, model_type, task, aggregation, "10fold", args, ncls)
    else:
        ## Testing
        if not os.path.exists(model_name):
            print(f"\tModel not found: {model_name}, please train the model first !!")
            return 0
        with open(model_name, 'rb') as file:
            model_dict = pickle.load(file)
        best_models = model_dict["models"]
        
        test(test_xs, test_ys, best_models)


if __name__ == "__main__":
    main()