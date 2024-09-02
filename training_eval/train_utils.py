import os
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


def initialize_model(model_type, eval_type):
    if model_type == 'rf':
        model = RandomForestClassifier(random_state=42, class_weight='balanced' if eval_type=="10fold" else None)
        param_grid = {
            'n_estimators': [25, 50, 75, 100, 110, 120, 130, 140, 150],
            'max_depth': [None, 1, 2, 4, 6],
            'min_samples_split': [2, 4, 6, 8],  # The minimum number of samples required to split an internal node
            'min_samples_leaf': [1, 2, 3, 4],  # The minimum number of samples required to be at a leaf node
        }
    elif model_type == 'xgb':
        model = XGBClassifier(random_state=42)
        param_grid = {
            'n_estimators': [25, 50, 75, 100, 110, 120, 130, 140, 150],
            'max_depth': [None, 1, 2, 4, 6],
            'learning_rate': [0.01, 0.1, 0.2],
            'gamma': [0, 0.1, 0.2],  # Minimum loss reduction required for further partition on a leaf node of the tree
        }
    else:
        raise ValueError("Invalid model type, choose between 'rf' and 'xgb'")

    print(f"\tModel initialized: {model_type}")

    return model, param_grid


def load_pickle_data(feature_dir, aggregation, config):
    source_data = {}
    for src in config["source_abbreviations"]:
        source_data[src] = pickle.load(open(os.path.join(feature_dir, "%s_features_%s.pkl"%(src, aggregation)), "rb"))

    ## Load data
    all_data = {}
    for src in config["source_abbreviations"]:
        all_data.update(source_data[src])

    return source_data, all_data

def load_speaker_kfold_splits(data_feature_dir, split_dir, aggregation, hierarchy, config):
    _, all_data = load_pickle_data(data_feature_dir, aggregation, config)

    train_splits, val_splits, test_splits = [], [], []
    for i in range(1,11):
        train_splits.append(open(os.path.join(split_dir, "level_%d"%(hierarchy), "10fold/train_%d.txt"%(i)), "r").read().splitlines())
        val_splits.append(open(os.path.join(split_dir, "level_%d"%(hierarchy), "10fold/val_%d.txt"%(i)), "r").read().splitlines())
        test_splits.append(open(os.path.join(split_dir, "level_%d"%(hierarchy), "10fold/test_%d.txt"%(i)), "r").read().splitlines())

    train_xs = [[all_data[key]["feature"] for key in train_splits[i]] for i in range(len(train_splits))]
    train_ys = [[all_data[key]["label_%d"%(hierarchy)] for key in train_splits[i]] for i in range(len(train_splits))]
    val_xs = [[all_data[key]["feature"] for key in val_splits[i]] for i in range(len(val_splits))]
    val_ys = [[all_data[key]["label_%d"%(hierarchy)] for key in val_splits[i]] for i in range(len(val_splits))]
    test_xs = [[all_data[key]["feature"] for key in test_splits[i]] for i in range(len(test_splits))]
    test_ys = [[all_data[key]["label_%d"%(hierarchy)] for key in test_splits[i]] for i in range(len(test_splits))]

    return train_xs, train_ys, val_xs, val_ys, test_xs, test_ys


def load_siutations_kfold_splits(data_feature_dir, split_dir, aggregation, config):
    _, all_data = load_pickle_data(data_feature_dir, aggregation, config)
    
    train_splits, val_splits, test_splits = [], [], []
    for i in range(1,11):
        train_splits.append(open(os.path.join(split_dir, "10fold/train_%d.txt"%(i)), "r").read().splitlines())
        val_splits.append(open(os.path.join(split_dir, "10fold/val_%d.txt"%(i)), "r").read().splitlines())
        test_splits.append(open(os.path.join(split_dir, "10fold/test_%d.txt"%(i)), "r").read().splitlines())

    train_xs = [[all_data[key]["feature"] for key in train_splits[i]] for i in range(len(train_splits))]
    train_ys = [[all_data[key]["label"] for key in train_splits[i]] for i in range(len(train_splits))]
    val_xs = [[all_data[key]["feature"] for key in val_splits[i]] for i in range(len(val_splits))]
    val_ys = [[all_data[key]["label"] for key in val_splits[i]] for i in range(len(val_splits))]
    test_xs = [[all_data[key]["feature"] for key in test_splits[i]] for i in range(len(test_splits))]
    test_ys = [[all_data[key]["label"] for key in test_splits[i]] for i in range(len(test_splits))]

    return train_xs, train_ys, val_xs, val_ys, test_xs, test_ys


def load_speaker_cross_splits(data_feature_dir, split_dir, aggregation, hierarchy, config, test_domain):
    _, all_data = load_pickle_data(data_feature_dir, aggregation, config)

    if test_domain == "tag":
        train_split = open(os.path.join(split_dir, "level_%d"%(hierarchy), "cross/bildcom_train.txt"), "r").read().splitlines()
        val_split = open(os.path.join(split_dir, "level_%d"%(hierarchy), "cross/bildcom_val.txt"), "r").read().splitlines()
        test_split = open(os.path.join(split_dir, "level_%d"%(hierarchy), "cross/tag_test.txt"), "r").read().splitlines()
    elif test_domain == "bild":
        train_split = open(os.path.join(split_dir, "level_%d"%(hierarchy), "cross/tagcom_train.txt"), "r").read().splitlines()
        val_split = open(os.path.join(split_dir, "level_%d"%(hierarchy), "cross/tagcom_val.txt"), "r").read().splitlines()
        test_split = open(os.path.join(split_dir, "level_%d"%(hierarchy), "cross/bild_test.txt"), "r").read().splitlines()
    elif test_domain == "com":
        train_split = open(os.path.join(split_dir, "level_%d"%(hierarchy), "cross/bildtag_train.txt"), "r").read().splitlines()
        val_split = open(os.path.join(split_dir, "level_%d"%(hierarchy), "cross/bildtag_val.txt"), "r").read().splitlines()
        test_split = open(os.path.join(split_dir, "level_%d"%(hierarchy), "cross/com_test.txt"), "r").read().splitlines()
    else:
        raise ValueError("Invalid test domain, choose between 'tag', 'bild', 'com'")

    train_x = [all_data[key]["feature"] for key in train_split]
    train_y = [all_data[key]["label_%d"%(hierarchy)]  for key in train_split]
    val_x = [all_data[key]["feature"] for key in val_split]
    val_y = [all_data[key]["label_%d"%(hierarchy)]  for key in val_split]
    test_x = [all_data[key]["feature"] for key in test_split]
    test_y = [all_data[key]["label_%d"%(hierarchy)]  for key in test_split]

    return train_x, train_y, val_x, val_y, test_x, test_y


def load_siutations_cross_splits(data_feature_dir, split_dir, aggregation, config, test_domain):
    _, all_data = load_pickle_data(data_feature_dir, aggregation, config)

    if test_domain == "tag":
        train_split = open(os.path.join(split_dir, "cross/bildcom_train.txt"), "r").read().splitlines()
        val_split = open(os.path.join(split_dir, "cross/bildcom_val.txt"), "r").read().splitlines()
        test_split = open(os.path.join(split_dir, "cross/tag_test.txt"), "r").read().splitlines()
    elif test_domain == "bild":
        train_split = open(os.path.join(split_dir, "cross/tagcom_train.txt"), "r").read().splitlines()
        val_split = open(os.path.join(split_dir, "cross/tagcom_val.txt"), "r").read().splitlines()
        test_split = open(os.path.join(split_dir, "cross/bild_test.txt"), "r").read().splitlines()
    elif test_domain == "com":
        train_split = open(os.path.join(split_dir, "cross/bildtag_train.txt"), "r").read().splitlines()
        val_split = open(os.path.join(split_dir, "cross/bildtag_val.txt"), "r").read().splitlines()
        test_split = open(os.path.join(split_dir, "cross/com_test.txt"), "r").read().splitlines()
    else:
        raise ValueError("Invalid test domain, choose between 'tag', 'bild', 'com'")
    
    train_x = [all_data[key]["feature"] for key in train_split]
    train_y = [all_data[key]["label"]  for key in train_split]
    val_x = [all_data[key]["feature"] for key in val_split]
    val_y = [all_data[key]["label"]  for key in val_split]
    test_x = [all_data[key]["feature"] for key in test_split]
    test_y = [all_data[key]["label"]  for key in test_split]

    return train_x, train_y, val_x, val_y, test_x, test_y