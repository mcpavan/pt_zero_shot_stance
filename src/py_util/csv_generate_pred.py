import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from itertools import product
from sklearn.metrics import recall_score, precision_score, f1_score
import matplotlib.pyplot as plt

dataset = "ustancebr" # "ustancebr" or "govbr"
folder = ""
pred_base_path = f"../../out/{dataset}/pred"
if folder != "":
    pred_base_path = f"{pred_base_path}/{folder}"
eval_base_path = f"../../out/{dataset}/eval"
out_path = f"{eval_base_path}/.results/data/pred_data.csv"
errors_out_path = f"{eval_base_path}/.results/data/errors_pred_data.csv"
os.makedirs(f"{eval_base_path}/.results/data/", exist_ok=True)

data_list = ["test"]
metric_list = ["f", "p", "r"]

if dataset in ["ustancebr", "govbr"]:
    class_list = ["macro", "0", "1"]
    labels_list = ["against", "for"]

pred_results_dict = {
    "data_folder": [],
    "source_topic": [],
    "destination_topic": [],
    "config_file_name": [],
    "version": [],
}

def get_best_threshold(df, proba_col, invert=False):
    best_f = np.array([0])
    best_thr0 = 0
    best_thr1 = 0
    p_min = int(df[proba_col].min()*100)+1
    p_max = int(df[proba_col].max()*100)
    step = max(int((p_max-p_min)/10), 1)
    cls1 = labels_list[0]
    cls2 = labels_list[-1]
    if invert:
        cls1 = labels_list[-1]
        cls2 = labels_list[0]
    
    if len(labels_list) == 2:
        for thr in range(p_min, p_max, step):
            current_pred = df[proba_col].apply(lambda x: cls1 if x<thr/100 else cls2)

            f_val = f1_score(
                y_true=df[target_col].apply(lambda x: cls_map[x.lower()]),
                y_pred=current_pred.apply(lambda x: cls_map[x.lower()]),
                average=None,
                zero_division=0,
            )

            if f_val.mean() > best_f.mean():
                best_f = f_val
                best_thr0 = thr

        return {
            "best_f1_0": best_f[0],
            "best_f1_1": best_f[1],
            "best_f1_avg": best_f.mean(),
            "best_thr": best_thr0/100,
        }
    else:
        for thr_0 in range(p_min, p_max, step):
            for thr_1 in range(thr_0+1, p_max, step):
                current_pred = df[proba_col].apply(lambda x: cls1 if x<thr_0/100 else cls2 if x>=thr_1/100 else labels_list[1])

                f_val = f1_score(
                    y_true=df[target_col].apply(lambda x: cls_map[x.lower()]),
                    y_pred=current_pred.apply(lambda x: cls_map[x.lower()]),
                    average=None,
                    zero_division=0,
                )

                if f_val.mean() > best_f.mean():
                    best_f = f_val
                    best_thr0 = thr_0
                    best_thr1 = thr_1
        
        return {
            "best_f1_0": best_f[0],
            "best_f1_1": best_f[1],
            "best_f1_2": best_f[2],
            "best_f1_avg": best_f.mean(),
            "best_thr0": best_thr0/100,
            "best_thr1": best_thr1/100,
        }

for data_, metric_, class_ in product(data_list, metric_list, class_list):
    pred_results_dict[f"{data_}_{metric_}{class_}"] = []

for pred_file_path in tqdm(glob(f"{pred_base_path}/**/*.csv", recursive=True)):
    pred_file_path = pred_file_path.replace("\\", "/")
    if folder == "":
        data_folder = pred_file_path.replace(f"{pred_base_path}/", "").split("/")[0]
    else:
        data_folder = folder
    saved_file_name = pred_file_path.replace(f"{pred_base_path}/", "").split("/")[-1]
    source_topic = saved_file_name.split("_")[0]
    destination_topic = saved_file_name.split("_")[1]
    config_file_name = "_".join(saved_file_name.split("_")[2:-1])
    version = saved_file_name.split("_")[-1].replace(".txt-test.csv","")[1:]

    df = pd.read_csv(pred_file_path)
    pred_col = [col for col in df.columns if col.endswith("_pred")][0]
    proba_col = [col for col in df.columns if col.endswith("_proba")][0]
    target_col = pred_col.replace("_pred", "")

    metric_dict = {
        "p": precision_score,
        "r": recall_score,
        "f": f1_score,
    }
    
    tgt_classes = df[target_col].unique()
    tgt_classes.sort()
    cls_map = {cls_.lower():i for i, cls_ in enumerate(tgt_classes)}
    pred_col = "my_pred"

    thr_dict = get_best_threshold(df, proba_col, invert=False)
    if len(labels_list) > 2:
        bins = [0, thr_dict["best_thr0"], thr_dict["best_thr1"], 1]
    else:
        bins = [0, thr_dict["best_thr"], 1]
    bins.sort()
    df["my_pred"] = pd.cut(
        x=df[proba_col],
        bins=bins,
        labels=labels_list,
    )

    for metric_name, metric_fn in metric_dict.items():
        metric_values = metric_fn(
            y_true=df[target_col].apply(lambda x: cls_map[x.lower()]),
            y_pred=df[pred_col].apply(lambda x: cls_map[x.lower()]),
            average=None,
            zero_division=0,
        )

        for i in range(len(metric_values)):
            pred_results_dict[f"test_{metric_name}{i}"] += [metric_values[i]]
        pred_results_dict[f"test_{metric_name}macro"] += [np.mean(metric_values)]

    pred_results_dict["data_folder"] += [data_folder]
    pred_results_dict["source_topic"] += [source_topic]
    pred_results_dict["destination_topic"] += [destination_topic]
    pred_results_dict["config_file_name"] += [config_file_name]
    pred_results_dict["version"] += [version]

    # INVERT

    thr_dict = get_best_threshold(df, proba_col, invert=True)
    if len(labels_list) > 2:
        bins = [0, thr_dict["best_thr0"], thr_dict["best_thr1"], 1]
    else:
        bins = [0, thr_dict["best_thr"], 1]
    bins.sort()
    df["new_pred"] = pd.cut(
        x=df[proba_col],
        bins=bins,
        labels=labels_list,
    )
    for metric_name, metric_fn in metric_dict.items():
        metric_values = metric_fn(
            y_true=df[target_col].apply(lambda x: cls_map[x.lower()]),
            y_pred=df["new_pred"].apply(lambda x: cls_map[x.lower()]),
            average=None,
            zero_division=0,
        )
        
        for i in range(len(metric_values)):
            pred_results_dict[f"test_{metric_name}{i}"] += [metric_values[i]]
        pred_results_dict[f"test_{metric_name}macro"] += [np.mean(metric_values)]

    pred_results_dict["data_folder"] += [data_folder]
    pred_results_dict["source_topic"] += [source_topic]
    pred_results_dict["destination_topic"] += [destination_topic]
    pred_results_dict["config_file_name"] += [config_file_name]
    pred_results_dict["version"] += [f"{version}inv"]

    current_length = len(pred_results_dict["data_folder"])

for key in pred_results_dict.keys():
    if len(pred_results_dict[key]) < current_length:
        pred_results_dict[key] += [None]

df_results = pd.DataFrame(pred_results_dict)
df_results = df_results.dropna(subset=["test_fmacro"], how="all")
df_results.to_csv(out_path, index=False)

errors_out_cols = [
    "config_file_name",
    "version",
    "source_topic",
    "destination_topic",
]
df_results \
    .query("test_p1 != test_p1") \
    [errors_out_cols] \
    .sort_values(errors_out_cols) \
    .to_csv(errors_out_path, index=False)