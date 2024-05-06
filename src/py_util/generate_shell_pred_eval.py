config_version_map = [
    # ("BertAAD", 101, 102),
    # ("BertBiCond", 101, 104),
    # ("BertAttn", 101, 116),
    # ("BertJointAttn", 101, 116),
    # ("BertCrossNet", 101, 108),
]

model_dataset = "ustancebr" #"ustancebr" or "govbr"
test_dataset = "govbr" #"ustancebr" or "govbr"
data_folder = "in_domain" #"hold1topic_out" or "in_domain"
out_path_prefix = f"./sh_auto_pred_eval_{model_dataset}_{test_dataset}_{data_folder}"

columns_set = ["Text","Target","Polarity"]

dataset_path = test_dataset
if test_dataset in ["ustancebr"]:
    dataset_path = "ustancebr/v2"

data_folder_ckp = data_folder
if data_folder == "in_domain_15k":
    data_folder = "in_domain"


if model_dataset in ["ustancebr", "govbr"]:
    model_target_list = ["bo", "lu", "co", "cl", "gl", "ig"]
    # model_target_list = ["ig"]

if test_dataset in ["ustancebr", "govbr"]:
    test_target_list = ["bo", "lu", "co", "cl", "gl", "ig"]

test_set_name = "test"
base_config = " -c ../../config/{model_dataset}/{data_folder}/{config_file_name}_v{k}.txt"
base_test_path = " -p ../../data/{dataset_path}/{data_folder_path}/final_{dst_tgt}_{test_set_name}.csv"
# base_vld_path = " -v ../../data/{dataset_path}/{data_folder_path}/final_{src_tgt}_valid.csv"
base_vld_path = ""
base_ckp_path = " -f ../../checkpoints/{model_dataset}/{data_folder_ckp}/{model_folder}/V{k}/ckp-{config_file_name}_{model_dataset}_{src_tgt}-BEST.tar"
base_override_columns = ""

if columns_set is not None:
    base_override_columns = f" -pt {columns_set[0]} -pg {columns_set[1]} -pl {columns_set[2]}"

if model_dataset == test_dataset:
    base_out =  " -o ../../out/{model_dataset}/pred/{data_folder_ckp}/{src_tgt}_{dst_tgt}_{config_file_name}_v{k}.txt"
    base_log =  " > ../../out/{model_dataset}/eval/{data_folder_ckp}/{src_tgt}_{dst_tgt}_{config_file_name}_v{k}.txt"
else:
    base_out =  " -o ../../out/{test_dataset}/pred/{model_dataset}_{data_folder_ckp}/{src_tgt}_{dst_tgt}_{config_file_name}_v{k}.txt"
    base_log =  " > ../../out/{test_dataset}/eval/{model_dataset}_{data_folder_ckp}/{src_tgt}_{dst_tgt}_{config_file_name}_v{k}.txt"


base_command = "python train_model.py -m pred_eval" + \
                base_config + \
                base_test_path + \
                base_vld_path + \
                base_ckp_path + \
                base_override_columns + \
                base_out + \
                base_log

if data_folder == "in_domain":
    base_text = ""
    for src_tgt in model_target_list:
        base_text += "\n"
        for dst_tgt in test_target_list:
            base_text += "\n" + base_command.replace("{src_tgt}", src_tgt).replace("{dst_tgt}", dst_tgt)

elif data_folder == "hold1topic_out":
    base_text = ""
    for tgt in model_target_list:
        base_text += "\n" + base_command.replace("{src_tgt}", tgt).replace("{dst_tgt}", tgt)

for config_file_name, init_version, final_version in config_version_map:
    out_path = f"{out_path_prefix}_{config_file_name}.sh"
    
    with open(out_path, "w") as f_:
        print("\n", file=f_)

    for k in range(init_version, final_version+1):
        partial_text = base_text \
        .replace("{config_file_name}", config_file_name) \
        .replace("{model_folder}", config_file_name.lower().replace("bert", "")) \
        .replace("{k}", str(k)) \
        .replace("{data_folder}", data_folder) \
        .replace("{data_folder_path}", data_folder_path) \
        .replace("{data_folder_ckp}", data_folder_ckp) \
        .replace("{model_dataset}", model_dataset) \
        .replace("{test_dataset}", test_dataset) \
        .replace("{dataset_path}", dataset_path) \
        .replace("{test_set_name}", test_set_name)

        with open(out_path, "a") as f_:
            print(partial_text, "\n", file=f_)
        
    with open(out_path, "a") as f_:
        print("\n", file=f_, flush=True)
