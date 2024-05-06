
config_version_map = [
    # ("BertAAD", 1, 2),
    ("BertBiCond", 1, 4),
    ("BertAttn", 1, 16),
    ("BertJointAttn", 1, 16),
    ("BertCrossNet", 1, 8),
]

# config_file_name = "BertAttn"
data_folder = "in_domain" #"hold1topic_out" or "in_domain"
dataset = "ustancebr" # or "govbr"

base_command = "python train_model.py -m train"
base_config = " -c ../../config/{dataset}/{data_folder}/{config_file_name}_v{k}.txt"

if dataset == "ustancebr" or dataset == "govbr":
    target_model_list = ["bo", "lu", "co", "cl", "gl", "ig"]
    base_trn_path = " -t ../../data/ustancebr/v2/{data_folder}/final_{tgt}_train.csv"
    base_vld_path = " -v ../../data/ustancebr/v2/{data_folder}/final_{tgt}_valid.csv"
    base_tst_path = " -p ../../data/ustancebr/v2/{data_folder}/final_{tgt}_test.csv"


base_others = " -n {tgt} -e 5 -s 1"
base_command = base_command + base_config + base_trn_path + base_vld_path + base_tst_path + base_others
base_out = " > ../../out/{dataset}/log/{data_folder}/{tgt}_{config_file_name}_v{k}.txt"

for config_file_name, init_version, final_version in config_version_map:
    out_path = f"./sh_auto_train_{dataset}_{data_folder}_{config_file_name}.sh"
    with open(out_path, "w") as f_:
        print("", end="", file=f_)

    base_text = "\n"

    if config_file_name == "BertAAD":

        if dataset == "ustancebr":
            tgt_trn_path = "../../data/ustancebr/v2/in_domain/final_"
            if data_folder == "in_domain": # manual crpss target
                tgt_train_list = ["lu", "bo", "cl", "co", "ig", "gl"]
            elif data_folder == "hold1topic_out":  # files already set up
                tgt_train_list = ["bo", "lu", "co", "cl", "gl", "ig"]        

        for k, tgt in enumerate(target_model_list):
            base_text += base_command.replace("{tgt}", tgt) + " -g " + tgt_trn_path + tgt_train_list[k] + "_train.csv" + base_out.replace("{tgt}", tgt) + "\n"
    
    elif dataset == "govbr":
        sample_size = {
            "bo": " -a 15000 -j 60000",
            "lu": " -a 15000 -j 20000",
            "co": " -a 15000",
            "cl": " -a 15000 -j 2500",
            "gl": " -a 15000 -j 4000",
            "ig": " -a 15000",
        }

        for k, tgt in enumerate(target_model_list):
            testtgt = ""
            if isinstance(tgt, tuple):
                testtgt = tgt[1]
                tgt = "_".join(tgt)
            base_text += base_command.replace("{tgt}", tgt).replace("{testtgt}", testtgt) + sample_size.get(tgt, "") + base_out.replace("{tgt}", tgt) + "\n"

    else:
        for k, tgt in enumerate(target_model_list):
            base_text += base_command.replace("{tgt}", tgt) + base_out.replace("{tgt}", tgt) + "\n"

    for k in range(init_version, final_version+1):
        partial_text = base_text \
        .replace("{config_file_name}", config_file_name) \
        .replace("{k}", str(k)) \
        .replace("{dataset}", dataset) \
        .replace("{data_folder}", data_folder) \

        with open(out_path, "a") as f_:
            print(partial_text, "\n\n", file=f_)
    
    with open(out_path, "a") as f_:
        print("\n", file=f_, flush=True)
