from itertools import product
import os

first_v_config = 1
folder = "simple_domain" # "simple_domain" or "hold1topic_out""or "zero_shot"
dataset = "ustancebr"

modelname2example = {
    "BertAAD": "./example/Bert_AAD_example.txt",
    "BiCondBertLstm": "./example/Bert_BiCondLstm_example.txt",
    "BertBiLSTMAttn": "./example/Bert_BiLstmAttn_example.txt",
    "BertBiLSTMJointAttn": "./example/Bert_BiLstmJointAttn_example.txt",
    "BertCrossNet": "./example/Bert_CrossNet_example.txt",
    "Llama_4bit": "./Llama_4bit_example.txt",
}

default_params_ustancebr = {
    "text_col":"Text",
    "topic_col":"Target",
    "label_col":"Polarity",
    "sample_weights": 1,
    "n_output_classes": 2,
}

values_dict_ustancebr = {
    "BertAAD": {
        "bert_pretrained_model": [
            "pablocosta/bertabaporu-base-uncased",
        ],
        "bert_layers": [
            "-4,-3,-2,-1",
        ],
        "learning_rate": [
            1e-7,
        ],
        "discriminator_learning_rate": [
            1e-7,
        ],
        "discriminator_dim": [
            1024,
            3072,
        ]

    },
    "BertBiCond": {
        "bert_pretrained_model": [
            "pablocosta/bertabaporu-base-uncased",
        ],
        "bert_layers": [
            "-4,-3,-2,-1",
        ],
        "lstm_layers": [
            "1",
            "2",
        ],
        "lstm_hidden_dim": [
            "16",
            "128",
        ],
        "learning_rate": [
            1e-7,
        ]
    },
    "BertAttn": {
        "bert_pretrained_model": [
            "pablocosta/bertabaporu-base-uncased",
        ],
        "bert_layers": [
            "-4,-3,-2,-1",
        ],
        "lstm_layers": [
            "1",
            "2",
        ],
        "lstm_hidden_dim": [
            "16",
            "128",
        ],
        "attention_density": [
            "32",
            "64",
        ],
        "attention_heads": [
            "1",
            "16",
        ],
    },
    "BertJointAttn": {
        "bert_pretrained_model": [
            "pablocosta/bertabaporu-base-uncased",
        ],
        "bert_layers": [
            "-4,-3,-2,-1",
        ],
        "lstm_layers": [
            "1",
            "2",
        ],
        "lstm_hidden_dim": [
            "16",
            "128",
        ],
        "attention_density": [
            "32",
            "64",
        ],
        "attention_heads": [
            "1",
            "16",
        ],
    },
    "BertCrossNet": {
        "bert_pretrained_model": [
            "pablocosta/bertabaporu-base-uncased",
        ],
        "bert_layers": [
            "-4,-3,-2,-1",
        ],
        "lstm_layers": [
            "1",
            "2",
        ],
        "lstm_hidden_dim": [
            "16",
            "128",
        ],
        "attention_density": [
            "100",
            "200",
        ],
        "learning_rate": [
            1e-7,
        ],
    },
    "Llama_4bit": {
        "pretrained_model_name": [
            "../../data/LLMs/ggml-alpaca-7b-q4.bin",
        ],
        "prompt": [
            {
                "prompt_template_file": "../../data/ustancebr/prompts/stance_prompt_alpaca_score10_0.md",
                "output_format": "score",
                "output_max_score": 10,
                "output_err_default": 0.0,
            },
            {
                "prompt_template_file": "../../data/ustancebr/prompts/stance_prompt_alpaca_score10_1.md",
                "output_format": "score",
                "output_max_score": 10,
                "output_err_default": 0.0,
            },
            {
                "prompt_template_file": "../../data/ustancebr/prompts/stance_prompt_alpaca_score100_0.md",
                "output_format": "score",
                "output_max_score": 100,
                "output_err_default": 0.0,
            },
            {
                "prompt_template_file": "../../data/ustancebr/prompts/stance_prompt_alpaca_score100_1.md",
                "output_format": "score",
                "output_max_score": 100,
                "output_err_default": 0.0,
            },
            {
                "prompt_template_file": "../../data/ustancebr/prompts/stance_prompt_alpaca_score10_0.md",
                "output_format": "set",
                "output_max_score": 0,
                "output_err_default": 1.0,
            },
        ],
        "batch_size": [
            1,
        ],
    },
}

def load_config_file(config_file_path):
    with open(config_file_path, 'r') as f:
        config = dict()
        for l in f.readlines():
            config[l.strip().split(":")[0]] = l.strip().split(":")[1]
    
    return config


out_path = "./{dataset}/{folder}/{model_name_out_file}_v{k}.txt"
ckp_path = "../../checkpoints/{dataset}/{folder}/{name}/V{k}/"

if dataset == "ustancebr":
    values_dict = values_dict_ustancebr
    default_params = default_params_ustancebr

for model_name_out_file, example_file in modelname2example.items():
    os.makedirs(
        "/".join(out_path.split("/")[:-1]) \
           .replace("{dataset}", dataset) \
           .replace("{folder}", folder),
        exist_ok=True
    )

    base_config_dict = load_config_file(example_file)
    base_config_dict["name"] = f"{model_name_out_file}_{dataset}_"
    k = first_v_config
    for combination in product(*list(values_dict[model_name_out_file].values())):
        
        new_config_dict = base_config_dict
        # setting default params
        for key, value in default_params.items():
            new_config_dict[key] = value

        # setting combination specific params
        for key, value in zip(values_dict[model_name_out_file].keys(), combination):
            if model_name_out_file == "Llama_4bit" and key in ["prompt", "model"]:
                for prompt_key, prompt_value in value.items():
                    new_config_dict[prompt_key] = prompt_value
            else:
                new_config_dict[key] = value
        
        new_config_dict["ckp_path"] = ckp_path \
            .replace("{dataset}", dataset) \
            .replace("{folder}", folder) \
            .replace(
                "{name}",
                model_name_out_file.lower().replace("bert", ""),
            ) \
            .replace("{k}", str(k))             
        current_out_path = out_path \
            .replace("{dataset}", dataset) \
            .replace("{folder}", folder) \
            .replace("{model_name_out_file}", model_name_out_file) \
            .replace("{k}", str(k))
        
        with open(current_out_path, "w") as f_:
            new_config_str = "\n".join(f"{k}:{v}" for k, v in new_config_dict.items())
            print(new_config_str, end="", file=f_, flush=True)
        
        k += 1
    