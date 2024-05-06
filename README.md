# pt_zero_shot_stance
A benchmark for Portuguese zero-shot stance detection

# Requirements:
- python==3.7.10
- numpy==1.19.2
- matplotlib==3.5.2
- pandas==1.2.4
- seaborn==0.11.2
- tqdm==4.59.0
- scikit-learn==0.24.2
- pytorch==1.13.1 (Cuda 11.7)
- transformers==4.28.0

## Only for llama_cpp
- llama-cpp-python==0.1.48

# Instructions
Below are the general instructions to run the code to train, generate predictions and evaluate models in this project.

## Generate config data
Open the file in [config/generate_config.py](config/generate_config.py), check the models for which config files will be generated (`modelname2example` dict), check the parameters for each model (`values_dict_ustancebr` dict). Other parameters will have default values copied from the respective example file in `config/example/*.txt`. From folder [config](config), execute `python generate_config.py` to generate the setted configuration files.

## Train/predict/evaluate a model
General python code for training, prediction and evaluation reports is located in folder [src/py_util](src/py_util).

General syntax:
```
python train_model.py -m train -c [config_file] -t [train_data] -v [valid_data] -p [test_data] -n [experiment_name] -e [early_stopping] -s [save_checkpoints]
```

Example:
```
python train_model.py -m train -c ../../config/BertAttn_example.txt -t ../../data/UStanceBR/v2/in_domain/final_bo_train.csv -v ../../data/UStanceBR/v2/in_domain/final_bo_valid.csv -n bo -e 5 -s 1
```

For more argument options, check at the `main` function in the file [src/py_util/train_model.py](src/py_util/train_model.py). At the end of the file there a few other examples of how to use this file.

To train/predict/evaluate multiple models, the following giles can be used to generate a shell script useful to run multiple models/targets with one line command:
- [src/py_util/generate_shell_eval.py](src/py_util/generate_shell_eval.py)
- [src/py_util/generate_shell_pred_eval.py](src/py_util/generate_shell_pred_eval.py)
- [src/py_util/generate_shell_train.py](src/py_util/generate_shell_train.py)

## Reports
To generate CSV files with all the results returned in the logs, check the files:
- [src/py_util/csv_generate_eval.py](src/py_util/csv_generate_eval.py)
- [src/py_util/csv_generate_pred.py](src/py_util/csv_generate_pred.py)
- [src/py_util/csv_generate_train_log.py](src/py_util/csv_generate_train_log.py)
- [src/py_util/csv_analysis_eval.py](src/py_util/csv_analysis_eval.py)
- [src/py_util/csv_analysis_pred.py](src/py_util/csv_analysis_pred.py)
- [src/py_util/csv_analysis_train_log.py](src/py_util/csv_analysis_train_log.py)

The `csv_generate_*.py` files are useful to compile all results in a single CSV file, while `csv_analysis_*.py` files can be used to select only the best models for each task combination.

# Contact:
- [Matheus Camasmie Pavan](linkedin.com/in/matheus-camasmie-pavan) ([matheus.pavan@usp.br](matheus.pavan@usp.br))