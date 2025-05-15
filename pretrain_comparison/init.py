import os
import sys
import json
import argparse
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(sequence=4, offset=2)

def update_data_split(data_path):
    script_path = 'comparison/config/update_data_split.py'
    full_data_path = os.path.join(data_path, 'SSC_pre_2015/SSC_pre_2015')
    input_json_path = os.path.join(os.getcwd(), 'comparison/config/SSC_dataset_split_data_path.json')
    output_json_path = os.path.join(os.getcwd(), 'comparison/config/SSC_dataset_split_new.json')
    command = f"python {script_path} --data_path {full_data_path} --input {input_json_path} --output {output_json_path}"
    print(f"Running command: {command}")
    os.system(command)

def update_pretrain_config_files():
    cwd = os.getcwd()
    split_path = os.path.join(cwd, 'comparison/config/SSC_dataset_split_new.json')
    save_path = os.path.join(cwd, 'output/final_models')
    channel_groups_path = os.path.join(cwd, 'comparison/config/channel_groups.json')

    config_files = [
        'config_CL.yaml',
        'config_fft_MAE.yaml',
        'config_fft_MAE_masked.yaml',
        'config_fft_noise.yaml',
        'config_noise.yaml',
        'config_multimodalMAE_masked.yaml',
        'config_multimodalMAE.yaml',
    ]

    for config_file_name in config_files:
        config_path = os.path.join(cwd, 'comparison/config', config_file_name)
        print(f'Updating: {config_path}')

        with open(config_path, 'r') as f:
            config = yaml.load(f)

        # Only update specific keys
        config['split_path'] = split_path
        config['save_path'] = save_path
        config['channel_groups_path'] = channel_groups_path

        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        print(f"✅ Updated config: {config_path}")

def update_fine_tune_config(data_path):
    cwd = os.getcwd()
    fine_tune_config_path = os.path.join(cwd, 'fine_tune/config_fine_tune.yaml')

    updated_fields = {
        'ahi_labels_path': os.path.join(data_path, 'labels/labels/ahi_stats_labels_not_normalized_outliers_removed.csv'),
        'death_labels_path': os.path.join(data_path, 'labels/labels/phewas_tte_prediction_10_18_24_num_labels_1042'),
        'demographics_labels_path': os.path.join(data_path, 'labels/labels/demographics_info_10_14_24.csv'),
        'diagnosis_labels_path': os.path.join(data_path, 'labels/labels/phewas_tte_prediction_10_18_24_num_labels_12'),
        'labels_path': os.path.join(data_path, 'labels/labels/sleep_stages'),
        'embedding_path': os.path.join(cwd, 'output/final_embeddings'),
        'split_path': os.path.join(cwd, 'comparison/config/SSC_dataset_split_new.json'),
        'save_path': os.path.join(cwd, 'output/results'),
    }


    print(f'Updating: {fine_tune_config_path}')
    with open(fine_tune_config_path, 'r') as f:
        config = yaml.load(f)

    for key, value in updated_fields.items():
        config[key] = value

    with open(fine_tune_config_path, 'w') as f:
        yaml.dump(config, f)
    

    print(f"✅ Updated config: {fine_tune_config_path}")

def update_embed_config():
    cwd = os.getcwd()
    split_path = os.path.join(cwd, 'comparison/config/SSC_dataset_split_new.json')
    CL_config_file = os.path.join(cwd, 'comparison/config/config_CL_LOO.json')
    embed_file = os.path.join(cwd, 'comparison/config/config_embed.yaml')

    # --- Update the JSON file ---
    with open(CL_config_file, 'r') as f:
        cl_config = json.load(f)

    cl_config['split_path'] = split_path

    with open(CL_config_file, 'w') as f:
        json.dump(cl_config, f, indent=2)

    print(f"✅ Updated split_path in JSON: {CL_config_file}")

    # --- Update the YAML file ---
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(sequence=4, offset=2)

    with open(embed_file, 'r') as f:
        embed_config = yaml.load(f)

    embed_config['split_path'] = split_path

    with open(embed_file, 'w') as f:
        yaml.dump(embed_config, f)

    print(f"✅ Updated split_path in YAML: {embed_file}")

def __main__():
    parser = argparse.ArgumentParser(description='Update data split and config files.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory')
    args = parser.parse_args()

    update_data_split(args.data_path)
    update_pretrain_config_files()
    update_fine_tune_config(args.data_path)
    update_embed_config()
    print("✅ All configs initialized and updated.")

if __name__ == "__main__":
    __main__()
