import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Replace <DATA_PATH> in a JSON file.")
    parser.add_argument('--data_path', required=True, help='Path to replace <DATA_PATH> with')
    parser.add_argument('--input', default='comparison/config/SSC_dataset_split_data_path.json', help='Input JSON file')
    parser.add_argument('--output', default='comparison/config/SSC_dataset_split_new.json', help='Output JSON file')
    
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)

    for key in data.keys():
        for i in range(len(data[key])):
            if isinstance(data[key][i], str):
                data[key][i] = data[key][i].replace('<DATA_PATH>', args.data_path)

    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Updated JSON written to {args.output}")

if __name__ == "__main__":
    main()
