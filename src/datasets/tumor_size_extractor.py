import csv
import yaml
from pathlib import Path
from tuh_dataset import TUHDataset

def extract_tumor_sizes(dataset: TUHDataset, output_csv: str):
    results = []
    for scan in dataset.scans:
        if 'tumor' not in scan.class_to_idx:
            print(f"Warning: 'tumor' class not found in scan {scan.name}. Skipping.")
            continue
        tumor_size = (scan.segm == scan.class_to_idx['tumor']).sum()
        results.append({'scan_name': scan.name, 'tumor_size': tumor_size})

    # Write results to a CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['scan_name', 'tumor_size'])
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    # Load configuration from YAML file
    config_path = "/users/lillemag/makatoo/coin_test/counterfactual-search/configs/THESIS_EXPERIMETNS/Classifier_OG_COIN.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    dataset_config = config['dataset']['datasets'][0]
    scan_params = dataset_config['scan_params']
    root_dir = dataset_config['root_dir']
    split_dir = dataset_config['split_dir']
    split = "test"  # You can make this configurable if needed

    # Initialize dataset
    dataset = TUHDataset(
        root_dir=root_dir,
        split=split,
        split_dir=split_dir,
        **scan_params
    )

    # Output CSV file
    output_csv = "tumor_sizes_test.csv"
    extract_tumor_sizes(dataset, output_csv)
    print(f"Tumor sizes saved to {output_csv}")
