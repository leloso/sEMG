import argparse
import sys
import os

import yaml # Import PyYAML
import pandas as pd
import numpy as np

from utils import sliding_window, preprocess_signal

# Suppress specific pandas warning if using pd.read_csv for flexibility
# warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


# --- 1. Gesture Map ---
# Define the gesture_map globally or pass it. If it's fixed, global is fine.
# Note: In your original script, you had two gesture_maps.
# Let's keep one here and adjust the label mapping as you did (label-1)
gesture_map = {
    "fist": 0,
    "middlefinger": 1,
    "two": 2,
    "hand": 3,
    "forefinger": 4,
    "varus": 5,
    "eversion": 6,
}


# --- 2. Preprocessing Function ---


def process_csv_file(csv_filepath: str, input_base_dir: str, output_base_dir: str, config) -> bool:
    """
    Loads a CSV, preprocesses it, and saves it as a compressed NPZ file.

    Args:
        csv_filepath (str): Full path to the input CSV file.
        input_base_dir (str): The root directory from which the script started scanning.
        output_base_dir (str): The base directory where NPZ files should be saved.
        preprocessing_func (callable): The function to apply for preprocessing.

    Returns:
        bool: True if processing was successful, False otherwise.
    """
    try:
        print(f"Loading CSV: {csv_filepath}")
        
        df = pd.read_csv(csv_filepath, header=None)

        if df.shape[1] != 8:
            print(f"  Warning: CSV {csv_filepath} has {df.shape[1]} columns, expected 8. Skipping.")
            return False

        #Convert to numpy array and ensure integer type (if it's truly all integers)
        
        signal_data = np.loadtxt(csv_filepath, delimiter=',', dtype=np.float32) # Converts to numpy array

        # Apply preprocessing
        processed_signal_data = preprocess_signal(signal_data, config)

        windows = sliding_window(
                processed_signal_data,
                window_size=config['windowing']['window_size'],
                stride=config['windowing']['stride']
        )

        # Determine output path while maintaining hierarchy
        relative_path = os.path.relpath(csv_filepath, start=input_base_dir)
        output_npz_filename = os.path.splitext(relative_path)[0] + '.npz'
        output_npz_filepath = os.path.join(output_base_dir, output_npz_filename)

        # Create output directory if it doesn't exist
        output_dir_for_npz = os.path.dirname(output_npz_filepath)
        if not os.path.exists(output_dir_for_npz):
            os.makedirs(output_dir_for_npz)

        # Save to compressed NPZ
        np.savez_compressed(output_npz_filepath, samples=np.swapaxes(windows, 1, 2))
        return True
    except ValueError as e:
        # Catches errors from to_numpy().astype(int) if data isn't pure integer
        print(f"  Error converting data in {csv_filepath} to integer (or other value error): {e}. Skipping.")
        return False
    except Exception as e:
        print(f"  An unexpected error occurred while processing {csv_filepath}: {e}. Skipping.")
        return False


def main():

    parser = argparse.ArgumentParser(
        description="Recursively loads CSV files, preprocesses signals, and saves them as compressed NPZ, replicating directory hierarchy."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True, 
        help="The root directory to recursively search for CSV files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True, 
        help="The root directory where processed NPZ files will be saved. "
             "The original file hierarchy will be replicated here."
    )
    parser.add_argument(
        '--config_file', 
        type=str, 
        required=True, 
        help='Path to the YAML configuration file.'
    )

    args = parser.parse_args()

    input_directory = os.path.abspath(args.input_dir)
    output_directory = os.path.abspath(args.output_dir)
    config_file_path = os.path.abspath(args.config_file)


    if not os.path.isdir(input_directory):
        print(f"Error: Input directory not found: {input_directory}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")
    elif not os.path.isdir(output_directory):
        print(f"Error: Output path exists but is not a directory: {output_directory}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(config_file_path):
        print(f"Error: Configuration file not found at {config_file_path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_file_path}")
        # Basic validation of config structure
        required_sections = ['preprocessing', 'windowing']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in config file.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {e}", file=sys.stderr)
        sys.exit(1)
    

    # Create the output directory if it doesn't exist
   

    print(f"Scanning for CSV files in: {input_directory}")
    print(f"Saving processed NPZ files to: {output_directory}")

    csv_found_count = 0
    npz_saved_count = 0

    for root, _, files in os.walk(input_directory):
        for filename in files:
            if filename.lower().endswith('.csv'):
                csv_found_count += 1
                csv_filepath = os.path.join(root, filename)
                
                if process_csv_file(csv_filepath, input_directory, output_directory, config):
                    npz_saved_count += 1

    print("\n--- Processing Summary ---")
    print(f"Total CSV files found: {csv_found_count}")
    print(f"Total NPZ files successfully saved: {npz_saved_count}")
    print(f"Files failed/skipped: {csv_found_count - npz_saved_count}")


if __name__ == "__main__":
    main()