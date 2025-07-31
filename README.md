# sEMG Signal Processing and Classification

This repository provides a complete pipeline for processing and classifying surface Electromyography (sEMG) signals from the [SeNic dataset](https://github.com/BoZhuBo/SeNic). The project uses PyTorch Lightning for model training and evaluation, and provides a structured workflow from raw data extraction to model analysis.

## Features

*   **Automated Data Preparation**: Scripts to handle `.rar` extraction, signal preprocessing, and dataset consolidation.
*   **Configurable Workflow**: YAML files to easily manage preprocessing, training, and model parameters.
*   **Cross-Validation**: Built-in 3-fold cross-validation for robust model evaluation.
*   **Multiple Architectures**: Supports several deep learning models, including CNN, TCN, and MESTNet.
*   **Modular and Extensible**: The code is organized into logical modules for easy extension and modification.

## Getting Started

### Prerequisites

*   Python 3.8+
*   `unrar` command-line tool. Install it with `sudo apt-get install unrar` (Debian/Ubuntu) or `brew install unrar` (macOS).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/leloso/sEMG
    cd sEMG
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The workflow is divided into four main stages: data preparation, preprocessing, dataset building, and model training/evaluation.

### 1. Data Preparation

1.  **Download the raw data** from the [SeNic GitHub repository](https://github.com/BoZhuBo/SeNic).
2.  Place the downloaded `.rar` files into a directory, for example, `raw_data/`.
3.  **Extract the archives** using the `unrar_script.py`:
    ```bash
    python src/unrar_script.py raw_data/
    ```
    This will extract the contents into the same directory. You can use the `--delete` flag to remove the `.rar` files after extraction.

### 2. Preprocessing

The `preprocess.py` script processes the extracted CSV files, applies filters, and windows the data into `.npz` files. The preprocessing parameters are defined in `config/preprocess.yaml`.

```bash
python src/preprocess.py --input_dir raw_data/ --output_dir processed_data/ --config_file config/preprocess.yaml
```

### 3. Building the HDF5 Dataset

Consolidate the preprocessed `.npz` files into a single HDF5 file for efficient loading during training.

```bash
python src/build_hdf5.py --data_dir processed_data/ --hdf5_path emg_data.h5
```

### 4. Model Training

The `train.py` script performs 3-fold cross-validation for a specified model and experiment configuration. The training parameters are defined in `config/train.yaml`.

```bash
python src/train.py --result_dir results/ --config_file config/train.yaml
```

The script will train and save the best model for each fold in the `results/` directory.

### 5. Model Evaluation

After training, you can evaluate the models using the `eval.py` script.

```bash
python src/eval.py --models_dir results/ --config_file config/train.yaml --result_dir evaluation_results/
```

This will load the trained models and generate evaluation metrics, which will be saved in the `evaluation_results/` directory.

## Project Structure

```
sEMG/
├── config/
│   ├── preprocess.yaml
│   └── train.yaml
├── src/
│   ├── build_hdf5.py
│   ├── dataset.py
│   ├── eval.py
│   ├── models.py
│   ├── preprocess.py
│   ├── train.py
│   └── unrar_script.py
├── README.md
└── requirements.txt
```

## Model Architectures

The `src/models.py` file contains implementations of several deep learning architectures:

*   **CNN**: A baseline 1D Convolutional Neural Network.
*   **CNN_GRU, CNN_BiGRU, CNN_LSTM**: Hybrid models combining CNN layers for feature extraction and RNN layers for temporal modeling.
*   **TCN**: A Temporal Convolutional Network with dilated convolutions.
*   **MESTNet**: A Multi-scale EMG Signal Transformer with time-frequency analysis.

You can select the model to be trained in the `config/train.yaml` file.