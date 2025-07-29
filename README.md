 **SeNic Data Processing Pipeline**

This repository provides a robust pipeline for processing the raw data from the [SeNic dataset](https://www.google.com/search?q=https://github.com/BoZhuBo/SeNic.git), transforming it from compressed RAR archives into a structured HDF5 file suitable for machine learning applications, particularly with PyTorch.

## **✨ Features**

* **Automated Extraction**: Uncompress RAR archives recursively.  
* **Flexible Preprocessing**: Filter and window time-series data using a configurable YAML file.  
* **Hierarchical Output**: Maintain original directory structure for processed data.  
* **Unified Dataset**: Consolidate all processed data into a single HDF5 file for efficient loading.

## **🚀 Getting Started**

### **Prerequisites**

Before you begin, ensure you have the following installed:

* Python 3.8+  
* unrar command-line tool (install via your system's package manager, e.g., sudo apt-get install unrar on Debian/Ubuntu, brew install unrar on macOS).

### **Installation**

1. **Clone the repository**:  
   git clone https://github.com/your-repo/your-project-name.git  
   cd robust-semg

2. **Install Python dependencies**:  
   pip install \-r requirements.txt

## **📖 Usage**

Follow these steps to process the SeNic dataset:

### **1\. Acquire Raw Data**

First, download the raw .rar data files from the [SeNic GitHub repository](https://www.google.com/search?q=https://github.com/BoZhuBo/SeNic.git). Place these files in a designated root directory (e.g., raw\_data/).

### **2\. Extract RAR Archives**

Use the unrar\_script.py to extract the contents of the .rar files. This script will recursively search for .rar files and extract them.  
python unrar\_script.py \<directory\_path\> \[--delete\]

* \<directory\_path\>: The root directory where your .rar files are located (e.g., raw\_data).  
* \--delete: (Optional) If present, .rar files will be deleted after successful extraction.

