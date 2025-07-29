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
