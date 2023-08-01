# UTokyo_CNN_MRI
MRI data CNN controller. 
---
# Modular CNN for MRI and MNIST Data

This repository contains a modular Convolutional Neural Network (CNN) designed to test different pooling techniques, gradient functions, and feature classification strategies for MRI and MNIST data.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [VSCode Setup](#vscode-setup)
- [MRI Data Setup](#mri-data-setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Modular Design**: Easily swap out different components of the CNN, such as pooling layers or optimizers.
- **Support for Multiple Pooling Techniques**: Test with Max Pooling or Average Pooling.
- **Customizable Optimizers**: Choose between Adam and SGD optimizers.
- **Feature Classification**: Currently supports dense layers for feature classification.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Austin-TheTrueShinobi/UTokyo_CNN_MRI.git
   cd UTokyo_CNN_MRI
   ```

2. Install the required packages:
   ```bash
   pip install tensorflow
   ```

## VSCode Setup

1. **Installation**: If you haven't installed VSCode, download it from [Visual Studio Code](https://code.visualstudio.com/).

2. **Recommended Extensions**:
   - **Python**: Provides enhanced support for Python, including debugging, linting, IntelliSense, etc.
   - **TensorFlow Snippets**: Offers useful snippets for TensorFlow.
   - **Markdown All in One**: Essential for editing `README.md` files.
   - **GitLens**: Supercharge the Git capabilities built into Visual Studio Code.

3. **Environment Setup**:
   - Ensure you have Python 3.x installed.
   - Set up a virtual environment:
     ```bash
     python -m venv venv
     ```
   - Activate the virtual environment:
     - Windows: `.\venv\Scripts\activate`
     - MacOS/Linux: `source venv/bin/activate`
   - Open the project folder in VSCode.
   - When prompted, select the Python interpreter located inside the `venv` directory.

## MRI Data Setup from IDA

The Image and Data Archive (IDA) at the Laboratory of Neuro Imaging (LONI) is a data repository that hosts a variety of neuroimaging datasets. To utilize MRI data from IDA for our project:

### Registration and Data Access:

1. **Create an Account**:
   - Visit the [IDA database](https://ida.loni.usc.edu/) and click on `Register`.
   - Fill in the required details and complete the registration process.
   - Note: Some datasets may require additional permissions or agreements to access.

2. **Browse and Select Dataset**:
   - Once logged in, navigate to the `Collections` tab.
   - Browse through available datasets or use the search functionality to find specific MRI datasets.
   - Click on the desired dataset to view more details.

3. **Download Data**:
   - Within the dataset page, you'll find download options. Depending on the dataset, you might find raw MRI scans, pre-processed data, or both.
   - Download the MRI data files, which are typically in NIfTI format (`.nii` or `.nii.gz`).

### Preparing Data for the Project:

4. **Organize Downloaded Data**:
   - Create a folder named `data` within the project directory.
   - Extract and place the downloaded MRI data into the `data` folder. Ensure you maintain a structured directory if there are multiple categories or types of scans.

5. **Data Loading in the Script**:
   - The `main.py` script should be set up to load data from the `data` directory. If the MRI data has multiple categories or is divided into training and testing sets, ensure the script's data loading section reflects this structure.
   - For NIfTI files, you might need additional libraries like `nibabel` to read the data:
     ```bash
     pip install nibabel
     ```

6. **Pre-processing**:
   - MRI data often requires pre-processing before it can be used for training. This might include normalization, resizing, or augmentation.
   - Ensure the `main.py` script or any other pre-processing script you use is tailored to handle the specific characteristics of the MRI data from IDA.

### Usage:

7. **Run the Model**:
   - With the data in place, you can proceed to run the model:
     ```bash
     python main.py
     ```

8. **Model Training**:
   - The script will load the MRI data, preprocess it if necessary, and then train the CNN model on it.
   - Monitor the training process, and once completed, evaluate the model's performance on any test or validation data you've set aside.

---

## Contributing

1. Fork the repository on GitHub.
2. Clone the forked repository to your machine.
3. Make your changes and commit them to your fork.
4. Push your changes to GitHub.
5. Submit a pull request to the main repository.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

**Note**: Replace `your_username` with your actual GitHub username in the clone URL.
