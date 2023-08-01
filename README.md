# UTokyo_CNN_MRI
This repository is for MRI and MNIST data CNN controller and related experimental scripts. Some are out of date and are strictly for informational and historical records.

---

# Modular CNN for MRI and MNIST Data

This repository contains a modular Convolutional Neural Network (CNN) designed to test different pooling techniques, gradient functions, and feature classification strategies for MRI and MNIST data.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
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
   git clone https://github.com/your_username/modular-cnn.git
   cd modular-cnn
   ```

2. Install the required packages:
   ```bash
   pip install tensorflow
   ```

## Usage

1. Modify the `main.py` script if needed, especially if you want to load custom data or adjust model parameters.

2. Run the model:
   ```bash
   python main.py
   ```

3. The model summary will be displayed, and you can further train the model on your data.

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
