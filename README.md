# Spam Classification with Predictive Coding Network (PCN)

This project implements a spam classification system using a Predictive Coding Network (PCN) with three hidden layers. The PCN is built using the `ngclearn` library and trained on the [Spambase dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset) from Kaggle.

## Overview

This project demonstrates:

*   **Predictive Coding Network (PCN):** A neural network architecture based on the principles of predictive coding, where each layer attempts to predict the activity of the layer below it.
*   **Hebbian Learning:** The synaptic weights of the PCN are adapted using Hebbian learning, a biologically plausible learning rule.
*   **`ngclearn` Library:**  The project leverages the `ngclearn` library for building and simulating neural networks.
*   **Spam Classification:**  The PCN is trained to classify SMS messages as either "spam" or "ham" (non-spam).
*   **Data Preprocessing:** Text data is preprocessed using techniques such as TF-IDF vectorization, stop word removal, and lemmatization.

## Project Structure

The project is organized as follows:

*   `data/`: Contains the Spambase dataset (`spam.csv`).
*   `json_files/`: (Potentially) contains configuration files for the model (if used).
*   `models/`: Contains the PCN model definition (`pcn_model.py`) and the main training script (`main.py`).
    *   `pcn_model.py`: Defines the `PCN` class, which implements the predictive coding network.
    *   `main.py`: The main script for training and evaluating the PCN model.
*   `scripts/`: Contains the data preprocessing script (`preprocess.py`).
    *   `preprocess.py`: Defines the `preprocess` function for loading, cleaning, and preparing the data for training.
*   `requirements.txt`: Lists the Python packages required to run the project.
*   `README.md`: This file.

## Requirements

To run this project, you'll need the following:

*   Python 3.7+
*   The packages listed in `requirements.txt`.  Install them using:

    ```bash
    pip install -r requirements.txt
    ```

    This typically includes:
    *   `jax`
    *   `jaxlib` (Make sure the correct jaxlib is installed based on if you are using GPU or CPU, and what CUDA version you are using)
    *   `scikit-learn`
    *   `nltk`
    *   `pandas`
    *   `matplotlib`
    *   `ngclearn`
    *   `scipy`
    *   `tqdm` (recommended)
*   **NLTK Data:** The `nltk` library requires downloading some data resources. Download these *before* running the main script by running the following commands in a Python interpreter:

    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    ```

## Usage

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Miheret-Girmachew/TrainHub.git
    cd spam_classification
    ```

2.  **Install the requirements:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Download NLTK data:** Run the Python code snippet above to download the required NLTK data.

4.  **Run the main script:**

    ```bash
    python models/main.py
    ```

    The script will train the PCN model and print the training and validation accuracy. It will also save the trained model parameters and plot the learning curve.

## Model Architecture

The PCN consists of the following layers:

*   `z0`: Input layer (RateCell), representing the TF-IDF vectorized SMS messages.
*   `z1`: First hidden layer (RateCell), learning the first level of abstract features.
*   `z2`: Second hidden layer (RateCell), learning higher-level features.
*   `z3`: Third hidden layer (RateCell), learning even higher-level features.
*   `z4`: Output layer (RateCell), representing the predicted class (spam or ham).
*   `e1`, `e2`, `e3`, `e4`: Error units associated with each hidden layer, quantifying the prediction error.
*   `W1`, `W2`, `W3`, `W4`: Hebbian synapses connecting the layers, adapted using Hebbian learning.
*   `E2`, `E3`, `E4`: Static synapses used for feedback connections.

## Hyperparameters

The following hyperparameters can be adjusted in the `models/main.py` file:

*   `hidden_dim1`, `hidden_dim2`, `hidden_dim3`: Dimensionality of the hidden layers.
*   `n_iter`: Number of training epochs.
*   `mb_size`: Mini-batch size.
*   `eta`: Hebbian learning rate.
*   `patience`: Early stopping patience.
*   `act_fx`: Activation function for the hidden layers (e.g., "tanh", "relu").
*   `wlb`, `wub`: Weight bounds for the Hebbian synapses.

## Tuning the Model

The model's performance can be improved by tuning the hyperparameters. Consider experimenting with different values for the hyperparameters to find the optimal configuration. Some strategies include:

*   **Manual Tuning:** Manually change the hyperparameter values and re-run the training.
*   **Grid Search:** Define a grid of hyperparameter values and train/evaluate the model for every combination.
*   **Random Search:** Randomly sample hyperparameter values from a distribution and train/evaluate.
*   **Bayesian Optimization:** Use Bayesian methods to intelligently explore the hyperparameter space.



