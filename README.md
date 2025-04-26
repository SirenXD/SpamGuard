# SpamGuard

SpamGuard is an AI language model powered by [Google's RETVEC](https://github.com/google-research/retvec) text vectorizer to classify emails as spam.

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Predicting](#predicting)
- [Requirements](#requirements)
- [Recommended Requirements](#recommended-requirements)
- [Recommended Resources](#recommended-resources)

---

## Features

- **High Character Coverage**: Uses [Google's RETVEC](https://github.com/google-research/retvec) text vectorizer to provide encoding of the full UTF-8 character set.
- **Ease of Integration**: Can be imported and used as a Python class.
- **Extensible**: A lot of the functionality configuration is not forced, such that it is very flexible on how you want to provide or process data. Although the intent was to create an AI model for spam classification, this same model could easily be used for other text classification applications as a result.

---

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the `/src` folder from the repository.

---

## Usage

### Training

To train the model, use the `CSVDataGenerator` class to create training and validation data generators.

```python
from DataGenerator import CSVDataGenerator
from SpamGuard import SpamGuard

model = SpamGuard()

# Create a new model (for training from scratch)
model.create_new_model()

# or load a pre-trained model (for fine-tuning or continuing training)
model.load_model("../path/to/trained/model")

# Create training and validation data generators
train_gen = CSVDataGenerator('./path_to_training_csv', batch_size=32, num_classes=2)
val_gen = CSVDataGenerator('./path_to_validation_csv', batch_size=32, num_classes=2, shuffle=False)

# Train the model
model.train(train_gen, val_gen)

# Save the trained model
model.save_model("SpamGuard_trained")
```

---

### Predicting

To use the model for predictions:

```python
from SpamGuard import SpamGuard

# Load the trained model
model = SpamGuard()
model.load_model("../path/to/trained/model")

# Predict spam or not spam
classification, probability = model.predict("text to predict")
print(f"Classification: {classification}, Probability: {probability}%")
```

For detailed usage, refer to the example notebook:  
`/notebooks/SpamGuard_Example.ipynb`

---

## Requirements

- Python 3.8
- TensorFlow 2.8
- [RETVEC](https://github.com/google-research/retvec)
- Pandas
- Numpy
- matplotlib

---

## Recommended Requirements

- TensorFlow 2.8 GPU ([Jupyter Docker image](https://hub.docker.com/layers/tensorflow/tensorflow/2.8.0-gpu-jupyter/images/sha256-56677a6a426e87cba3dc5ae8eb82cffb6c76af7dd3683fe9caaba28dcd2a8958) preferred)
- BeautifulSoup4 for data preprocessing. See `DataPreprocessingExamples.ipynb` for example usage.

---

## Recommended Resources

The following datasets can be used to train or evaluate the SpamGuard model:
- [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/): A collection of spam and ham emails intended for training and testing spam detection models.
- [Enron Email Dataset](https://www.cs.cmu.edu/~enron/): A large dataset of real emails from the Enron corporation.
- [Untroubled.org Spam Archive](https://untroubled.org/spam/): A regularly updated archive of spam emails.
