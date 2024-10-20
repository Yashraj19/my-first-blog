# CSE256 PA1: Sentiment Classification Models

This repository contains code for training sentiment classification models using different architectures:

- **Bag-of-Words (BOW) Model**
- **Deep Averaging Network (DAN)**
- **Subword Deep Averaging Network (SUBWORDDAN)**

The main script, `main.py`, handles data loading, model training, evaluation, and plotting of accuracy over epochs.

## Table of Contents

- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
  - [Sentiment Data](#sentiment-data)
  - [GloVe Embeddings (Optional)](#glove-embeddings-optional)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Running the BOW Model](#running-the-bow-model)
  - [Running the DAN Model](#running-the-dan-model)
    - [Using GloVe Embeddings](#using-glove-embeddings)
    - [Using Random Embeddings](#using-random-embeddings)
  - [Running the SUBWORDDAN Model](#running-the-subworddan-model)
- [Examples](#examples)
- [Output](#output)
- [Notes](#notes)
- [License](#license)

---

## Requirements

- **Python 3.7** or higher
- **PyTorch**
- **matplotlib** (for plotting accuracy)
- **sentencepiece** (for subword tokenization in SUBWORDDAN)
- **Additional Python packages**: `numpy`, `argparse`, `os`, `datetime`

Install the required packages using:

```bash
pip install torch matplotlib sentencepiece numpy argparse
```

## Data Preparation

### Sentiment Data

The models expect the following data files:

- `data/train.txt`: Training data
- `data/dev.txt`: Development (validation) data

Ensure that the files are placed in a `data/` directory relative to `main.py`.

### GloVe Embeddings (Optional)

For the DAN model using GloVe embeddings, download and prepare GloVe vectors:

1. **Download GloVe embeddings**:

   - Download from the [GloVe website](https://nlp.stanford.edu/projects/glove/).
   - Use the `glove.6B.zip` file (822 MB).

2. **Extract the embeddings**:

   - Unzip and extract `glove.6B.300d.txt`.

3. **Place the embeddings**:

   - Place `glove.6B.300d.txt` in the `data/` directory.

4. **Relativization (Optional)**:

   - Relativize embeddings to include only words in your dataset to reduce memory usage.
   - Implement a script to extract relevant embeddings if necessary.

---

## Usage

### Command-Line Arguments

- `--model`: (**Required**) Specify the model type (`BOW`, `DAN`, or `SUBWORDDAN`).
- `--random_embeddings`: (**Optional**) Use random embeddings for the DAN model.
- `--vocab_size`: (**Optional**) Vocabulary size for SUBWORDDAN (default: `5000`).

**Basic Syntax:**

```bash
python main.py --model <MODEL_TYPE> [--random_embeddings] [--vocab_size <VOCAB_SIZE>]
```

### Running the BOW Model

To train and evaluate the **BOW** model:

```bash
python main.py --model BOW
```

Ensure that `BOWmodels.py` and other related modules are present.

### Running the DAN Model

#### Using GloVe Embeddings

To train and evaluate the **DAN** model with GloVe embeddings:

```bash
python main.py --model DAN
```

Ensure that `data/glove.6B.300d.txt` exists and `DANmodels.py` is available.

#### Using Random Embeddings

To train and evaluate the **DAN** model with random embeddings:

```bash
python main.py --model DAN --random_embeddings
```

### Running the SUBWORDDAN Model

To train and evaluate the **SUBWORDDAN** model:

```bash
python main.py --model SUBWORDDAN --vocab_size 8000
```

Adjust `--vocab_size` as needed.

---

## Examples

**BOW Model:**

```bash
python main.py --model BOW
```

**DAN Model with GloVe Embeddings:**

```bash
python main.py --model DAN
```

**DAN Model with Random Embeddings:**

```bash
python main.py --model DAN --random_embeddings
```

**SUBWORDDAN Model with Vocabulary Size 10000:**

```bash
python main.py --model SUBWORDDAN --vocab_size 10000
```

---

## Output

- Training and validation accuracy is printed every 5 epochs.
- Accuracy plots are saved as PNG files with timestamped filenames, e.g., `dan_glove_accuracy_20231105_123456.png`.

---

## Notes

- **Device Selection**: The script uses GPU if available; otherwise, it defaults to CPU.
- **Custom Modules**: Ensure that `BOWmodels.py`, `DANmodels.py`, and `sentiment_data.py` are in the same directory or accessible via your Python path.
- **Data Files**: Ensure all data files are in the correct directories.
- **SentencePiece Model**: For SUBWORDDAN, the SentencePiece model is trained if not already present.
- **Adjustable Parameters**: Modify hyperparameters like batch size, learning rate, or number of epochs directly in `main.py` if needed.
- **Error Handling**: Check file paths and module imports if you encounter errors.

---

*This README provides instructions on how to run the sentiment classification models using the provided `main.py` script. Ensure that all dependencies are installed and data files are correctly placed before executing the script.*
