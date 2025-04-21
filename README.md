# CBOW Word Embedding Training with Position-Dependent Weighting

This project implements a Continuous Bag of Words (CBOW) model for learning word embeddings from a tokenized text corpus, enhanced with **position-dependent weighting strategies** to improve performance.

---

## 🎯 Aim of the Project

The purpose of this project is to:
- Train a CBOW model using tokenized text data.
- Improve the quality of word embeddings by applying different context weighting strategies:
  - **Fixed scalar weights** (e.g., words closer to the center word get higher weight).
  - **Learnable scalar weights** (train one weight per position).
  - **Learnable vector weights** (train a full vector per position).
- Analyze learned word embeddings using cosine similarity.
- Provide a clean, generalizable PyTorch implementation that can be used with any tokenized dataset.

---

## 📚 Dataset Requirements

To run this code, you must prepare your own tokenized dataset files:

- You need **three files**:
  - `your_train_tokens_file`
  - `your_validation_tokens_file`
  - `your_test_tokens_file`

- Each file should be a **plain text file** where:
  - All text is **tokenized**.
  - **Tokens are separated by spaces**.
  - Example content:
    ```
    the cat sat on the mat the dog barked at the cat
    ```
- Typical datasets you can use:
  - Wikipedia dumps (tokenized)
  - News articles (tokenized)
  - Books corpus (tokenized)

**Important:**  
- Files must contain enough tokens (at least several hundred) to allow proper CBOW context windowing.
- The context window size (default 3) requires enough tokens around each center word.
- Small datasets may not be enough unless you reduce context size.

---

## ⚙️ How to Set Up

1. Modify the paths in `cbow_training.py` to point to your files:
   ```python
   TRAIN_FILE = "your_train_tokens_file"
   VALID_FILE = "your_validation_tokens_file"
   TEST_FILE = "your_test_tokens_file"
