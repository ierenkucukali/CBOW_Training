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

2. Adjust hyperparameters if needed:
   ```python
   EMBEDDING_DIM = 128
   BATCH_SIZE = 512
   EPOCHS = 10
   CONTEXT_SIZE = 3
   LEARNING_RATE = 0.001
   WEIGHT_STRATEGY = "fixed"  # options: "fixed", "scalar", "vector"

## 🚀 How to Run

  Install dependencies. Then run the script
  1. ```python
    pip install torch numpy
    python cbow_training.py


## 🔍 Features of This Implemetation
Three weighting strategies:
- Fixed scalar (hardcoded)
- Learnable scalar (train a scalar per position)
- Learnable vector (train full embedding per position)

Training loss and validation loss printed per epoch.
Robust: Safe handling if validation dataset is too small.
Word similarity and analogy evaluation (optional WordVector class).

## Word Embedding Analysis (Similarity and Analogy)
Finding Most Similar Words: After training the model, you can use the WordVector class to analyze the learned embeddings:
1. This will return a list of the 5 most similar words based on cosine similarity.
   ```python
    # Initialize WordVector analysis
    word_vector_tool = WordVector(model, idx_to_word)
    # Find top-5 most similar words to a given word
    word_vector_tool.most_similar(vocab['example_word'], top_k=5)
2. Solving Word Analogies: This will return the most likely word vector result for the analogy.
   ```python
    # Example analogy: king - man + woman = queen
    word_vector_tool.analogy(vocab['man'], vocab['king'], vocab['woman'], top_k=1)


## 💡 Future Improvements and Ideas
Add learning rate scheduler
Hyperparameter optimization
Visualization of embeddings (PCA or t-SNE)
Early stopping on validation loss
Batch context window adjustment
  

