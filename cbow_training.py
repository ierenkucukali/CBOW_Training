import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import os
import random

# ------------------------------
# Settings (you can modify these)
# ------------------------------
TRAIN_FILE = "your_train_tokens_file"
VALID_FILE = "your_validation_tokens_file"
TEST_FILE = "your_test_tokens_file"

EMBEDDING_DIM = 128
BATCH_SIZE = 512  #reduce for smaller datasets
EPOCHS = 10
CONTEXT_SIZE = 3
LEARNING_RATE = 0.001
WEIGHT_STRATEGY = "fixed"  # options: "fixed", "scalar", "vector"

def load_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        tokens = f.read().split()
    return tokens

def build_vocab(tokens, min_count=1):
    counter = Counter(tokens)
    vocab = {word: idx for idx, (word, count) in enumerate(counter.items()) if count >= min_count}
    idx_to_word = {idx: word for word, idx in vocab.items()}
    return vocab, idx_to_word

# ------------------------------
# Dataset Class
# ------------------------------
class CBOWDataset(Dataset):
    def __init__(self, tokens, vocab, context_size):
        self.data = []
        self.vocab = vocab
        self.context_size = context_size
        for i in range(context_size, len(tokens) - context_size):
            context = [tokens[i + j] for j in range(-context_size, context_size + 1) if j != 0]
            target = tokens[i]
            if all(word in vocab for word in context + [target]):
                context_indices = [vocab[w] for w in context]
                target_index = vocab[target]
                self.data.append((context_indices, target_index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])

# ------------------------------
# CBOW Model
# ------------------------------
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, strategy):
        super(CBOWModel, self).__init__()
        self.strategy = strategy
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        if strategy == "scalar":
            self.position_weights = nn.Parameter(torch.ones(2 * context_size))
        elif strategy == "vector":
            self.position_weights = nn.Parameter(torch.ones(2 * context_size, embedding_dim))
        else:
            self.register_buffer('fixed_weights', torch.tensor([1, 2, 3, 3, 2, 1], dtype=torch.float32))
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_idxs):
        embeds = self.embeddings(context_idxs)
        if self.strategy == "fixed":
            weighted = embeds * self.fixed_weights.unsqueeze(0).unsqueeze(-1)
            context_embed = weighted.sum(dim=1)
        elif self.strategy == "scalar":
            weighted = embeds * self.position_weights.unsqueeze(0).unsqueeze(-1)
            context_embed = weighted.sum(dim=1)
        elif self.strategy == "vector":
            weighted = embeds * self.position_weights.unsqueeze(0)
            context_embed = weighted.sum(dim=1)
        else:
            raise ValueError("Unknown strategy")
        out = self.linear(context_embed)
        return out

# ------------------------------
# Training Function
# ------------------------------
def train_model(model, train_loader, valid_loader, optimizer, criterion, device):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for contexts, targets in train_loader:
            contexts, targets = contexts.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(contexts)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {total_loss/len(train_loader):.4f}")
        validate_model(model, valid_loader, criterion, device)

# ------------------------------
# Validation Function
# ------------------------------
def validate_model(model, valid_loader, criterion, device):
    model.eval()
    if len(valid_loader) == 0:
        print("Warning: Validation set is empty. Skipping validation.")
        return
    total_loss = 0
    with torch.no_grad():
        for contexts, targets in valid_loader:
            contexts, targets = contexts.to(device), targets.to(device)
            output = model(contexts)
            loss = criterion(output, targets)
            total_loss += loss.item()
    print(f"Validation Loss: {total_loss/len(valid_loader):.4f}")


# ------------------------------
# Word Vector Analysis Class
# ------------------------------
class WordVector:
    def __init__(self, model, idx_to_word):
        self.embeddings = model.embeddings.weight.data.cpu().numpy()
        self.idx_to_word = idx_to_word

    def most_similar(self, word_idx, top_k=5):
        target_vec = self.embeddings[word_idx]
        sims = np.dot(self.embeddings, target_vec) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(target_vec) + 1e-10)
        sorted_idx = np.argsort(-sims)
        return [(self.idx_to_word[idx], sims[idx]) for idx in sorted_idx[1:top_k+1]]

    def analogy(self, word1_idx, word2_idx, word3_idx, top_k=1):
        vec = self.embeddings[word2_idx] - self.embeddings[word1_idx] + self.embeddings[word3_idx]
        sims = np.dot(self.embeddings, vec) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(vec) + 1e-10)
        sorted_idx = np.argsort(-sims)
        return [(self.idx_to_word[idx], sims[idx]) for idx in sorted_idx[:top_k]]

# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tokens = load_tokens(TRAIN_FILE)
    valid_tokens = load_tokens(VALID_FILE)
    test_tokens = load_tokens(TEST_FILE)

    vocab, idx_to_word = build_vocab(train_tokens)

    train_dataset = CBOWDataset(train_tokens, vocab, CONTEXT_SIZE)
    valid_dataset = CBOWDataset(valid_tokens, vocab, CONTEXT_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

    model = CBOWModel(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, WEIGHT_STRATEGY).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, valid_loader, optimizer, criterion, device)

    # Example usage of WordVector
    # word_vector_tool = WordVector(model, idx_to_word)
    # print(word_vector_tool.most_similar(vocab['example_word']))
    # print(word_vector_tool.analogy(vocab['man'], vocab['king'], vocab['woman']))
