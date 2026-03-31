

import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1. Reproducibility
# ============================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. Corpus (1000+ words)
# ============================================================
CORPUS = """
Machine learning is a field of computer science that studies how systems can learn from data.
A learning system receives examples, discovers patterns, and uses those patterns to make predictions.
Neural networks are models composed of layers of interconnected units.
Each layer transforms its input into a new representation.
By combining many layers, a neural network can represent complicated functions.

In natural language processing, text must be converted into numbers before a model can process it.
One common strategy is to map each word to a vector.
This vector is called a word embedding.
A word embedding is a numerical representation of a word in a continuous space.
Words with similar meanings often have embeddings that are close to one another.

Traditional one hot vectors represent each word by a long sparse vector.
In a one hot vector, only one position is equal to one and all other positions are equal to zero.
This representation is simple, but it does not capture semantic similarity.
For example, the words model and network are different words, but a one hot representation does not express any meaningful relationship between them.
Dense embeddings solve this problem by placing words in a learned vector space.

Modern transformers operate on sequences of embeddings.
Each token or word is first converted to an embedding.
Then positional information is added to preserve order.
After that, self attention allows each position to interact with other positions in the sequence.
This helps the model capture context, dependency, and meaning.

A transformer encoder consists of multiple layers.
Each layer includes multi head self attention, residual connections, normalization, and a feed forward network.
Multi head attention splits the embedding into smaller parts called heads.
Each head can learn a different type of relationship.
Some heads may focus on nearby words.
Other heads may focus on broader context or repeated structure.

The feed forward network is applied independently at each position.
It expands the representation to a hidden dimension, applies a nonlinearity, and projects the result back.
Residual connections help the gradient move through the network.
Layer normalization stabilizes training.
Together, these design choices make transformers effective and trainable.

A corpus is a body of text used for training or analysis.
In this program, the corpus is intentionally simple and repetitive.
It is designed for classroom use and demonstration.
The text discusses machine learning, embeddings, transformers, optimization, and prediction.
Because the corpus repeats important technical words, the resulting vocabulary is manageable and the model can learn visible patterns.

Training requires input examples and target outputs.
In next word prediction, the input is a sequence of words and the target is the word that comes next.
For example, if the sequence is the model learns from, the target could be data.
By observing many such examples, the network learns conditional structure in the corpus.
This setup is similar to language modeling, although our example is much smaller than a production system.

Word2Vec learns vector representations by using local context.
In the skip gram version, the model tries to predict surrounding words from a center word.
In the CBOW version, the model tries to predict a center word from surrounding words.
Both approaches use distributional structure in the corpus.
The general idea is that words used in similar contexts should have similar vectors.

We use Word2Vec in this script because the goal is word level modeling.
Word2Vec is more natural than a sentence embedding model when the units of interest are individual words.
The embedding dimension can be set directly.
Therefore, if we want an embedding dimension of four, eight, or sixteen, we can train Word2Vec with that exact dimension.
This avoids the extra step of applying dimensionality reduction after the embeddings are created.

During training, the transformer receives sequences of token indices.
It converts them into dense vectors using an embedding table.
In this script, that embedding table is initialized from the Word2Vec vectors.
The embedding layer may remain trainable, so the transformer can adapt the initial embeddings to the next word prediction task.
This gives us a pretrained starting point and task specific fine tuning at the same time.

Optimization adjusts the parameters to reduce the loss.
We use cross entropy loss for next word classification.
The model outputs one logit for every word in the vocabulary.
The correct next word index is used as the target label.
The optimizer updates the parameters through backpropagation.
Over repeated epochs, the model can learn the local structure of the corpus.

Evaluation can be performed by measuring the average loss or by checking whether the predicted word matches the true next word.
Accuracy is easy to understand, especially in classroom demonstrations.
Students can observe how the model improves as the number of epochs increases.
They can also test how performance changes when the embedding dimension changes.

There are several reasons to study this kind of small experiment.
First, it makes the transformer architecture concrete.
Second, it connects corpus based embeddings with sequence modeling.
Third, it shows how initialization affects learning.
Fourth, it creates an end to end pipeline from text corpus to vocabulary, from vocabulary to embeddings, from embeddings to transformer inputs, and from transformer outputs to predictions.

In larger systems, text is usually tokenized into subwords rather than full words.
This avoids out of vocabulary problems and handles rare terms better.
However, word level modeling is easier to explain to beginners.
Each token corresponds directly to a visible word in the corpus.
This makes debugging and interpretation simpler.

The attention mechanism computes relationships between positions.
Query vectors are compared with key vectors to produce attention scores.
The attention scores are normalized into weights.
Those weights are used to combine value vectors.
The result is a contextual representation that reflects the influence of other words in the sequence.
This process is repeated in every layer and every head.

The final hidden states can be used for many tasks.
For next word prediction, we take the hidden state at the last input position and map it to vocabulary logits.
The highest logit corresponds to the predicted next word.
With a small corpus and a compact model, the predictions are not perfect, but they often reveal clear learned patterns.

If the embedding dimension is four, then the number of heads must divide four.
Valid choices are one, two, and four.
If the embedding dimension is eight, then valid choices are one, two, four, and eight.
This requirement exists because each attention head receives an equal portion of the embedding vector.

Practical programming details are important.
The corpus must be cleaned and tokenized consistently.
A vocabulary must be built.
Special tokens such as padding and unknown should be included.
Training sequences must be created carefully to preserve order.
The model and tensors must be placed on the same device.
During evaluation, gradient computation should be disabled.
Reproducibility should be improved by setting random seeds.

This script is intended to be clear rather than large.
It demonstrates the key ideas in a compact form.
It uses a real word embedding model, builds a real transformer, and performs real gradient based training.
Because the implementation is self contained, it is suitable for classroom examples, homework preparation, or personal experimentation.

By changing a few parameters, one can explore many ideas.
One can change the corpus, the sequence length, the embedding dimension, the number of heads, the number of layers, the number of epochs, and the learning rate.
These experiments help develop intuition about language modeling and representation learning.
Even though the program is small, it touches many core concepts in modern deep learning.
"""


# ============================================================
# 3. Configuration
# ============================================================
@dataclass
class Config:
    target_dim: int = 4            # Word2Vec embedding dimension
    seq_len: int = 6
    num_heads: int = 2             # must divide target_dim
    num_layers: int = 2
    ff_hidden_dim: int = 32
    dropout: float = 0.1
    batch_size: int = 32
    num_epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_ratio: float = 0.8
    min_word_freq: int = 1
    freeze_initial_embeddings: bool = False

    # Word2Vec settings
    w2v_window: int = 5
    w2v_sg: int = 1                # 1 = skip-gram, 0 = CBOW
    w2v_epochs: int = 150
    w2v_min_count: int = 1
    w2v_workers: int = 1

    seed: int = 42


# ============================================================
# 4. Tokenization and vocabulary
# ============================================================
def tokenize_words(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9']+", text)
    return tokens


def split_text_into_sentences(text: str) -> List[str]:
    parts = re.split(r"[.!?]+", text.lower())
    return [p.strip() for p in parts if p.strip()]


def build_vocab(tokens: List[str], min_freq: int = 1) -> Tuple[Dict[str, int], Dict[int, str]]:
    counts = Counter(tokens)

    vocab = ["<PAD>", "<UNK>"]
    for word, freq in sorted(counts.items()):
        if freq >= min_freq:
            vocab.append(word)

    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for word, i in word_to_idx.items()}
    return word_to_idx, idx_to_word


def numericalize(tokens: List[str], word_to_idx: Dict[str, int]) -> List[int]:
    unk_idx = word_to_idx["<UNK>"]
    return [word_to_idx.get(tok, unk_idx) for tok in tokens]


# ============================================================
# 5. Word2Vec training and embedding matrix
# ============================================================
def train_word2vec_on_corpus(corpus_text: str, config: Config) -> Word2Vec:
    raw_sentences = split_text_into_sentences(corpus_text)
    tokenized_sentences = [tokenize_words(s) for s in raw_sentences]
    tokenized_sentences = [s for s in tokenized_sentences if len(s) > 0]

    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=config.target_dim,
        window=config.w2v_window,
        min_count=config.w2v_min_count,
        workers=config.w2v_workers,
        sg=config.w2v_sg,
        epochs=config.w2v_epochs,
        seed=config.seed
    )
    return model


def build_embedding_matrix_from_word2vec(
    word_to_idx: Dict[str, int],
    w2v_model: Word2Vec,
    embed_dim: int
) -> np.ndarray:
    vocab_size = len(word_to_idx)
    matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)

    # PAD stays zero vector
    pad_idx = word_to_idx["<PAD>"]
    matrix[pad_idx] = np.zeros(embed_dim, dtype=np.float32)

    for word, idx in word_to_idx.items():
        if word == "<PAD>":
            continue
        elif word in w2v_model.wv:
            matrix[idx] = w2v_model.wv[word]
        else:
            matrix[idx] = np.random.normal(0.0, 0.02, size=(embed_dim,)).astype(np.float32)

    return matrix


# ============================================================
# 6. Dataset for next-word prediction
# ============================================================
class WordSequenceDataset(Dataset):
    def __init__(self, token_ids: List[int], seq_len: int):
        self.X = []
        self.y = []

        for i in range(len(token_ids) - seq_len):
            self.X.append(token_ids[i:i + seq_len])
            self.y.append(token_ids[i + seq_len])

        self.X = np.asarray(self.X, dtype=np.int64)
        self.y = np.asarray(self.y, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.X[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


# ============================================================
# 7. Positional encoding
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# ============================================================
# 8. Transformer model for next-word prediction
# ============================================================
class TransformerWordPredictor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_hidden_dim: int,
        dropout: float,
        max_seq_len: int,
        initial_embedding_matrix: np.ndarray = None,
        freeze_embeddings: bool = False
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        if initial_embedding_matrix is not None:
            if initial_embedding_matrix.shape != (vocab_size, embed_dim):
                raise ValueError(
                    f"Expected initial_embedding_matrix shape {(vocab_size, embed_dim)}, "
                    f"got {initial_embedding_matrix.shape}"
                )
            with torch.no_grad():
                self.embedding.weight.copy_(torch.tensor(initial_embedding_matrix, dtype=torch.float32))

        self.embedding.weight.requires_grad = not freeze_embeddings

        self.positional_encoding = PositionalEncoding(embed_dim, max_len=max_seq_len + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len] token ids
        returns logits: [batch, vocab_size]
        """
        emb = self.embedding(x)          # [B, S, D]
        emb = self.positional_encoding(emb)
        h = self.encoder(emb)            # [B, S, D]
        last_h = self.norm(h[:, -1, :])  # [B, D]
        logits = self.output_layer(last_h)
        return logits


# ============================================================
# 9. Training utilities
# ============================================================
def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)
            acc = compute_accuracy(logits, yb)

            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            total_acc += acc * batch_size
            total_count += batch_size

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_acc / max(total_count, 1)
    return avg_loss, avg_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    device: torch.device
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, config.num_epochs + 1):
        model.train()

        total_loss = 0.0
        total_acc = 0.0
        total_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            acc = compute_accuracy(logits, yb)
            batch_size = xb.size(0)

            total_loss += loss.item() * batch_size
            total_acc += acc * batch_size
            total_count += batch_size

        train_loss = total_loss / max(total_count, 1)
        train_acc = total_acc / max(total_count, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d}/{config.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )


# ============================================================
# 10. Prediction helper
# ============================================================
def predict_next_words(
    model: nn.Module,
    context_words: List[str],
    word_to_idx: Dict[str, int],
    idx_to_word: Dict[int, str],
    device: torch.device,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    unk_idx = word_to_idx["<UNK>"]
    x = [word_to_idx.get(w.lower(), unk_idx) for w in context_words]
    x = torch.tensor([x], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        values, indices = torch.topk(probs, k=top_k)

    results = []
    for v, idx in zip(values.cpu().numpy(), indices.cpu().numpy()):
        results.append((idx_to_word[int(idx)], float(v)))
    return results


# ============================================================
# 11. Main
# ============================================================
def main():
    config = Config()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Step 1: Tokenize corpus into WORDS")
    print("=" * 70)
    tokens = tokenize_words(CORPUS)
    print(f"Corpus word count: {len(tokens)}")
    print("First 30 tokens:")
    print(tokens[:30])

    print("\n" + "=" * 70)
    print("Step 2: Build vocabulary")
    print("=" * 70)
    word_to_idx, idx_to_word = build_vocab(tokens, min_freq=config.min_word_freq)
    vocab_size = len(word_to_idx)
    print(f"Vocabulary size: {vocab_size}")

    token_ids = numericalize(tokens, word_to_idx)

    print("\n" + "=" * 70)
    print("Step 3: Train Word2Vec on the corpus")
    print("=" * 70)
    w2v_model = train_word2vec_on_corpus(CORPUS, config)
    print(f"Word2Vec vocabulary size: {len(w2v_model.wv)}")
    print(f"Word2Vec vector size: {w2v_model.vector_size}")

    print("\nExample nearest words to 'model' (if available):")
    if "model" in w2v_model.wv:
        try:
            for word, score in w2v_model.wv.most_similar("model", topn=5):
                print(f"   {word:15s} {score:.4f}")
        except Exception:
            print("   Could not compute nearest words.")
    else:
        print("   Word 'model' not in Word2Vec vocabulary.")

    print("\n" + "=" * 70)
    print("Step 4: Build initial embedding matrix from Word2Vec")
    print("=" * 70)
    initial_embedding_matrix = build_embedding_matrix_from_word2vec(
        word_to_idx=word_to_idx,
        w2v_model=w2v_model,
        embed_dim=config.target_dim
    )
    print(f"Initial embedding matrix shape: {initial_embedding_matrix.shape}")

    print("\n" + "=" * 70)
    print("Step 5: Build next-word dataset")
    print("=" * 70)
    dataset = WordSequenceDataset(token_ids, seq_len=config.seq_len)
    print(f"Number of sequence examples: {len(dataset)}")

    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        indices,
        train_size=config.train_ratio,
        shuffle=True,
        random_state=config.seed
    )

    train_subset = torch.utils.data.Subset(dataset, train_idx.tolist())
    val_subset = torch.utils.data.Subset(dataset, val_idx.tolist())

    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)

    print(f"Train examples: {len(train_subset)}")
    print(f"Validation examples: {len(val_subset)}")

    print("\n" + "=" * 70)
    print("Step 6: Create Transformer model")
    print("=" * 70)
    model = TransformerWordPredictor(
        vocab_size=vocab_size,
        embed_dim=config.target_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_hidden_dim=config.ff_hidden_dim,
        dropout=config.dropout,
        max_seq_len=config.seq_len,
        initial_embedding_matrix=initial_embedding_matrix,
        freeze_embeddings=config.freeze_initial_embeddings
    ).to(device)

    print(model)

    print("\n" + "=" * 70)
    print("Step 7: Train model")
    print("=" * 70)
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    print("\n" + "=" * 70)
    print("Step 8: Show sample predictions")
    print("=" * 70)
    example_positions = [0, 20, 50]

    for pos in example_positions:
        if pos + config.seq_len >= len(tokens):
            continue

        context = tokens[pos:pos + config.seq_len]
        true_next = tokens[pos + config.seq_len]
        preds = predict_next_words(
            model=model,
            context_words=context,
            word_to_idx=word_to_idx,
            idx_to_word=idx_to_word,
            device=device,
            top_k=5
        )

        print("\nContext:")
        print(" ", " ".join(context))
        print("True next word:")
        print(" ", true_next)
        print("Top predictions:")
        for word, prob in preds:
            print(f"   {word:15s}  {prob:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
