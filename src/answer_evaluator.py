# ============================================================
# answer_evaluator.py
# Siamese LSTM with GloVe for short answer grading
# ============================================================

import os
import re
import string
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
TRAINING_DATA_PATH = os.path.join("data", "sem_eval", "Training_data.csv")
GLOVE_PATH = os.path.join("data", "glove", "glove.6B.100d.txt")
MODEL_SAVE_PATH = os.path.join("models", "best_siamese_lstm_model.pth")
PREPROCESSOR_SAVE_PATH = os.path.join("models", "preprocessor.pkl")

# ============================================================
# CONFIGURATION
# ============================================================
MAX_SEQUENCE_LENGTH = 150
MAX_VOCAB_SIZE = 30000
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001


class TextPreprocessor:
    # Builds vocabulary and converts text to padded sequences

    def __init__(self, maxVocabSize=MAX_VOCAB_SIZE, maxSeqLen=MAX_SEQUENCE_LENGTH):
        self.maxVocabSize = maxVocabSize
        self.maxSeqLen = maxSeqLen
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocabSize = 2

    def cleanText(self, text):
        # Lowercase, remove punctuation, strip whitespace
        if pd.isna(text):
            text = ""
        text = str(text).lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def buildVocabulary(self, texts):
        # Build word index from most frequent words in texts list
        wordCounter = Counter()
        for text in texts:
            wordCounter.update(self.cleanText(text).split())
        for word, _ in wordCounter.most_common(self.maxVocabSize - 2):
            if word not in self.word2idx:
                self.word2idx[word] = self.vocabSize
                self.idx2word[self.vocabSize] = word
                self.vocabSize += 1

    def textToSequence(self, text):
        # Convert a text string to a fixed-length index list
        words = self.cleanText(text).split()
        seq = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words]
        if len(seq) > self.maxSeqLen:
            seq = seq[:self.maxSeqLen]
        else:
            seq = seq + [self.word2idx["<PAD>"]] * (self.maxSeqLen - len(seq))
        return seq


class SAGDataset(Dataset):
    # Pairs (question + reference) with student answer and score

    def __init__(self, questions, referenceAnswers, studentAnswers, scores, preprocessor):
        self.preprocessor = preprocessor
        self.encoderA = [f"{q} {ref}" for q, ref in zip(questions, referenceAnswers)]
        self.encoderB = studentAnswers
        self.scores = scores

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        seqA = self.preprocessor.textToSequence(self.encoderA[idx])
        seqB = self.preprocessor.textToSequence(self.encoderB[idx])
        return (
            torch.tensor(seqA, dtype=torch.long),
            torch.tensor(seqB, dtype=torch.long),
            torch.tensor(self.scores[idx], dtype=torch.float)
        )


class SiameseLSTM(nn.Module):
    # Dual-encoder BiLSTM that scores answer similarity

    def __init__(self, vocabSize, embeddingDim=EMBEDDING_DIM, hiddenDim=HIDDEN_DIM,
                 numLayers=NUM_LAYERS, dropout=DROPOUT, pretrainedEmbeddings=None):
        super(SiameseLSTM, self).__init__()
        self.hiddenDim = hiddenDim
        if pretrainedEmbeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrainedEmbeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocabSize, embeddingDim, padding_idx=0)
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            self.embedding.weight.data[0] = torch.zeros(embeddingDim)
        self.lstm = nn.LSTM(embeddingDim, hiddenDim, numLayers, batch_first=True,
                            bidirectional=True, dropout=dropout if numLayers > 1 else 0)
        self.fc = nn.Linear(hiddenDim * 2, hiddenDim)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x):
        # Encode sequence to one hidden vector
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        combined = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return torch.tanh(self.fc(combined))

    def forward(self, inputA, inputB):
        # Returns cosine similarity scaled to [0, 1]
        sim = nn.functional.cosine_similarity(self.encode(inputA), self.encode(inputB), dim=1)
        return (sim + 1) / 2


def loadGloveEmbeddings(word2idx, glovePath, embeddingDim=EMBEDDING_DIM):
    # Load pretrained GloVe vectors into an embedding matrix
    embeddings = np.random.randn(len(word2idx), embeddingDim) * 0.01
    embeddings[0] = 0
    if os.path.exists(glovePath):
        with open(glovePath, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in word2idx:
                    idx = word2idx[word]
                    vec = np.array(values[1:], dtype="float32")
                    if len(vec) == embeddingDim:
                        embeddings[idx] = vec
    return torch.tensor(embeddings, dtype=torch.float)


def loadSagDataset(csvPath):
    # Parse CSV and convert accuracy labels to scores
    df = pd.read_csv(csvPath)
    labelToScore = {
        "correct": 1.0,
        "partially_correct_incomplete": 0.7,
        "contradictory": 0.3,
        "irrelevant": 0.0,
        "non_domain": 0.0
    }
    data = []
    for _, row in df.iterrows():
        if pd.isna(row["Question_Text"]) or pd.isna(row["Reference_Answer_Text"]) or pd.isna(row["Student_Answer_Text"]):
            continue
        score = labelToScore.get(row["Student_Answer_Accuracy"], 0.0)
        data.append({
            "question": row["Question_Text"],
            "referenceAnswer": row["Reference_Answer_Text"],
            "studentAnswer": row["Student_Answer_Text"],
            "score": score
        })
    return pd.DataFrame(data)


def trainModel(model, trainLoader, valLoader, device, epochs=EPOCHS, lr=LEARNING_RATE):
    # Train with MSELoss and ReduceLROnPlateau, save best checkpoint
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    bestValLoss = float("inf")

    for epoch in range(epochs):
        model.train()
        trainLoss = 0
        for inputA, inputB, scores in trainLoader:
            inputA, inputB, scores = inputA.to(device), inputB.to(device), scores.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputA, inputB), scores)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            trainLoss += loss.item()

        model.eval()
        valLoss = 0
        allPreds, allScores = [], []
        with torch.no_grad():
            for inputA, inputB, scores in valLoader:
                inputA, inputB, scores = inputA.to(device), inputB.to(device), scores.to(device)
                outputs = model(inputA, inputB)
                valLoss += criterion(outputs, scores).item()
                allPreds.extend(outputs.cpu().numpy())
                allScores.extend(scores.cpu().numpy())

        avgVal = valLoss / len(valLoader)
        pearsonCorr, _ = pearsonr(allScores, allPreds)
        print(f"Epoch {epoch+1}/{epochs} | Train: {trainLoss/len(trainLoader):.4f} | Val: {avgVal:.4f} | Pearson: {pearsonCorr:.4f}")

        if avgVal < bestValLoss:
            bestValLoss = avgVal
            os.makedirs("models", exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "vocab_size": model.embedding.num_embeddings}, MODEL_SAVE_PATH)

        scheduler.step(avgVal)

    return model


def runTraining(csvPath=TRAINING_DATA_PATH, glovePath=GLOVE_PATH):
    # Full pipeline: load data, build vocab, train, save preprocessor
    print("Loading dataset...")
    df = loadSagDataset(csvPath)
    allTexts = []
    for q, ref, student in zip(df["question"], df["referenceAnswer"], df["studentAnswer"]):
        allTexts.append(f"{q} {ref}")
        allTexts.append(student)

    preprocessor = TextPreprocessor()
    preprocessor.buildVocabulary(allTexts)
    print(f"Vocab size: {preprocessor.vocabSize} | Samples: {len(df)}")

    trainIdx, valIdx = train_test_split(range(len(df)), test_size=0.2, random_state=42)

    def makeDataset(idx):
        return SAGDataset(
            df["question"].iloc[idx].tolist(),
            df["referenceAnswer"].iloc[idx].tolist(),
            df["studentAnswer"].iloc[idx].tolist(),
            df["score"].iloc[idx].tolist(),
            preprocessor
        )

    trainLoader = DataLoader(makeDataset(trainIdx), batch_size=BATCH_SIZE, shuffle=True)
    valLoader = DataLoader(makeDataset(valIdx), batch_size=BATCH_SIZE, shuffle=False)

    pretrainedEmbeddings = None
    if os.path.exists(glovePath):
        print("Loading GloVe embeddings...")
        pretrainedEmbeddings = loadGloveEmbeddings(preprocessor.word2idx, glovePath)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseLSTM(preprocessor.vocabSize, pretrainedEmbeddings=pretrainedEmbeddings)
    model = trainModel(model, trainLoader, valLoader, device)

    os.makedirs("models", exist_ok=True)
    with open(PREPROCESSOR_SAVE_PATH, "wb") as f:
        pickle.dump(preprocessor, f)

    print("Training complete. Files saved.")
    return model, preprocessor


class AnswerGrader:
    # Loads trained model and grades student answers on demand

    def __init__(self, modelPath=MODEL_SAVE_PATH, preprocessorPath=PREPROCESSOR_SAVE_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(preprocessorPath, "rb") as f:
            self.preprocessor = pickle.load(f)
        checkpoint = torch.load(modelPath, map_location=self.device, weights_only=False)
        stateDict = checkpoint.get("model_state_dict", checkpoint)
        self.model = SiameseLSTM(self.preprocessor.vocabSize)
        self.model.load_state_dict(stateDict)
        self.model.to(self.device)
        self.model.eval()
        print(f"Answer evaluator loaded. Vocab size: {self.preprocessor.vocabSize}")

    def gradeAnswer(self, question, referenceAnswer, studentAnswer, maxMarks):
        # Return (similarity, marks) for a single answer
        inputA = f"{question} {referenceAnswer}"
        seqA = self.preprocessor.textToSequence(inputA)
        seqB = self.preprocessor.textToSequence(studentAnswer)
        tensorA = torch.tensor([seqA], dtype=torch.long).to(self.device)
        tensorB = torch.tensor([seqB], dtype=torch.long).to(self.device)
        with torch.no_grad():
            similarity = self.model(tensorA, tensorB).item()
        return round(similarity, 4), round(similarity * maxMarks, 2)

    def gradeFromText(self, question, passage, studentAnswer, maxMarks):
        # Grade a single text-input answer and return result dict
        similarity, marks = self.gradeAnswer(question, passage, studentAnswer, maxMarks)
        return {"question": question, "studentAnswer": studentAnswer,
                "similarity": similarity, "marksObtained": marks, "maxMarks": maxMarks}

    def gradeFromDataframe(self, df):
        # Grade all rows in a dataframe with standardized columns
        results = []
        for _, row in df.iterrows():
            similarity, marks = self.gradeAnswer(
                row["question"], row["passage"], row["studentAnswer"], float(row["maxMarks"])
            )
            results.append({"Question": row.get("question", ""), "Student Answer": row.get("studentAnswer", ""),
                            "Marks Obtained": marks, "Maximum Marks": float(row["maxMarks"]), "Similarity": similarity})
        return pd.DataFrame(results)


if __name__ == "__main__":
    runTraining()
