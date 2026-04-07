# ============================================================
# btl_classifier.py
# Classifies questions by Bloom's Taxonomy Level (BTL)
# ============================================================

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
BTL_DATASET_PATH = os.path.join("data", "EduQG", "eduqg_llm_formatted.csv")
BTL_MODEL_SAVE_PATH = os.path.join("models", "btl_model.h5")
BTL_TOKENIZER_PATH = os.path.join("models", "btl_tokenizer.pkl")
BTL_ENCODER_PATH = os.path.join("models", "btl_label_encoder.pkl")

# ============================================================
# CONFIGURATION
# ============================================================
MAX_LEN = 50
VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
LSTM_UNITS = 128
DENSE_UNITS = 64
DROPOUT_RATE = 0.5
EPOCHS = 10
BATCH_SIZE = 32

# ============================================================
# BLOOM KEYWORD RULES
# ============================================================

BLOOM_RULES = {
    "Create": ["design", "develop", "formulate", "construct", "create", "propose", "solution"],
    "Evaluate": ["assess", "justify", "evaluate", "critique", "criteria", "why", "opinion"],
    "Analyze": ["compare", "difference", "analyze", "analyse", "distinguish", "relationship"],
    "Apply": ["calculate", "solve", "use", "demonstrate", "implement"],
    "Remember": ["define", "what is", "how many", "list", "identify", "name"],
    "Understand": ["explain", "describe", "interpret", "which", "what would happen"]
}


def assignBloomLevel(question):
    # Return Bloom level based on keyword matching
    q = question.lower()
    for level, keywords in BLOOM_RULES.items():
        if any(keyword in q for keyword in keywords):
            return level
    return "Understand"


def cleanQuestion(question):
    # Remove common boilerplate phrases from question text
    q = question.lower().strip()
    q = q.replace("which of the following", "")
    q = q.replace("select the correct answer", "")
    q = q.replace("choose the correct option", "")
    return q.strip()


def loadBtlDataset(csvPath):
    # Load CSV and return questions list with Bloom labels
    df = pd.read_csv(csvPath)
    questions = df["prompt"].dropna().tolist()
    questions = [q.lower().strip() for q in questions if len(q.split()) > 4]
    extraQuestions = [
        "Explain how neural networks work",
        "Compare supervised and unsupervised learning",
        "Design a system for traffic prediction",
        "Evaluate the impact of climate change"
    ]
    questions.extend(extraQuestions)
    labels = [assignBloomLevel(q) for q in questions]
    return questions, labels


def tokenizeQuestions(questions, tokenizer=None):
    # Convert questions to padded integer sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
        tokenizer.fit_on_texts(questions)
    sequences = tokenizer.texts_to_sequences(questions)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")
    return padded, tokenizer


def buildBtlModel(numClasses):
    # BiLSTM + Conv1D classification model
    from tensorflow.keras.layers import (
        Embedding, Bidirectional, LSTM, Conv1D,
        GlobalMaxPooling1D, Dense, Dropout
    )
    inputLayer = tf.keras.Input(shape=(MAX_LEN,))
    x = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(inputLayer)
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Conv1D(64, 3, activation="relu")(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(DENSE_UNITS, activation="relu")(x)
    x = Dropout(DROPOUT_RATE)(x)
    output = Dense(numClasses, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputLayer, outputs=output)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def runBtlTraining(csvPath=BTL_DATASET_PATH):
    # Full training pipeline for BTL classifier
    print("Loading BTL dataset...")
    questions, labels = loadBtlDataset(csvPath)
    padded, tokenizer = tokenizeQuestions(questions)
    labelEncoder = LabelEncoder()
    y = labelEncoder.fit_transform(labels)
    numClasses = len(labelEncoder.classes_)
    xTrain, xTest, yTrain, yTest = train_test_split(padded, y, test_size=0.2, random_state=42)
    yTrainCat = tf.keras.utils.to_categorical(yTrain, numClasses)
    yTestCat = tf.keras.utils.to_categorical(yTest, numClasses)
    classWeights = compute_class_weight(class_weight="balanced", classes=np.unique(yTrain), y=yTrain)
    classWeightDict = dict(enumerate(classWeights))
    model = buildBtlModel(numClasses)
    model.fit(xTrain, yTrainCat, validation_data=(xTest, yTestCat),
              epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=classWeightDict)
    os.makedirs("models", exist_ok=True)
    model.save(BTL_MODEL_SAVE_PATH)
    with open(BTL_TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(BTL_ENCODER_PATH, "wb") as f:
        pickle.dump(labelEncoder, f)
    print("BTL model saved.")
    return model, tokenizer, labelEncoder


class BtlClassifier:
    # Loads model and classifies questions by Bloom level

    def __init__(self, modelPath=BTL_MODEL_SAVE_PATH, tokenizerPath=BTL_TOKENIZER_PATH, encoderPath=BTL_ENCODER_PATH):
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        self.padSequences = pad_sequences
        self.model = load_model(modelPath)
        with open(tokenizerPath, "rb") as f:
            self.tokenizer = pickle.load(f)
        with open(encoderPath, "rb") as f:
            self.labelEncoder = pickle.load(f)
        self.idxToLabel = {i: label for i, label in enumerate(self.labelEncoder.classes_)}
        print("BTL classifier loaded.")

    def predictBloom(self, question):
        # Return (bloom level, confidence) for one question using model
        q = cleanQuestion(question)
        seq = self.tokenizer.texts_to_sequences([q])
        padded = self.padSequences(seq, maxlen=MAX_LEN, padding="post")
        pred = self.model.predict(padded, verbose=0)[0]
        label = self.idxToLabel[np.argmax(pred)]
        confidence = round(float(np.max(pred)), 2)
        return label, confidence

    def hybridPredict(self, question):
        # Apply keyword rules first, fall back to model prediction
        q = question.lower()
        for level, keywords in BLOOM_RULES.items():
            if any(keyword in q for keyword in keywords[:3]):
                return level, 0.99
        return self.predictBloom(question)

    def classifyFromText(self, rawText):
        # Parse numbered questions from text block and classify each
        results = []
        for line in rawText.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            match = re.match(r"(Q?\d+)\.\s*(.*)", line)
            if match:
                qno = match.group(1)
                question = match.group(2)
                level, confidence = self.hybridPredict(question)
                results.append({"qno": qno, "question": question, "btlLevel": level, "confidence": confidence})
        return results

    def classifyFromFile(self, filePath):
        # Read .txt file and classify each numbered question
        with open(filePath, "r", encoding="utf-8") as f:
            rawText = f.read()
        return self.classifyFromText(rawText)

    def classifyQuestionList(self, questions):
        # Classify a plain list of question strings
        results = []
        for i, question in enumerate(questions):
            level, confidence = self.hybridPredict(question)
            results.append({"qno": i + 1, "question": question, "btlLevel": level, "confidence": confidence})
        return results


if __name__ == "__main__":
    runBtlTraining()
