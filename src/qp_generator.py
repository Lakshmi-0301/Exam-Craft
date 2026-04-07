# ============================================================
# qp_generator.py
# Question Paper Generator - converted from qp-generator-2.ipynb
#
# Uses one model: fine-tuned T5-small (saved as exam_gen_model/)
# Train on Kaggle -> download exam_gen_model/ -> place in models/
# ============================================================

import os
import json
import glob
import random

import torch
import spacy
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from rouge_score import rouge_scorer

# ============================================================
# PATHS - datasets go in data/, model goes in models/
# ============================================================
RACE_FOLDER_PATH = os.path.join("data", "race", "RACE")
SQUAD_TRAIN_PATH = os.path.join("data", "squad", "train-v1.1.json")
SQUAD_DEV_PATH   = os.path.join("data", "squad", "dev-v1.1.json")
MODEL_SAVE_PATH  = os.path.join("models", "exam_gen_model")

# ============================================================
# CONFIG
# ============================================================
MAX_INPUT_LEN  = 512
MAX_TARGET_LEN = 64
BATCH_SIZE     = 8
NUM_EPOCHS     = 8
LEARNING_RATE  = 5e-5
TRAIN_SAMPLES  = 5000

WEAK_STARTERS = {"the", "a", "an", "this", "that", "these", "those", "one", "many"}

# Device - CPU on local is fine since we load from saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load spacy NER model
try:
    nlpModel = spacy.load("en_core_web_sm")
except OSError:
    nlpModel = None
    print("spacy model missing. Run: python -m spacy download en_core_web_sm")


# ============================================================
# SQUAD LOADER
# ============================================================

def parseSquadFile(filePath):
    # Parse one SQuAD v1.1 JSON file into a flat list of dicts
    with open(filePath, "r", encoding="utf-8") as f:
        rawData = json.load(f)
    samples = []
    for article in rawData["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                samples.append({
                    "context": context,
                    "question": qa["question"],
                    "answers": qa["answers"]
                })
    return samples


def loadSquadLocal(trainPath=SQUAD_TRAIN_PATH, devPath=SQUAD_DEV_PATH):
    # Load both SQuAD splits and return as a dict
    trainSamples = parseSquadFile(trainPath)
    devSamples   = parseSquadFile(devPath)
    print(f"SQuAD train: {len(trainSamples)} | dev: {len(devSamples)}")
    return {"train": trainSamples, "validation": devSamples}


# ============================================================
# RACE LOADER
# ============================================================

def parseRaceFile(filePath):
    # Parse one RACE .txt JSON file into a list of sample dicts
    with open(filePath, "r", encoding="utf-8") as f:
        rawData = json.load(f)
    samples = []
    article     = rawData.get("article", "")
    questions   = rawData.get("questions", [])
    optionsList = rawData.get("options", [])
    answers     = rawData.get("answers", [])
    for i, question in enumerate(questions):
        samples.append({
            "article": article,
            "question": question,
            "options": optionsList[i],
            "answer": answers[i]
        })
    return samples


def loadRaceSplit(splitFolderPath):
    # Load all RACE files from one split folder (high + middle)
    allSamples = []
    for pattern in ["high", "middle"]:
        for filePath in glob.glob(os.path.join(splitFolderPath, pattern, "*.txt")):
            try:
                allSamples.extend(parseRaceFile(filePath))
            except Exception as err:
                print(f"Skipping {filePath}: {err}")
    return allSamples


def loadRaceLocal(raceFolderPath=RACE_FOLDER_PATH):
    # Load RACE train and dev splits and return as a dict
    trainSamples = loadRaceSplit(os.path.join(raceFolderPath, "train"))
    devSamples   = loadRaceSplit(os.path.join(raceFolderPath, "dev"))
    print(f"RACE train: {len(trainSamples)} | dev: {len(devSamples)}")
    return {"train": trainSamples, "validation": devSamples}


# ============================================================
# NER / CONCEPT EXTRACTION
# ============================================================

def extractKeyEntities(text):
    # Return named entity strings from text using spacy
    if nlpModel is None:
        return []
    doc = nlpModel(text)
    return [ent.text for ent in doc.ents]


def extractNounChunks(text):
    # Return noun chunk strings from text using spacy
    if nlpModel is None:
        return []
    doc = nlpModel(text)
    return [chunk.text for chunk in doc.noun_chunks]


def extractKeyConcepts(text):
    # Combine named entities and noun chunks, deduplicated
    return list(set(extractKeyEntities(text) + extractNounChunks(text)))


def isValidConcept(concept):
    # Concept must have 2+ real words and not start with a weak article
    words = [w for w in concept.split() if w.isalpha()]
    firstWord = words[0].lower() if words else ""
    return len(words) >= 2 and firstWord not in WEAK_STARTERS


def extractPhrasesFromSentence(sentence):
    # Return valid noun chunk phrases from one sentence
    if nlpModel is None:
        return []
    doc = nlpModel(sentence)
    return [chunk.text for chunk in doc.noun_chunks if isValidConcept(chunk.text)]


def pickDistractors(passage, correctAnswer, numDistractors=3):
    # Build distractors from sibling noun phrases in the same sentences
    sentences = [s.strip() for s in passage.split(".") if len(s.strip()) > 10]
    distractors = []
    seen = {correctAnswer.lower()}

    for sentence in sentences:
        if correctAnswer.lower() in sentence.lower():
            for phrase in extractPhrasesFromSentence(sentence):
                if len(distractors) >= numDistractors:
                    break
                if phrase.lower() not in seen:
                    distractors.append(phrase)
                    seen.add(phrase.lower())

    for sentence in sentences:
        if len(distractors) >= numDistractors:
            break
        for phrase in extractPhrasesFromSentence(sentence):
            if len(distractors) >= numDistractors:
                break
            if phrase.lower() not in seen:
                distractors.append(phrase)
                seen.add(phrase.lower())

    return distractors[:numDistractors]


# ============================================================
# T5 MODEL - load from local saved folder
# ============================================================

def loadT5Model(modelPath=MODEL_SAVE_PATH):
    # Load T5 from the local saved folder (downloaded from Kaggle)
    if not os.path.isdir(modelPath) or not os.listdir(modelPath):
        raise FileNotFoundError(
            f"Model folder '{modelPath}' is empty or missing.\n"
            f"Run the Kaggle notebook to train and download exam_gen_model/, "
            f"then place it in {modelPath}."
        )
    print(f"Loading T5 model from: {modelPath}")
    tokenizer = T5Tokenizer.from_pretrained(modelPath)
    model = T5ForConditionalGeneration.from_pretrained(modelPath)
    model = model.to(device)
    print(f"T5 model loaded on {device}")
    return tokenizer, model


# ============================================================
# T5 FINE-TUNING (runs on Kaggle, not needed locally)
# ============================================================

def formatSquadSample(sample):
    # Format one SQuAD sample as T5 input/target pair dict
    inputText  = f"generate question: context: {sample['context']} answer: {sample['answers'][0]['text']}"
    targetText = sample["question"]
    return {"inputText": inputText, "targetText": targetText}


def tokenizeSample(sample, tokenizer, maxInputLen=MAX_INPUT_LEN, maxTargetLen=MAX_TARGET_LEN):
    # Tokenize one formatted sample into tensors for T5
    inputEncoded = tokenizer(
        sample["inputText"], max_length=maxInputLen,
        padding="max_length", truncation=True, return_tensors="pt"
    )
    targetEncoded = tokenizer(
        sample["targetText"], max_length=maxTargetLen,
        padding="max_length", truncation=True, return_tensors="pt"
    )
    labels = targetEncoded["input_ids"].squeeze()
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        "input_ids":      inputEncoded["input_ids"].squeeze(),
        "attention_mask": inputEncoded["attention_mask"].squeeze(),
        "labels":         labels
    }


class QuestionDataset(Dataset):
    # PyTorch dataset wrapping formatted SQuAD samples for T5

    def __init__(self, samples, tokenizer, maxInputLen=MAX_INPUT_LEN, maxTargetLen=MAX_TARGET_LEN):
        self.samples      = samples
        self.tokenizer    = tokenizer
        self.maxInputLen  = maxInputLen
        self.maxTargetLen = maxTargetLen

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return tokenizeSample(self.samples[idx], self.tokenizer, self.maxInputLen, self.maxTargetLen)


def trainOneEpoch(model, dataLoader, optimizer):
    # Train one epoch and return average loss
    model.train()
    totalLoss = 0
    for batch in dataLoader:
        inputIds      = batch["input_ids"].to(device)
        attentionMask = batch["attention_mask"].to(device)
        labels        = batch["labels"].to(device)
        optimizer.zero_grad()
        output = model(input_ids=inputIds, attention_mask=attentionMask, labels=labels)
        output.loss.backward()
        optimizer.step()
        totalLoss += output.loss.item()
    return totalLoss / len(dataLoader)


def evaluateModel(model, dataLoader):
    # Evaluate model on validation set without weight updates
    model.eval()
    totalLoss = 0
    with torch.no_grad():
        for batch in dataLoader:
            inputIds      = batch["input_ids"].to(device)
            attentionMask = batch["attention_mask"].to(device)
            labels        = batch["labels"].to(device)
            output = model(input_ids=inputIds, attention_mask=attentionMask, labels=labels)
            totalLoss += output.loss.item()
    return totalLoss / len(dataLoader)


def runTrainingPipeline(squadData, savePath=MODEL_SAVE_PATH):
    # Full training pipeline: load T5, train on SQuAD, save model
    t5Tokenizer, t5Model = loadT5FromHub()
    trainSamples = [formatSquadSample(s) for s in squadData["train"][:TRAIN_SAMPLES]]
    valSamples   = [formatSquadSample(s) for s in squadData["validation"][:TRAIN_SAMPLES]]
    trainLoader  = DataLoader(QuestionDataset(trainSamples, t5Tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    valLoader    = DataLoader(QuestionDataset(valSamples, t5Tokenizer),   batch_size=BATCH_SIZE, shuffle=False)
    optimizer    = AdamW(t5Model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        trainLoss = trainOneEpoch(t5Model, trainLoader, optimizer)
        valLoss   = evaluateModel(t5Model, valLoader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train: {trainLoss:.4f} | Val: {valLoss:.4f}")

    os.makedirs(savePath, exist_ok=True)
    t5Model.save_pretrained(savePath)
    t5Tokenizer.save_pretrained(savePath)
    print(f"Model saved to {savePath}")
    return t5Tokenizer, t5Model


def loadT5FromHub(modelName="t5-small"):
    # Download T5 from HuggingFace (used during Kaggle training only)
    tokenizer = T5Tokenizer.from_pretrained(modelName)
    model     = T5ForConditionalGeneration.from_pretrained(modelName).to(device)
    return tokenizer, model


# ============================================================
# QUESTION GENERATION
# ============================================================

def generateQuestion(model, tokenizer, context, answer, maxLen=MAX_TARGET_LEN):
    # Generate one question given a context passage and an answer string
    inputText    = f"generate question: context: {context} answer: {answer}"
    inputEncoded = tokenizer(inputText, return_tensors="pt", max_length=MAX_INPUT_LEN, truncation=True).to(device)
    outputIds    = model.generate(
        input_ids=inputEncoded["input_ids"],
        attention_mask=inputEncoded["attention_mask"],
        max_length=maxLen,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputIds[0], skip_special_tokens=True)


def generateDescriptiveQuestion(model, tokenizer, article, maxLen=128):
    # Generate a descriptive question about the main idea of an article
    inputText    = f"generate question: context: {article[:512]} answer: main idea"
    inputEncoded = tokenizer(inputText, return_tensors="pt", max_length=MAX_INPUT_LEN, truncation=True).to(device)
    outputIds    = model.generate(
        input_ids=inputEncoded["input_ids"],
        attention_mask=inputEncoded["attention_mask"],
        max_length=maxLen,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputIds[0], skip_special_tokens=True)


# ============================================================
# MCQ GENERATION (from RACE)
# ============================================================

def resolveRaceAnswer(sample):
    # Map RACE answer letter A/B/C/D to the actual option text
    optionLabels = ["A", "B", "C", "D"]
    correctIndex = optionLabels.index(sample["answer"])
    return sample["options"][correctIndex]


def buildMcqFromRaceSample(sample):
    # Build MCQ dict - question and options only, no concept shown
    return {
        "question":     sample["question"],
        "options":      sample["options"],
        "correctAnswer": resolveRaceAnswer(sample)
    }


def generateMcqs(raceData, count=5):
    # Return a list of MCQ dicts from RACE validation split
    return [
        buildMcqFromRaceSample(raceData["validation"][i])
        for i in range(min(count, len(raceData["validation"])))
    ]


# ============================================================
# SHORT ANSWER GENERATION
# ============================================================

def generateShortAnswerQuestions(squadData, model, tokenizer, count=5):
    # Generate short-answer questions from SQuAD validation samples
    resultList = []
    for i in range(min(count, len(squadData["validation"]))):
        sample   = squadData["validation"][i]
        context  = sample["context"]
        answer   = sample["answers"][0]["text"]
        question = generateQuestion(model, tokenizer, context, answer)
        resultList.append({"question": question, "answer": answer})
    return resultList


def generateShortAnswersFromPassage(model, tokenizer, passage, count):
    # Generate short-answer questions from a user-provided passage
    allConcepts   = extractKeyConcepts(passage)
    validConcepts = [c for c in allConcepts if isValidConcept(c)]
    halfPoint     = len(validConcepts) // 2
    shortConcepts = validConcepts[halfPoint:]
    seenQuestions = set()
    resultList    = []

    for concept in shortConcepts:
        if len(resultList) >= count:
            break
        question    = generateQuestion(model, tokenizer, passage, concept)
        questionKey = question.strip().lower()
        if questionKey not in seenQuestions:
            seenQuestions.add(questionKey)
            resultList.append({"question": question, "answer": concept})

    return resultList


# ============================================================
# DESCRIPTIVE QUESTION GENERATION
# ============================================================

def generateDescriptiveQuestions(raceData, model, tokenizer, count=3):
    # Generate descriptive questions from RACE validation articles
    resultList = []
    for i in range(min(count, len(raceData["validation"]))):
        article  = raceData["validation"][i]["article"]
        question = generateDescriptiveQuestion(model, tokenizer, article)
        resultList.append({"question": question, "articleSnippet": article[:200]})
    return resultList


def generateDescriptiveFromPassage(model, tokenizer, passage, count, existingQuestions):
    # Generate descriptive questions from sentences in a user passage
    sentences     = [s.strip() for s in passage.split(".") if len(s.strip()) > 30]
    seenQuestions = set(q.strip().lower() for q in existingQuestions)
    resultList    = []

    for sentence in sentences:
        if len(resultList) >= count:
            break
        question    = generateQuestion(model, tokenizer, passage, sentence)
        questionKey = question.strip().lower()
        if questionKey not in seenQuestions:
            seenQuestions.add(questionKey)
            resultList.append({"question": question, "hint": sentence})

    return resultList


# ============================================================
# EXAM PAPER ASSEMBLY
# ============================================================

def generateExamPaper(squadData, raceData, model, tokenizer, mcqCount=5, shortCount=5, descCount=3):
    # Build a complete exam paper from SQuAD + RACE datasets
    print("Generating MCQs from RACE...")
    mcqs = generateMcqs(raceData, count=mcqCount)

    print("Generating short answer questions from SQuAD...")
    shortAnswers = generateShortAnswerQuestions(squadData, model, tokenizer, count=shortCount)

    print("Generating descriptive questions from RACE...")
    descriptive = generateDescriptiveQuestions(raceData, model, tokenizer, count=descCount)

    return {
        "mcqs":                  mcqs,
        "shortAnswerQuestions":  shortAnswers,
        "descriptiveQuestions":  descriptive
    }


def generateExamPaperFromPassage(model, tokenizer, passage, mcqCount=3, shortCount=3, descCount=2):
    # Build a complete exam paper from a single user-provided passage
    allConcepts   = extractKeyConcepts(passage)
    validConcepts = [c for c in allConcepts if isValidConcept(c)]
    mcqConcepts   = validConcepts[:mcqCount]

    # MCQs from passage concepts
    seenQ = set()
    mcqs  = []
    for concept in mcqConcepts:
        if len(mcqs) >= mcqCount:
            break
        question = generateQuestion(model, tokenizer, passage, concept)
        if question.strip().lower() not in seenQ:
            seenQ.add(question.strip().lower())
            distractors = pickDistractors(passage, concept)
            allOptions  = distractors + [concept]
            random.shuffle(allOptions)
            mcqs.append({"question": question, "options": allOptions, "correctAnswer": concept})

    # Short answers from second half of concepts
    shortAnswers = generateShortAnswersFromPassage(model, tokenizer, passage, shortCount)

    # Descriptive from sentences
    existingQ   = [qa["question"] for qa in shortAnswers] + [m["question"] for m in mcqs]
    descriptive = generateDescriptiveFromPassage(model, tokenizer, passage, descCount, existingQ)

    return {
        "mcqs":                  mcqs,
        "shortAnswerQuestions":  shortAnswers,
        "descriptiveQuestions":  descriptive
    }


def countMaxUniqueQuestions(passage):
    # Return max possible short and descriptive question counts for a passage
    allConcepts   = extractKeyConcepts(passage)
    validConcepts = [c for c in allConcepts if isValidConcept(c)]
    sentences     = [s.strip() for s in passage.split(".") if len(s.strip()) > 30]
    return len(validConcepts), len(sentences)


# ============================================================
# ROUGE-L EVALUATION
# ============================================================

def computeRougeScore(generatedQuestion, referenceQuestion):
    # Return ROUGE-L F1 score between two question strings
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(referenceQuestion, generatedQuestion)
    return scores["rougeL"].fmeasure


def evaluateGeneratedQuestions(squadData, model, tokenizer, count=10):
    # Print per-sample and average ROUGE-L on unique SQuAD contexts
    seenContexts = set()
    resultList   = []
    index        = 0
    validationData = squadData["validation"]

    while len(resultList) < count and index < len(validationData):
        sample     = validationData[index]
        contextKey = sample["context"][:100]
        if contextKey not in seenContexts:
            seenContexts.add(contextKey)
            answer            = sample["answers"][0]["text"]
            referenceQuestion = sample["question"]
            generatedQuestion = generateQuestion(model, tokenizer, sample["context"], answer)
            score             = computeRougeScore(generatedQuestion, referenceQuestion)
            resultList.append(score)
            print(f"Sample {len(resultList)} | ROUGE-L: {score:.4f} | Generated: {generatedQuestion}")
        index += 1

    avgScore = sum(resultList) / len(resultList)
    print(f"\nAverage ROUGE-L: {avgScore:.4f}")
    return avgScore


if __name__ == "__main__":
    print("Loading T5 model from models/exam_gen_model/ ...")
    t5Tokenizer, t5Model = loadT5Model()
    print("Model ready. Use generateExamPaper() or generateExamPaperFromPassage().")
