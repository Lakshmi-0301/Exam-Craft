# ============================================================
# app.py
# Streamlit Dashboard for Deep Learning Exam Paper Project
# Three modules: Answer Evaluator, BTL Classifier, QP Generator
# ============================================================

import os
import sys
import io
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Deep Learning Exam Tools",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("Deep Learning Exam Tools")
st.sidebar.markdown("---")
module = st.sidebar.radio(
    "Select Module",
    ["Answer Evaluator", "BTL Classifier", "QP Generator"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Paths**")
st.sidebar.markdown("Place your data files in the `data/` folder:")

# ============================================================
# HELPER: LOAD CSV FROM UPLOAD OR MANUAL INPUT
# ============================================================

def parseUploadedCsv(uploadedFile):
    # Parse a Streamlit uploaded CSV file into a DataFrame
    try:
        content = uploadedFile.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))
        return df
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None


def autoDetectColumns(df):
    # Find likely column names for question/passage/answer/marks fields
    cols = {c.lower(): c for c in df.columns}
    result = {}

    def findCol(candidates, default=None):
        for candidate in candidates:
            for lowerCol, origCol in cols.items():
                if candidate in lowerCol:
                    return origCol
        return default or list(df.columns)[0]

    questionCols = [col for col in df.columns if col.lower().strip() == "question"]
    if questionCols:
        result["question"] = questionCols[0]
    else:
        result["question"] = findCol(["question_text", "question"])
    result["passage"] = findCol(["reference_answer", "passage", "reference"])
    result["studentAnswer"] = findCol(["student_answer", "student"])
    result["maxMarks"] = findCol(["max_marks", "marks", "max"])
    result["qNum"] = findCol(["question number", "q_id", "id", "qno"])
    return result


# ============================================================
# MODULE 1: ANSWER EVALUATOR
# ============================================================

def runAnswerEvaluator():
    st.title("Answer Evaluator")
    st.markdown("Grade student answers using a trained Siamese LSTM model.")

    tab1, tab2, tab3 = st.tabs(["Train Model", "Evaluate via Text Input", "Evaluate via CSV Upload"])

    # --- TAB 1: TRAIN ---
    with tab1:
        st.subheader("Train the Answer Evaluation Model")
        trainCsvPath = st.text_input(
            "Training CSV Path",
            value=os.path.join("data", "sem_eval", "Training_data.csv"),
            help="Path to Training_data.csv from SemEval SAG dataset"
        )
        glovePath = st.text_input(
            "GloVe Embeddings Path",
            value=os.path.join("data", "glove", "glove.6B.100d.txt"),
            help="Optional: Path to glove.6B.100d.txt for better embeddings"
        )
        if st.button("Start Training", key="train_eval"):
            if not os.path.exists(trainCsvPath):
                st.error(f"Training CSV not found: {trainCsvPath}")
            else:
                with st.spinner("Training model... this may take several minutes."):
                    try:
                        from answer_evaluator import runTraining
                        logContainer = st.empty()
                        runTraining(csvPath=trainCsvPath, glovePath=glovePath)
                        st.success("Training complete! Model saved to models/")
                    except Exception as e:
                        st.error(f"Training error: {e}")

    # --- TAB 2: TEXT INPUT ---
    with tab2:
        st.subheader("Grade a Single Answer")
        col1, col2 = st.columns(2)
        with col1:
            question = st.text_area("Question", height=80, placeholder="Enter the question here...")
            passage = st.text_area("Reference Answer / Passage", height=120, placeholder="Enter the reference answer or passage...")
        with col2:
            studentAnswer = st.text_area("Student Answer", height=120, placeholder="Enter the student's answer here...")
            maxMarks = st.number_input("Maximum Marks", min_value=1, max_value=100, value=5)

        if st.button("Grade Answer", key="grade_text"):
            if not question or not passage or not studentAnswer:
                st.warning("Please fill in all fields.")
            else:
                modelPath = os.path.join("models", "best_siamese_lstm_model.pth")
                preprocessorPath = os.path.join("models", "preprocessor.pkl")
                if not os.path.exists(modelPath) or not os.path.exists(preprocessorPath):
                    st.error("Model not found. Please train the model first in the 'Train Model' tab.")
                else:
                    try:
                        from answer_evaluator import AnswerGrader
                        grader = AnswerGrader(modelPath, preprocessorPath)
                        result = grader.gradeFromText(question, passage, studentAnswer, maxMarks)
                        st.markdown("---")
                        col1r, col2r, col3r = st.columns(3)
                        col1r.metric("Similarity Score", f"{result['similarity']:.4f}")
                        col2r.metric("Marks Obtained", f"{result['marksObtained']:.2f}")
                        col3r.metric("Maximum Marks", result["maxMarks"])
                        percentage = (result["marksObtained"] / maxMarks) * 100
                        st.progress(min(percentage / 100, 1.0))
                        st.caption(f"Score: {percentage:.1f}%")
                        if percentage >= 80:
                            st.success("Excellent answer!")
                        elif percentage >= 60:
                            st.info("Good answer - partially correct.")
                        elif percentage >= 40:
                            st.warning("Partial credit - answer needs improvement.")
                        else:
                            st.error("Answer is mostly incorrect or irrelevant.")
                    except Exception as e:
                        st.error(f"Grading error: {e}")

    # --- TAB 3: CSV UPLOAD ---
    with tab3:
        st.subheader("Grade Multiple Answers from CSV")
        st.markdown("Upload a CSV with columns: `Question number`, `Question`, `Passage`, `student_answer`, `max_marks`")

        uploadedFile = st.file_uploader("Upload CSV file", type=["csv"], key="eval_csv")

        if uploadedFile is not None:
            df = parseUploadedCsv(uploadedFile)
            if df is not None:
                st.write(f"Loaded {len(df)} rows. Preview:")
                st.dataframe(df.head(3))
                colMap = autoDetectColumns(df)
                st.markdown("**Detected Columns:**")
                st.json(colMap)

                if st.button("Grade All Answers", key="grade_csv"):
                    modelPath = os.path.join("models", "best_siamese_lstm_model.pth")
                    preprocessorPath = os.path.join("models", "preprocessor.pkl")
                    if not os.path.exists(modelPath) or not os.path.exists(preprocessorPath):
                        st.error("Model not found. Please train the model first.")
                    else:
                        try:
                            from answer_evaluator import AnswerGrader
                            grader = AnswerGrader(modelPath, preprocessorPath)
                            results = []
                            totalObtained = 0
                            totalMax = 0
                            progressBar = st.progress(0)
                            for i, row in df.iterrows():
                                maxM = float(row.get(colMap["maxMarks"], 10))
                                similarity, marks = grader.gradeAnswer(
                                    row[colMap["question"]],
                                    row[colMap["passage"]],
                                    row[colMap["studentAnswer"]],
                                    maxM
                                )
                                totalObtained += marks
                                totalMax += maxM
                                results.append({
                                    "Q#": row.get(colMap["qNum"], i + 1),
                                    "Question": row[colMap["question"]],
                                    "Student Answer": row[colMap["studentAnswer"]],
                                    "Marks Obtained": marks,
                                    "Maximum Marks": maxM,
                                    "Similarity": similarity
                                })
                                progressBar.progress((i + 1) / len(df))

                            resultDf = pd.DataFrame(results)
                            st.markdown("---")
                            col1r, col2r, col3r = st.columns(3)
                            col1r.metric("Total Score", f"{totalObtained:.2f}")
                            col2r.metric("Out of", f"{totalMax:.0f}")
                            col3r.metric("Percentage", f"{(totalObtained / totalMax * 100):.1f}%")
                            st.dataframe(resultDf)
                            csvOut = resultDf.to_csv(index=False)
                            st.download_button("Download Results CSV", csvOut, "grading_results.csv", "text/csv")
                        except Exception as e:
                            st.error(f"Grading error: {e}")


# ============================================================
# MODULE 2: BTL CLASSIFIER
# ============================================================

def runBtlClassifier():
    st.title("BTL Classifier")
    st.markdown("Classify questions by Bloom's Taxonomy Level using a BiLSTM + CNN model.")

    tab1, tab2, tab3 = st.tabs(["Train Model", "Classify via Text Input", "Classify via File Upload"])

    # --- TAB 1: TRAIN ---
    with tab1:
        st.subheader("Train the BTL Classification Model")
        btlCsvPath = st.text_input(
            "EduQG Dataset CSV Path",
            value=os.path.join("data", "EduQG", "eduqg_llm_formatted.csv"),
            help="Path to eduqg_llm_formatted.csv dataset"
        )
        if st.button("Start BTL Training", key="train_btl"):
            if not os.path.exists(btlCsvPath):
                st.error(f"Dataset not found: {btlCsvPath}")
            else:
                with st.spinner("Training BTL model... this may take a few minutes."):
                    try:
                        from btl_classifier import runBtlTraining
                        runBtlTraining(csvPath=btlCsvPath)
                        st.success("BTL model training complete!")
                    except Exception as e:
                        st.error(f"Training error: {e}")

    # --- TAB 2: TEXT INPUT ---
    with tab2:
        st.subheader("Classify Questions by Bloom's Taxonomy Level")
        st.markdown("Enter one question per line in the format: `Q1. Your question here`")

        inputText = st.text_area(
            "Questions",
            height=200,
            placeholder="Q1. What is photosynthesis?\nQ2. Explain how plants make food.\nQ3. Why is chlorophyll important?"
        )

        useModel = st.checkbox("Use trained model (uncheck for rule-based only)", value=False)

        if st.button("Classify Questions", key="classify_text"):
            if not inputText.strip():
                st.warning("Please enter at least one question.")
            else:
                try:
                    if useModel:
                        modelPath = os.path.join("models", "btl_model.h5")
                        if not os.path.exists(modelPath):
                            st.error("BTL model not found. Please train first or use rule-based mode.")
                        else:
                            from btl_classifier import BtlClassifier
                            classifier = BtlClassifier()
                            results = classifier.classifyFromText(inputText)
                    else:
                        import re
                        from btl_classifier import assignBloomLevel, BLOOM_RULES
                        results = []
                        for line in inputText.strip().split("\n"):
                            line = line.strip()
                            if not line:
                                continue
                            match = re.match(r"(Q?\d+)\.\s*(.*)", line)
                            if match:
                                qno = match.group(1)
                                question = match.group(2)
                                level = assignBloomLevel(question)
                                results.append({"qno": qno, "question": question, "btlLevel": level, "confidence": "Rule-based"})

                    if results:
                        resultDf = pd.DataFrame(results)
                        resultDf.columns = ["Q#", "Question", "BTL Level", "Confidence"]
                        st.dataframe(resultDf, use_container_width=True)
                        levelCounts = resultDf["BTL Level"].value_counts()
                        st.bar_chart(levelCounts)
                        csvOut = resultDf.to_csv(index=False)
                        st.download_button("Download Results", csvOut, "btl_results.csv", "text/csv")
                    else:
                        st.warning("No questions matched the expected format (Q1. Question...)")
                except Exception as e:
                    st.error(f"Classification error: {e}")

    # --- TAB 3: FILE UPLOAD ---
    with tab3:
        st.subheader("Classify Questions from a .txt File")
        st.markdown("Upload a .txt file with one question per line: `Q1. Question text`")

        uploadedFile = st.file_uploader("Upload .txt file", type=["txt"], key="btl_file")

        if uploadedFile is not None:
            content = uploadedFile.read().decode("utf-8")
            st.text_area("File Preview", value=content[:500], height=150)

            useModelFile = st.checkbox("Use trained model", value=False, key="btl_model_file")

            if st.button("Classify Questions from File", key="classify_file"):
                try:
                    import re
                    if useModelFile:
                        modelPath = os.path.join("models", "btl_model.h5")
                        if not os.path.exists(modelPath):
                            st.error("BTL model not found. Please train first.")
                        else:
                            from btl_classifier import BtlClassifier
                            classifier = BtlClassifier()
                            results = classifier.classifyFromText(content)
                    else:
                        from btl_classifier import assignBloomLevel
                        results = []
                        for line in content.strip().split("\n"):
                            line = line.strip()
                            if not line:
                                continue
                            match = re.match(r"(Q?\d+)\.\s*(.*)", line)
                            if match:
                                qno = match.group(1)
                                question = match.group(2)
                                level = assignBloomLevel(question)
                                results.append({"qno": qno, "question": question, "btlLevel": level, "confidence": "Rule-based"})

                    if results:
                        resultDf = pd.DataFrame(results)
                        resultDf.columns = ["Q#", "Question", "BTL Level", "Confidence"]
                        st.dataframe(resultDf, use_container_width=True)
                        csvOut = resultDf.to_csv(index=False)
                        st.download_button("Download Results", csvOut, "btl_results.csv", "text/csv")
                except Exception as e:
                    st.error(f"Classification error: {e}")


# ============================================================
# MODULE 3: QP GENERATOR
# ============================================================

@st.cache_resource
def loadSavedT5Model():
    # Load the saved T5 model once and cache it for the session
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    modelPath = os.path.join("models", "exam_gen_model")
    tokenizer = T5Tokenizer.from_pretrained(modelPath)
    model     = T5ForConditionalGeneration.from_pretrained(modelPath)
    return tokenizer, model


def getSavedT5Model():
    # Return cached model or show a clear error if the folder is missing
    modelPath = os.path.join("models", "exam_gen_model")
    if not os.path.isdir(modelPath) or not os.listdir(modelPath):
        st.error(
            "**Saved T5 model not found.**\n\n"
            "Please:\n"
            "1. Train on Kaggle with `qp-generator-2.ipynb` (saves `exam_gen_model/`)\n"
            "2. Download and unzip `exam_gen_model.zip` from the Kaggle output panel\n"
            "3. Place the folder at `models/exam_gen_model/`\n"
            "4. Restart this dashboard"
        )
        return None, None
    return loadSavedT5Model()


def runQpGenerator():
    st.title("Question Paper Generator")
    st.markdown(
        "Generate **short-answer and descriptive** questions from a custom passage using the "
        "saved T5 model, or generate all three question types from RACE + SQuAD datasets."
    )

    tab1, tab2 = st.tabs(["Generate from Custom Passage", "Generate from RACE + SQuAD Datasets"])

    # ----------------------------------------------------------------
    # TAB 1 — CUSTOM PASSAGE
    # Uses pre-saved model directly. No training. No MCQs.
    # ----------------------------------------------------------------
    with tab1:
        st.subheader("Generate from a Custom Passage")
        st.markdown(
            "Paste any passage or upload a `.txt` file. "
            "The **saved** T5 model at `models/exam_gen_model/` will generate "
            "**short-answer** and **descriptive** questions — no training required."
        )

        # Live model presence indicator
        t5ModelPath = os.path.join("models", "exam_gen_model")
        if os.path.isdir(t5ModelPath) and os.listdir(t5ModelPath):
            st.success(f" Saved model found at `{t5ModelPath}`")
        else:
            st.warning(f" No saved model found at `{t5ModelPath}` — generation will fail until you place the model there.")

        st.markdown("---")

        inputMethod = st.radio(
            "Input Method",
            ["Paste Text", "Upload .txt File"],
            horizontal=True,
            key="passage_input_method"
        )

        passage = ""
        if inputMethod == "Paste Text":
            passage = st.text_area(
                "Paste your passage here",
                height=250,
                placeholder=(
                    "The Great Fire swept across London in 1666, originating in a baker's shop "
                    "on Pudding Lane. The fire destroyed 13,200 houses and 87 churches..."
                )
            )
        else:
            uploadedPassage = st.file_uploader("Upload passage .txt file", type=["txt"], key="passage_file")
            if uploadedPassage is not None:
                passage = uploadedPassage.read().decode("utf-8")
                st.text_area(
                    "File Preview",
                    value=passage[:500] + ("..." if len(passage) > 500 else ""),
                    height=140
                )

        st.markdown("---")
        col1, col2 = st.columns(2)
        shortCountP = col1.number_input("Short Answer Questions", min_value=0, max_value=10, value=3, key="short_p")
        descCountP  = col2.number_input("Descriptive Questions",  min_value=0, max_value=5,  value=2, key="desc_p")

        # Live capacity hint — shows without needing to click Generate
        if passage.strip():
            try:
                from qp_generator import countMaxUniqueQuestions
                maxShort, maxDesc = countMaxUniqueQuestions(passage)
                st.caption(
                    f"Passage capacity: up to **{maxShort}** short-answer and "
                    f"**{maxDesc}** descriptive questions."
                )
            except Exception:
                pass

        if st.button("Generate Questions", type="primary", key="gen_passage"):
            if not passage.strip():
                st.warning("Please provide a passage first.")
            else:
                t5Tokenizer, t5Model = getSavedT5Model()
                if t5Model is not None:
                    with st.spinner("Generating questions — this may take 30-60 s on CPU..."):
                        try:
                            from qp_generator import countMaxUniqueQuestions, generateExamPaperFromPassage

                            maxShort, maxDesc = countMaxUniqueQuestions(passage)
                            finalShort = min(shortCountP, maxShort)
                            finalDesc  = min(descCountP,  maxDesc)

                            if maxShort == 0 and maxDesc == 0:
                                st.error(
                                    "Passage is too short to extract concepts. "
                                    "Please use a longer passage (at least 100 words)."
                                )
                            else:
                                if finalShort < shortCountP:
                                    st.info(f"Only {maxShort} short-answer question(s) possible from this passage.")
                                if finalDesc < descCountP:
                                    st.info(f"Only {maxDesc} descriptive question(s) possible from this passage.")

                                # mcqCount=0 skips MCQ generation entirely
                                examPaper = generateExamPaperFromPassage(
                                    t5Model, t5Tokenizer, passage,
                                    mcqCount=0,
                                    shortCount=finalShort,
                                    descCount=finalDesc
                                )
                                displayPassageExamPaper(examPaper)
                        except Exception as e:
                            st.error(f"Generation error: {e}")

    # ----------------------------------------------------------------
    # TAB 2 — RACE + SQUAD DATASETS  (unchanged)
    # ----------------------------------------------------------------
    with tab2:
        st.subheader("Generate Exam Paper from SQuAD + RACE")
        col1, col2, col3 = st.columns(3)
        mcqCount   = col1.number_input("Number of MCQs",        min_value=1, max_value=20, value=5)
        shortCount = col2.number_input("Short Answer Questions", min_value=1, max_value=20, value=5)
        descCount  = col3.number_input("Descriptive Questions",  min_value=1, max_value=10, value=3)

        raceFolderPath = st.text_input("RACE Folder Path",  value=os.path.join("data", "race", "RACE"))
        squadTrainPath = st.text_input("SQuAD Train JSON",  value=os.path.join("data", "squad", "train-v1.1.json"), key="squad2")
        squadDevPath   = st.text_input("SQuAD Dev JSON",    value=os.path.join("data", "squad", "dev-v1.1.json"),   key="squad3")

        if st.button("Generate Exam Paper", key="gen_dataset"):
            missingPaths = []
            if not os.path.isdir(t5ModelPath) or not os.listdir(t5ModelPath):
                missingPaths.append(f"T5 model folder: `{t5ModelPath}`")
            if not os.path.isdir(raceFolderPath):
                missingPaths.append(f"RACE folder: `{raceFolderPath}`")
            if not os.path.exists(squadTrainPath):
                missingPaths.append(f"SQuAD train: `{squadTrainPath}`")
            if not os.path.exists(squadDevPath):
                missingPaths.append(f"SQuAD dev: `{squadDevPath}`")

            if missingPaths:
                st.error("Missing files/folders:\n" + "\n".join(missingPaths))
            else:
                with st.spinner("Loading datasets and generating questions..."):
                    try:
                        from qp_generator import loadSquadLocal, loadRaceLocal, generateExamPaper
                        t5Tokenizer, t5Model = getSavedT5Model()
                        if t5Model is not None:
                            squadData = loadSquadLocal(squadTrainPath, squadDevPath)
                            raceData  = loadRaceLocal(raceFolderPath)
                            examPaper = generateExamPaper(
                                squadData, raceData, t5Model, t5Tokenizer,
                                mcqCount=mcqCount,
                                shortCount=shortCount,
                                descCount=descCount
                            )
                            displayFullExamPaper(examPaper)
                    except Exception as e:
                        st.error(f"Generation error: {e}")


# ============================================================
# RENDER HELPERS FOR QP GENERATOR
# ============================================================

def displayPassageExamPaper(examPaper):
    """Render short-answer + descriptive only (passage / custom text mode)."""
    shortAnswers = examPaper.get("shortAnswerQuestions", [])
    descriptive  = examPaper.get("descriptiveQuestions", [])

    st.success(
        f"Generated: {len(shortAnswers)} Short Answer | {len(descriptive)} Descriptive"
    )

    if shortAnswers:
        st.markdown("---")
        st.subheader("Section A — Short Answer Questions")
        for i, qa in enumerate(shortAnswers, 1):
            st.markdown(f"**Q{i}.** {qa['question']}")

    if descriptive:
        st.markdown("---")
        st.subheader("Section B — Descriptive Questions")
        for i, item in enumerate(descriptive, 1):
            st.markdown(f"**Q{i}.** {item['question']}")

    st.markdown("---")
    exportPaper = {
        "shortAnswerQuestions": shortAnswers,
        "descriptiveQuestions": descriptive
    }
    examJson = json.dumps(exportPaper, indent=2)
    st.download_button(
        label="Download Exam Paper (JSON)",
        data=examJson,
        file_name="exam_paper.json",
        mime="application/json"
    )


def displayFullExamPaper(examPaper):
    """Render all three sections (dataset mode — MCQ + short + descriptive)."""
    mcqs         = examPaper.get("mcqs", [])
    shortAnswers = examPaper.get("shortAnswerQuestions", [])
    descriptive  = examPaper.get("descriptiveQuestions", [])

    st.success(
        f"Generated: {len(mcqs)} MCQs | "
        f"{len(shortAnswers)} Short Answer | "
        f"{len(descriptive)} Descriptive"
    )

    if mcqs:
        st.markdown("---")
        st.subheader("Section A — Multiple Choice Questions")
        for i, mcq in enumerate(mcqs, 1):
            with st.expander(f"Q{i}: {mcq['question']}"):
                for j, opt in enumerate(mcq["options"]):
                    st.write(f"  {chr(65+j)}) {opt}")

    if shortAnswers:
        st.markdown("---")
        st.subheader("Section B — Short Answer Questions")
        for i, qa in enumerate(shortAnswers, 1):
            st.markdown(f"**Q{i}.** {qa['question']}")

    if descriptive:
        st.markdown("---")
        st.subheader("Section C — Descriptive Questions")
        for i, item in enumerate(descriptive, 1):
            st.markdown(f"**Q{i}.** {item['question']}")

    st.markdown("---")
    examJson = json.dumps(examPaper, indent=2)
    st.download_button(
        label="Download Exam Paper (JSON)",
        data=examJson,
        file_name="exam_paper.json",
        mime="application/json"
    )


# ============================================================
# MAIN ROUTER
# ============================================================

if module == "Answer Evaluator":
    runAnswerEvaluator()
elif module == "BTL Classifier":
    runBtlClassifier()
elif module == "QP Generator":
    runQpGenerator()