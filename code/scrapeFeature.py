import pymupdf as fitz  # Correct import for PyMuPDF
import pandas as pd
import re
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the PDF and extract text from pages 3 to 138
pdf_path = "./committee-papers-Anhydrous sodium thiosulfate-Recommend.pdf"
doc = fitz.open(pdf_path)

# Define the set of questions
questions = [
    "Does the Drug/Medical technology bring innovation?",
    "Is it an old HT applying for a new indication or a new HT?",
    "Is this the second submission for this HT?",
    "What is the type of HT (Drug, Medical Device, Other treatment)?",
    "Was an appropriate comparator used?",
    "In which country was the HTA carried out?",
    "What type of disease does the applied HT try to help?",
    "Is the HT for a paediatric population?",
    "What is the disease severity?",
    "Are there clinical uncertainties in the literature/studies submitted?",
    "Does the HT try to treat an unmet need?",
    "What year was the application and decision?",
    "Was it in the time of a big crisis/change (e.g., COVID, BREXIT)?",
    "Was the efficacy/effectiveness of the HT acceptable? Was it high or low?",
    "Was there a QALY benefit? High, medium, low?",
    "Did the HT submit a RCT (Randomized Controlled Trial)?",
    "Was the Quality of Evidence high?",
    "Were there surrogate endpoints used, or hard ones?",
    "Was there an indirect or direct comparison to the alternative?",
    "How many patients did the submitted literature have?",
    "Did they perform a Meta-Analysis?",
    "How many RCTs did they submit?",
    "How many Observational studies did they submit?",
    "Was there a comparator in the submitted studies?",
    "Was there an economic (cost-effectiveness) analysis?",
    "What was the ICER (Incremental Cost-Effectiveness Ratio)?",
    "What was the total intervention's expenditure estimate (Budget Impact) for the country?"
]

text = ""

footer_pattern = re.compile(r"Company evidence submission for anhydrous sodium thiosulfate.*?Page \d+ of \d+", re.DOTALL)
reference_pattern = re.compile(r"(\.\s*\d+([,-]\d+)*)(?=\s|$)")  
dash_reference_pattern = re.compile(r"(\–\s*\d+([,-]\d+)*)(?=\s|$)")

for page_num in range(12, 138):  # Pages are 0-indexed, so Page 3 is index 2
    page_text = doc[page_num].get_text("text")
    page_text = re.sub(footer_pattern, "", page_text)  # Remove footers
    page_text = re.sub(reference_pattern, ".", page_text)  # Remove reference numbers
    page_text = re.sub(dash_reference_pattern, ".", page_text)  # Remove –4-8 cases
    text += page_text + "\n"

# Convert extracted text into sentences
sentences = text.split(". ")

# Create a DataFrame with one sentence per row
df_sentences = pd.DataFrame({"Sentence": sentences})

# Save to CSV
df_sentences.to_csv("./pdf_sentences.csv", index=False)

# Search for the best matching sentence for each question
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)  # Convert sentences to vectors

answers = []
for question in questions:
    question_vector = vectorizer.transform([question])  # Convert question to vector
    scores = np.dot(question_vector, X.T).toarray().flatten()  # Compute cosine similarity
    best_sentence = sentences[np.argmax(scores)]  # Select highest match
    answers.append(best_sentence)

# Create a DataFrame with the extracted answers
df_answers = pd.DataFrame({"Question": questions, "Answer": answers})

# Save answers to CSV file
df_answers.to_csv("./extracted_answers_tfidf.csv", index=False)

print(df_answers)

import tensorflow as tf
from sentence_transformers import SentenceTransformer, util
print("NumPy version:", np.__version__)
print("TensorFlow version:", tf.__version__)
model = SentenceTransformer("all-MiniLM-L6-v2")  # Load SBERT
print("SentenceTransformer loaded successfully!")

# Convert sentences and questions into embeddings
sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Find best match for each question
answers = []
for i, question in enumerate(questions):
    query_embedding = question_embeddings[i].unsqueeze(0)  # Reshape for comparison
    scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
    best_sentence = sentences[scores.argmax().item()]  # Select most relevant sentence
    answers.append(best_sentence)

# Create a DataFrame with the extracted answers
df_answers_sbert = pd.DataFrame({"Question": questions, "Answer": answers})

# Save answers to CSV file
df_answers_sbert.to_csv("./extracted_answers_sbert.csv", index=False)

print(df_answers_sbert)