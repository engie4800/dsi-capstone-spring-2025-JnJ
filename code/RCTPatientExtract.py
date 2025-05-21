import os
import fitz
import pdfplumber
from PIL import Image
from io import BytesIO
import base64
import openai
import pandas as pd
import re
import time

def count_pdfs_with_multiple_phrase_occurrences(folder_path, phrase, min_occurrences=2):
    phrase = phrase.lower().strip()
    count = 0
    total = 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            total += 1
            file_path = os.path.join(folder_path, filename)

            try:
                with fitz.open(file_path) as doc:
                    full_text = ""
                    for page in doc:
                        full_text += page.get_text()

                # Convert to lowercase and count phrase occurrences
                text_lower = full_text.lower()
                occurrence_count = text_lower.count(phrase)

                if occurrence_count >= min_occurrences:
                    count += 1
                    print(count)

            except Exception as e:
                print(f"Error reading {filename}: {e}")

    print(f"\nFound the phrase at least {min_occurrences} times in {count} out of {total} PDFs.")
    return count

"""
# Example usage
folder_path = "./sampledocs"
count_pdfs_with_multiple_phrase_occurrences(
    folder_path,
    phrase="Statistical analysis and definition of study groups in",
    min_occurrences=1
) # end up with 353 / 483 PDFs
"""

openai.api_key = os.getenv("OPENAI_API_KEY")

def encode_image(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def find_last_occurrence_page(pdf_path, phrase):
    phrase = phrase.lower()
    last_occurrence = None
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and phrase in text.lower():
                last_occurrence = i
    return last_occurrence

def find_page_with_llm(pdf_path, phrase):
    page_num = None
    prompt = """Answer the following questions based on the content of the page:

    Question 1: Does this page contain a heading exactly called "Contents"?  
    → If so, immediately respond with "No", explain why in one sentence, and skip the next question.

    Otherwise, proceed to the next question:

    Question 2: Does this page contain a visible section header that clearly represents section B.2 titled 'Clinical effectiveness' (e.g., 'B.2. Clinical effectiveness' or similar)?  

    Final Rule:  
    If the answer to Question 2 is yes, respond only with **"Yes"**.  
    Otherwise, respond with **"No"** and explain why in one sentence.

    When responding with "Yes", do not respond with anything else.
    """
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and phrase in text and ("B.2 Company evidence submission" in text or "B.2. Company evidence submission" in text):
                page_num = i
                print(page_num)
    
                doc = fitz.open(pdf_path)
                page = doc[page_num]
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img = encode_image(img)
                images = []
                images.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img}"
                    }
                })

                response = openai.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt}
                        ] + images}
                    ],
                    max_tokens=300,
                )
                print(response.choices[0].message.content.strip())
                if response.choices[0].message.content.strip() == "Yes":
                    print(page_num)
                    return page_num
                else:
                    continue
    return None

def render_pages_as_images(pdf_path, start_page, num_pages=10, dpi=300):
    doc = fitz.open(pdf_path)
    end_page = min(start_page + num_pages, len(doc))
    images = []
    for page_num in range(start_page, end_page):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images

def ask_openai_for_rct_summary(encoded_images):
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "These pages are from a clinical document. Please identify the number of randomized controlled trials (RCTs) mentioned, and sum up the total number of patients across all RCTs. Return only the final numbers in this format:\n\nNumber of RCTs: <number>\nTotal number of patients: <number>"}
            ] + encoded_images}
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()

def parse_rct_summary(summary):
    rcts = None
    patients = None
    match = re.search(r"Number of RCTs:\s*(\d+).*?Total number of patients:\s*(\d+)", summary, re.DOTALL)
    if match:
        rcts = int(match.group(1))
        patients = int(match.group(2))
    return rcts, patients

def process_pdf_for_rcts(pdf_path):
    last_occurrence_page = find_last_occurrence_page(pdf_path, "Statistical analysis and definition of study groups in")
    images = []
    if last_occurrence_page:
        images = render_pages_as_images(pdf_path, last_occurrence_page, num_pages=10)

    clinical_evidence_page = find_page_with_llm(pdf_path, "Clinical effectiveness")
    additional_images = []
    if clinical_evidence_page:
        additional_images = render_pages_as_images(pdf_path, clinical_evidence_page, num_pages=5)

    images.extend(additional_images)
    if not images:
        return os.path.basename(pdf_path), None, None

    encoded_images = []
    for image in images:
        b64_image = encode_image(image)
        encoded_images.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64_image}"
            }
        })

    summary = ask_openai_for_rct_summary(encoded_images)
    time.sleep(1)
    rcts, patients = parse_rct_summary(summary)
    return os.path.basename(pdf_path), rcts, patients


# Example usage
folder_path = "./50_Committee_Papers"

results = []

for filename in os.listdir(folder_path):
    if filename.lower().endswith(".pdf"):
        file_path = os.path.join(folder_path, filename)
        pdf_name, rcts, patients = process_pdf_for_rcts(file_path)
        results.append({
            "PDF File": pdf_name,
            "Number of RCTs": rcts,
            "Number of Patients": patients
        })
    print(f"{filename} processed")

df_results = pd.DataFrame(results)
print(df_results)

# Optional: save to CSV
df_results.to_csv("rct_patients_50.csv", index=False)