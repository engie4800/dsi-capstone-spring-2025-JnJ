import os
import openai
import pdfplumber
from PIL import Image
import base64
from io import BytesIO
import fitz
import pandas as pd
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

def encode_image(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def find_relevant_pages(pdf_path, keywords=["consider", "technology", "innovative", "potential"]):
    relevant_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and all(kw.lower() in text.lower() for kw in keywords):
                relevant_pages.append(i)
    return relevant_pages

def ask_openai_vision_pairwise(images):
    all_responses = []

    for i in range(0, len(images), 2):
        img_pair = images[i:i+2]

        # for img in img_pair:
        #     img.show()

        # Encode each image
        encoded_images = []
        for image in img_pair:
            b64_image = encode_image(image)
            encoded_images.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_image}"
                }
            })

        # Make one OpenAI API call per pair
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "How is this question 'Do you consider the technology to be innovative in its potential to make a significant and substantial impact on health-related benefits and how might it improve the way that current need is met?' responded in the images? The question may appear partially in image 1 and continue in image 2. Please extract and combine both image responses as a single answer. Return only the response."}
                ] + encoded_images}
            ],
            max_tokens=500,
        )

        content = response.choices[0].message.content
        all_responses.append(f"{i//2 + 1}. {content.strip()}")

    return "\n\n".join(all_responses)


def render_page_as_image(pdf_path, page_num, dpi=200):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def process_pdf(pdf_path):
    print(f"Scanning: {pdf_path}")
    relevant_pages = find_relevant_pages(pdf_path)
    if not relevant_pages:
        print("No relevant pages found.")
        return "NULL"

    # Convert relevant pages (and optionally next pages) to images using fitz
    selected_images = []
    doc = fitz.open(pdf_path)
    for page_num in relevant_pages:
        selected_images.append(render_page_as_image(pdf_path, page_num))
        if page_num + 1 < len(doc):  # optionally include next page
            selected_images.append(render_page_as_image(pdf_path, page_num + 1))
    doc.close()

    result = ask_openai_vision_pairwise(selected_images)
    return result

def analyze_innovation_percentage(evaluation_text, model="gpt-4-turbo"):
    if pd.isna(evaluation_text):
        return None

    prompt = f"""
    The following is a numbered list of evaluations, each of which evaluates whether the health technology is innovative in its potential to make a significant and substantial impact on health-related benefits.
    Simply return a single number between 0 and 1 representing the fraction of evaluations that are positive about its innovativeness (e.g., return 0.75 if 3 out of 4 are positive). Do not return anything else.

    Evaluations:
    {evaluation_text}
    """

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
        )
        raw_output = response.choices[0].message.content.strip()
        # print(raw_output)

        score = float(raw_output)
        if 0 <= score <= 1:
            return score
        else:
            return None

    except Exception as e:
        print("Error with OpenAI:", e)
        return None


if __name__ == "__main__":
    pdf_folder = "./downloaded_committee_papers"
    results = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            answer = process_pdf(pdf_path)
            results.append({
                "pdf_file": pdf_file,
                "extracted_answer": answer
            })

    df = pd.DataFrame(results)
    print(df)

    # df.to_csv("innovation_raw_answers.csv", index=False)

    df = pd.read_csv("./innovation_raw_answers.csv", na_values=["NULL"])    

    for i, row in df.iterrows():
        df.at[i, "percent_innovative"] = analyze_innovation_percentage(row["extracted_answer"])
        time.sleep(1)
    # df.to_csv("./innovation_percentage.csv")
    
