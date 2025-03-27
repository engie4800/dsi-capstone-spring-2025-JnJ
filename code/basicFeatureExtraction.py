import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import pdfplumber
import os
from openai import OpenAI

def store_decision_dates():
    pdf_df = pd.read_csv("./guidance_decisionACR_paperlink_validation.csv")

    df_basic_features = pdf_df[["Title", "Decision"]]

    decision_dates = []

    for guidance in pdf_df["Guidance URL"]:
        decision_date = scrape_decision_date(guidance)
        decision_dates.append(decision_date)

    df_basic_features['Decision_Date'] = decision_dates
    df_basic_features.to_csv("./basic_features.csv", index = False)

def scrape_decision_date(link):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(link, headers=headers)
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    decision_time_tag = soup.find("time")
    datetime_value = None
    if decision_time_tag:
        datetime_value = decision_time_tag["datetime"]
    
    return datetime_value

def store_application_date():
    i = 0
    committee_paper_folder = "./downloaded_committee_papers"

    pdf_dates = {}

    for filename in os.listdir(committee_paper_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(committee_paper_folder, filename)
            extracted_date = extract_application_date_from_pdf(pdf_path)
            pdf_dates[filename] = extracted_date
            i = i + 1
            print(f"{i} pdfs processed")

    df = pd.DataFrame(pdf_dates.items(), columns=["PDF File", "Extracted Date"])
    df.to_csv("extracted_dates.csv", index=False)

    print("Extraction complete! Dates saved to extracted_dates.csv")

def extract_application_date_from_pdf(pdf_path):
    month_year_pattern = re.compile(r'^(January|February|March|April|May|June|July|August|September|October|November|December),?\s(\d{4})$')
    month_mapping = {
        "January": "01", "February": "02", "March": "03", "April": "04",
        "May": "05", "June": "06", "July": "07", "August": "08",
        "September": "09", "October": "10", "November": "11", "December": "12"
    }

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and "Company evidence submission" in text:
                    lines = text.split("\n")
                    for line in lines:
                        line = line.strip()
                        match = month_year_pattern.match(line)
                        if match:
                            year = match.group(2)
                            month = month_mapping[match.group(1)]
                            return f"{year}-{month}" 
            return None
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def incorporate_application_date():
    date_df = pd.read_csv("./extracted_dates.csv")
    basic_features_df = pd.read_csv("./basic_features.csv")
    basic_features_df["PDF File"] = basic_features_df["Title"].str.strip().replace(" ", "_", regex = True) + ".pdf"
    basic_features_df = basic_features_df.merge(date_df, on = "PDF File", how = "left")
    basic_features_df = basic_features_df.rename(columns={'Extracted Date': 'Application Date'})
    basic_features_df.to_csv("basic_features_v2.csv", index = False)
    print("Updated CSV saved as 'basic_features_v2.csv'.")

def classify_disease(title):
    # List of disease categories
    disease_categories = [
        "Cardiovascular", "Chronic", "Nervous System", "Obstetrics/Gynaecology", "Urinary-Track Diseases", 
        "Cancer", "Rare Disease (Orphan Drugs)", "Ultra-Rare Disease (Ultra Orphan Drugs)", "Musculoskeletal diseases"
    ]
    
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    prompt = f"""
    Classify the disease mentioned in the following title into one or more types from the given categories:

    Title: "{title}"

    Disease Categories: {", ".join(disease_categories)}

    Either return only the most relevant disease categories from the list or return "NULL" if there does not exist any suitable categories. Do not return extra words
    """

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return completion.choices[0].message.content.strip()

def classify_HT_via_LLM(title):
    ht_categories = ["Drug", "Medical Device", "Other treatment"]
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    prompt = f"""
    Classify the health technology mentioned in the following title into one type from the given categories:

    Title: "{title}"

    Disease Categories: {", ".join(ht_categories)}

    Either return only the most relevant health technology categories from the list or return NULL if there does not exist any suitable categories. Do not return extra words
    """

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return completion.choices[0].message.content.strip()

def classify_HT_via_KeywordSearch():
    pass

def incorporate_innovation():
    innovation_df = pd.read_csv("./innovation_percentage.csv")
    basic_features_df = pd.read_csv("./basic_features_v2.csv")
    innovation_df = innovation_df.rename(columns={'pdf_file': 'PDF File'})
    innovation_df = innovation_df.rename(columns={'percent_innovative': 'frac_eval_innovative'})
    basic_features_df = basic_features_df.merge(innovation_df[["PDF File", "frac_eval_innovative"]], on = "PDF File", how = "left")
    basic_features_df.to_csv("basic_features_v3.csv", index = False)

if __name__ == "__main__":
    store_decision_dates()
    store_application_date()
    incorporate_application_date()
    df = pd.read_csv("./basic_features_v2.csv")
    df["Disease_Category"] = df["Title"].apply(classify_disease)
    df["HT_Category"] = df["Title"].apply(classify_HT_via_LLM)
    # df.to_csv("./basic_features_v2.csv", index=False)
    incorporate_innovation()
    df_feature3 = pd.read_csv("./basic_features_v3.csv")
    df_url = pd.read_csv("guidance_decisionACR_paperlink_validation.csv")  # or use your actual DataFrame
    df_url["TA Code"] = df_url["Guidance URL"].str.extract(r'(ta\d+)$')
    df_feature4 = df_feature3.merge(df_url[["Title", "TA Code", "is_valid_link"]], on="Title", how="left")
    # df_feature4.to_csv("basic_features_v4.csv", index = False)