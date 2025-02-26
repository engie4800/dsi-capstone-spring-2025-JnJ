import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import pdfplumber
import os

def store_decision_dates():
    pdf_df = pd.read_csv("./guidance_decisionACR_paperlink_validation.csv")

    df_basic_features = pdf_df[["Title", "Decision"]]

    decision_dates = []

    for guidance in pdf_df["Guidance URL"]:
        decision_date = scrape_decision_date(guidance)
        decision_dates.append(decision_date)

    df_basic_features['Decision_Date'] = decision_dates
    # print(df_basic_features.head())
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
    basic_features_df.to_csv("basic_features_v2.csv", index = False)
    print("Updated CSV saved as 'basic_features_v2.csv'.")


if __name__ == "__main__":
    # store_decision_dates()
    store_application_date()