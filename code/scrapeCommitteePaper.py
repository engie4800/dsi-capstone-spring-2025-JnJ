import os
import pandas as pd
import requests

csv_file = "./guidance_decisionACR_paperlink.csv"
df = pd.read_csv(csv_file)

# folder to save PDFs
save_folder = "downloaded_committee_papers"
os.makedirs(save_folder, exist_ok=True)

# Check if a URL exists
def is_valid_link(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

# Download PDF
def download_pdf(url, filename):
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            file_path = os.path.join(save_folder, filename)
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed: {filename} (Status {response.status_code})")
    except requests.RequestException:
        print(f"Error downloading: {filename}")

# Process each link and update DataFrame
valid_links = []
for index, row in df.iterrows():
    pdf_url = row["Committee Paper URL"]
    file_name = row["Title"].strip().replace(" ", "_") + ".pdf"
    
    valid = is_valid_link(pdf_url)
    valid_links.append(valid)
    
    if valid:
        download_pdf(pdf_url, file_name)
    else:
        print(f"Invalid link: {pdf_url}")

# Add results to the original DataFrame
df["is_valid_link"] = valid_links

df.to_csv("./guidance_decisionACR_paperlink_validation.csv", index=False)
print("CSV updated.")
