import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "https://www.nice.org.uk"

# Get individual guidance links
def get_guidance_links(page_url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(page_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    guidance_links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.startswith("https://www.nice.org.uk/guidance/ta"):
            guidance_links.append(href)
    
    return guidance_links

# Scrape title, recommendation text, and decision from individual guidance pages
def scrape_guidance_details(guidance_url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(guidance_url + "/chapter/1-Recommendation", headers=headers)
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    title_tag = soup.find("h1")
    if title_tag:
        title = title_tag.get_text(strip=True)
    else:
        title = "Title Not Found"
    
    # Extract recommendations from <article class="numbered-paragraph recommendation">
    recommendations = []
    for article in soup.find_all("article", class_="numbered-paragraph recommendation"):
        rec_body = article.find("div", class_="recommendation__body")
        if rec_body:
            full_text = " ".join(p.get_text(strip=True) for p in rec_body.find_all("p"))
            recommendations.append(full_text)
    
    # Extract recommendations from <p class="numbered-paragraph>"
    for paragraph in soup.find_all("p", class_="numbered-paragraph"):
        recommendations.append(paragraph.get_text(strip=True))

    # Join all text together
    recommendations_text = " | ".join(recommendations) if recommendations else "Recommendation Not Found"
    
    decision = "null"

    if " is recommended" in recommendations_text:
        decision = "Recommend"
    elif " not recommended" in recommendations_text:
        decision = "Not Recommend"
    else:
        decision = "Unable to Make Any Recommendation"

    return title, recommendations_text, decision


if __name__ == "__main__":
    guidances_url = "https://www.nice.org.uk/guidance/published?ngt=Technology%20appraisal%20guidance&ps=9999"
    guidance_links = get_guidance_links(guidances_url)
    # print("Extracted Links:", guidance_links)

    data = []
    for link in guidance_links:
        title, recommendations, decision = scrape_guidance_details(link)
        data.append({"Title": title, "Recommendation text": recommendations, "Decision": decision})

    df = pd.DataFrame(data)

    df.to_csv("./guidance_decision.csv", index=False)

    print(df.head)