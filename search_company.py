import os
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

if not BRAVE_API_KEY:
    raise ValueError("API Key not found! Make sure you set BRAVE_API_KEY in your environment variables.")

def search_company_info(company_name: str, job_role: str):
    """
    Searches the Brave Search API for company information.
    
    Args:
        company_name: The name of the company.
        job_role: The job role the user is applying for.
        
    Returns:
        A JSON string containing raw search results and a curated list of keywords extracted from the combined snippet text.
    """
    # Refined search query using quotation marks around company name
    search_query = f'about "{company_name}" company profile, values, products, sectors, future plans UK'
    search_url = f"https://api.search.brave.com/res/v1/web/search?q={search_query}"

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY
    }

    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        web_results = []
        if "web" in data and "results" in data["web"]:
            for result in data["web"]["results"][:5]:
                title = result.get("title", "No title")
                url = result.get("url", "No URL")
                snippet = result.get("description", "No description")
                web_results.append({"title": title, "url": url, "snippet": snippet})

        # Combine all snippet text for advanced keyword extraction.
        combined_text = " ".join([result["snippet"] for result in web_results if result.get("snippet")])
        
        # Import and use the keyword_extraction module (ensure it's installed and in your path)
        from keyword_extraction import extract_keywords
        extracted_keywords = extract_keywords(combined_text, top_n=10) if combined_text else []

        result_data = {
            "search_results": web_results,
            "extracted_keywords": extracted_keywords
        }

        return json.dumps(result_data) if result_data else json.dumps({"message": "No relevant company information found."})

    except requests.exceptions.RequestException as e:
        logging.error(f"Request error: {e}")
        return json.dumps({"error": f"Request error: {e}"})
    except json.JSONDecodeError:
        logging.error("Invalid JSON response from API.")
        return json.dumps({"error": "Invalid JSON response from API."})
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return json.dumps({"error": f"An unexpected error occurred: {e}"})
