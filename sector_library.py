import logging
from fuzzywuzzy import fuzz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sector_library = {
    "Automotive": [
        "automotive", "vehicles", "cars", "EV", "electric vehicle", "OEM", "motorsport", "drivetrain",
        "ADAS", "autonomous driving", "chassis", "powertrain", "automobile", "transportation"
    ],
    "Aerospace & Defence": [
        "aerospace", "aviation", "aircraft", "defence", "missiles", "satellites", "aerodynamics", "flight control",
        "military", "space systems", "UAV", "propulsion", "jet engines"
    ],
    "Rail & Transportation": [
        "rail", "railway", "track design", "train", "rolling stock", "infrastructure", "metro", "transport authority",
        "light rail", "transportation systems"
    ],
    "Energy & Power": [
        "energy", "power systems", "smart grid", "renewables", "solar", "wind", "battery storage", "electricity",
        "nuclear", "offshore wind", "power plant", "hydroelectric", "utilities"
    ],
    "Oil, Gas & Petrochemical": [
        "oil", "gas", "petroleum", "offshore", "rigs", "pipelines", "chemical refinery", "LNG", "downstream", "upstream",
        "reservoir", "exploration", "drilling"
    ],
    "Construction & Infrastructure": [
        "construction", "infrastructure", "civil engineering", "structural", "piling", "bridges", "foundations",
        "roads", "contracting", "planning applications", "urban development"
    ],
    "Technology & Software": [
        "technology", "software", "web development", "mobile apps", "cloud", "machine learning", "AI",
        "data science", "SaaS", "DevOps", "cybersecurity", "platform", "full stack"
    ],
    "Banking & Finance": [
        "banking", "investment", "financial services", "capital markets", "retail banking", "private equity",
        "hedge fund", "asset management", "compliance", "mortgages", "trading", "risk", "IPO", "wealth management"
    ],
    "Consulting & Advisory": [
        "consulting", "strategy", "management consultancy", "advisory", "business transformation",
        "process improvement", "digital transformation", "client engagement", "solutions", "stakeholder management"
    ],
    "Healthcare & Life Sciences": [
        "healthcare", "pharmaceutical", "clinical trials", "medical device", "life sciences", "biotech",
        "patient care", "diagnostics", "biomedical", "regulatory affairs", "health tech"
    ],
    "Retail & FMCG": [
        "retail", "consumer goods", "FMCG", "supply chain", "logistics", "inventory", "wholesale",
        "store operations", "ecommerce", "online retail", "product lifecycle"
    ],
    "Education & Research": [
        "education", "university", "teaching", "research", "academic", "curriculum", "learning platforms",
        "e-learning", "student engagement", "lecturing", "training", "STEM outreach"
    ],
    "Logistics & Supply Chain": [
        "logistics", "freight", "supply chain", "distribution", "inventory control", "ERP systems",
        "demand forecasting", "warehouse", "procurement", "transportation management"
    ],
    "Telecommunications": [
        "telecom", "broadband", "5G", "wireless", "satellite", "networks", "data transmission", "fiber optics",
        "network infrastructure", "ISP", "signal processing"
    ],
    "Manufacturing": [
        "manufacturing", "production", "factory", "plant", "assembly line", "lean", "six sigma", "machining",
        "fabrication", "CNC", "operations", "industrial engineering"
    ],
    "Environment & Sustainability": [
        "sustainability", "environmental", "carbon footprint", "climate change", "ESG", "green tech",
        "waste management", "water treatment", "pollution control", "environmental consulting"
    ],
    "Public Sector & Government": [
        "government", "public sector", "policy", "infrastructure projects", "local council", "defence contracts",
        "regulatory", "municipal planning", "public works", "transport funding"
    ],
    "Media & Entertainment": [
        "media", "film", "TV", "streaming", "entertainment", "broadcast", "editing", "production",
        "animation", "games development", "XR", "AR/VR", "post-production"
    ]
}

def get_company_sector(company_info: dict) -> str:
    """
    Determines the company sector by comparing search result snippets (from company_info)
    against the keywords in sector_library using fuzzy matching.
    
    Args:
        company_info (dict): A dictionary that contains a "search_results" key with a list of results.
        
    Returns:
        str: A comma-separated string of matched sectors, or None if no match is found.
    """
    matched_sectors = set()
    search_results = company_info.get("search_results", [])
    for item in search_results:
        snippet = item.get("snippet", "").lower()
        for sector, keywords in sector_library.items():
            for keyword in keywords:
                if fuzz.partial_ratio(keyword.lower(), snippet) > 75:
                    matched_sectors.add(sector)
    return ", ".join(sorted(matched_sectors)) if matched_sectors else None
