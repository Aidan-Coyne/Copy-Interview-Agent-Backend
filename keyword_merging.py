from sentence_transformers import SentenceTransformer, util

# Initialize the model once (you can choose a different model if needed)
model = SentenceTransformer('all-MiniLM-L6-v2')

def merge_keywords(cv_keywords: list, context_keywords: list, threshold: float = 0.7) -> list:
    """
    Merge and filter keywords from a CV and context (company/job) based on semantic similarity.
    
    Args:
        cv_keywords (list): List of keywords extracted from the CV.
        context_keywords (list): List of keywords extracted from the company/job context.
        threshold (float, optional): Similarity threshold for considering a CV keyword relevant. Defaults to 0.7.
    
    Returns:
        list: A merged list of keywords that are relevant to both the CV and the context.
    """
    if not cv_keywords or not context_keywords:
        # If one list is empty, simply return the non-empty one.
        return cv_keywords or context_keywords

    # Compute embeddings for each list of keywords
    cv_embeddings = model.encode(cv_keywords, convert_to_tensor=True)
    context_embeddings = model.encode(context_keywords, convert_to_tensor=True)

    filtered_cv_keywords = []
    # For each keyword from the CV, check if it is similar to any context keyword
    for i, cv_kw in enumerate(cv_keywords):
        similarities = util.pytorch_cos_sim(cv_embeddings[i], context_embeddings)
        if similarities.max() >= threshold:
            filtered_cv_keywords.append(cv_kw)

    # Merge filtered CV keywords with context keywords (taking union, avoiding duplicates)
    merged_keywords = list(set(filtered_cv_keywords + context_keywords))
    return merged_keywords

# Example usage:
if __name__ == "__main__":
    # Example keywords from a CV and from context (company/job)
    cv_keywords = ["automation", "data analysis", "CAD", "robotics", "innovation", "efficiency"]
    context_keywords = ["electric vehicles", "innovation", "automation", "sustainable technology"]

    merged = merge_keywords(cv_keywords, context_keywords, threshold=0.7)
    print("Merged Keywords:", merged)
