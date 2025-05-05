import requests

API_URL = "http://localhost:8000/recommend"

# Example test queries and their ground-truth relevant assessment names (fill these with your actual data)
TEST_QUERIES = [
    {
        "query": "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
        "relevant": ["Java 8 (New)", "Core Java (Advanced Level) (New)", "Core Java (Entry Level) (New)", "Java 2 Platform Enterprise Edition 1.4 Fundamental", "Java Platform Enterprise Edition 7 (Java EE 7)", "Java Design Patterns (New)", "Java Frameworks (New)", "Java Web Services (New)"]
    },
    {
        "query": "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes.",
        "relevant": ["Python (New)", "SQL (New)", "JavaScript (New)"]
    },
    {
        "query": "https://jobs.zs.com/jobs/21819?lang=en-us",
        "relevant": ["Python (New)", "SQL (New)", "Microsoft Excel 365 (New)", "MS Excel (New)"]
    },
]

def recall_at_k(recommended, relevant, k):
    recommended_top_k = recommended[:k]
    hits = sum(1 for item in recommended_top_k if item in relevant)
    return hits / len(relevant) if relevant else 0

def average_precision_at_k(recommended, relevant, k):
    score = 0.0
    hits = 0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k) if relevant else 0

def main():
    recall_scores = []
    map_scores = []
    k = 3

    for test in TEST_QUERIES:
        payload = {"query": test["query"]}
        response = requests.post(API_URL, json=payload, params={"top_k": k})
        if response.status_code == 200:
            results = response.json()
            recommended_names = [item["name"] for item in results]
            # print("Top 3 recommended:", recommended_names)
            recall = recall_at_k(recommended_names, test["relevant"], k)
            ap = average_precision_at_k(recommended_names, test["relevant"], k)
            recall_scores.append(recall)
            map_scores.append(ap)
            print(f"Query: {test['query']}\nRecall@{k}: {recall:.2f}, AP@{k}: {ap:.2f}\n")
        else:
            print(f"API error for query: {test['query']}")

    print(f"Mean Recall@{k}: {sum(recall_scores)/len(recall_scores):.2f}")
    print(f"MAP@{k}: {sum(map_scores)/len(map_scores):.2f}")

if __name__ == "__main__":
    main()