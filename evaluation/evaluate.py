import requests
import numpy as np
from fuzzywuzzy import fuzz
import json
from typing import List, Dict

API_URL = "https://shl-recommendation-system-yoow.onrender.com"

# Test queries with ground-truth relevant assessment names
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
    # Add more test cases if needed
]

def normalize_name(name: str) -> str:
    """Normalize assessment name for better comparison"""
    return name.lower().replace("(new)", "").replace("-", " ").strip()

def is_similar(a: str, b: str, threshold: int = 80) -> bool:
    """Check if two assessment names are similar using fuzzy matching"""
    a_norm = normalize_name(a)
    b_norm = normalize_name(b)
    
    # Direct substring match
    if a_norm in b_norm or b_norm in a_norm:
        return True
    
    # Fuzzy token matching (better for rearranged words)
    ratio = fuzz.token_sort_ratio(a_norm, b_norm)
    return ratio >= threshold

def recall_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """Calculate recall@k with fuzzy matching"""
    recommended_top_k = recommended[:k]
    hits = sum(1 for item in recommended_top_k if any(is_similar(item, rel) for rel in relevant))
    return hits / len(relevant) if relevant else 0

def precision_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """Calculate precision@k with fuzzy matching"""
    recommended_top_k = recommended[:k]
    hits = sum(1 for item in recommended_top_k if any(is_similar(item, rel) for rel in relevant))
    return hits / len(recommended_top_k) if recommended_top_k else 0

def evaluate_query(query: str, relevant: List[str], k_values: List[int] = [3, 5, 10]) -> Dict[int, Dict[str, float]]:
    """Evaluate a single query for multiple k values"""
    try:
        payload = {"query": query}
        response = requests.post(API_URL, json=payload, params={"top_k": max(k_values)}, timeout=90)
        
        if response.status_code == 200:
            results = response.json()
            recommended_names = [item["name"] for item in results]
            
            print(f"Top {min(3, len(recommended_names))} recommended: {recommended_names[:3]}")
            print(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
            
            metrics = {}
            for k in k_values:
                if k <= len(recommended_names):
                    recall = recall_at_k(recommended_names, relevant, k)
                    precision = precision_at_k(recommended_names, relevant, k)
                    
                    metrics[k] = {
                        "recall": recall,
                        "precision": precision,
                    }
                    
                    print(f"k={k}: Recall={recall:.2f}, Precision={precision:.2f}")
            
            # Analyze incorrect results
            if k_values[0] <= len(recommended_names):
                k = k_values[0]  # Use the smallest k for error analysis
                false_positives = [r for r in recommended_names[:k] 
                                  if not any(is_similar(r, rel) for rel in relevant)]
                false_negatives = [rel for rel in relevant 
                                  if not any(is_similar(rel, r) for r in recommended_names[:k])]
                
                if false_positives:
                    print(f"Incorrect recommendations: {false_positives}")
                if false_negatives:
                    print(f"Missing relevant assessments: {false_negatives}")
            
            print("")
            return metrics
            
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return {}
            
    except Exception as e:
        print(f"Error evaluating query: {e}")
        return {}

def main():
    k_values = [3, 5, 10]
    all_metrics = {k: {"recall": [], "precision": []} for k in k_values}
    
    print(f"Evaluating against API: {API_URL}\n")
    
    for test in TEST_QUERIES:
        query_metrics = evaluate_query(test["query"], test["relevant"], k_values)
        
        # Aggregate metrics
        for k, metrics in query_metrics.items():
            for metric_name, value in metrics.items():
                all_metrics[k][metric_name].append(value)
    
    # Print summary statistics
    print("\n=== EVALUATION SUMMARY ===")
    for k in k_values:
        metrics = all_metrics[k]
        if metrics["recall"]:  # Check if we have any data
            mean_recall = np.mean(metrics["recall"])
            mean_precision = np.mean(metrics["precision"])
            
            print(f"=== Performance at k={k} ===")
            print(f"Mean Recall@{k}: {mean_recall:.2f}")
            print("")
    
    # Save results to file
    try:
        with open("evaluation_results.json", "w") as f:
            json.dump(all_metrics, f, indent=2)
        print("Results saved to evaluation_results.json")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()