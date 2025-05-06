import json
from typing import List, Dict

def load_shl_data(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def prepare_text_for_embedding(assessment: Dict) -> str:
    name = assessment.get("name", "")
    description = assessment.get("description", "")
    test_type = assessment.get("test_type", "")
    duration = assessment.get("duration", "")
    job_level = assessment.get("job_level", "")
    return (
        f"Assessment Name: {name}. "
        f"Description: {description}. "
        f"Test Type: {test_type}. "
        f"Duration: {duration}. "
        f"Job Level: {job_level}."
    )

def get_all_texts_for_embedding(data: List[Dict]) -> List[str]:
    return [prepare_text_for_embedding(a) for a in data]