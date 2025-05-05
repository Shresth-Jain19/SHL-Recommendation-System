# SHL Assessment Recommendation System

![SHL Recommender](https://img.shields.io/badge/AI-Powered-brightgreen) ![Python](https://img.shields.io/badge/Python-3.12+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688) ![ChromaDB](https://img.shields.io/badge/ChromaDB-Latest-purple)

An AI-powered system for recommending SHL assessments based on job descriptions, using vector search and natural language processing.

## 🔗 Important Links

- **Streamlit App**: [https://shl-assessment-recommendation-system-shresth-jn.streamlit.app/](https://shl-assessment-recommendation-system-shresth-jn.streamlit.app/)
- **API Endpoint**: [https://shl-recommendation-system-yoow.onrender.com](https://shl-recommendation-system-yoow.onrender.com)
- **Project Documentation**: [https://drive.google.com/file/d/12vaBVp5QY2OcSWhAiUXSsV_xho4KUkMO/view](https://drive.google.com/file/d/12vaBVp5QY2OcSWhAiUXSsV_xho4KUkMO/view)

## 🚀 Features

- **Natural Language Search**: Enter job descriptions in plain text format
- **URL Parsing**: Automatically extract relevant details from job posting URLs
- **Vector Similarity**: Find the most semantically relevant assessments
- **User-Friendly Interface**: Clean Streamlit frontend for easy interaction
- **API Access**: Fully documented FastAPI backend for programmatic access

## 📊 Evaluation Results

The system was evaluated using realistic job description queries and URL parsing tests against a set of ground-truth relevant assessments. The evaluation uses fuzzy matching with normalized names to account for slight variations in assessment naming.

### Evaluation Methodology:
- **Test Queries**: Multiple test queries including direct text descriptions and job URLs
- **Ground Truth**: Each query has predefined relevant assessment names
- **Metrics**: Recall, Precision, Average Precision (AP), NDCG, and F1 scores
- **Fuzzy Matching**: Names are normalized and compared with fuzzy matching (threshold 80%)
- **K Values**: The system was evaluated at K=3, K=5, and K=10 recommendations

### Evaluation Results:

#### For K=3 Recommendations:
- **Recall@3**: 0.57 (percentage of relevant assessments found in top 3)
- **Precision@3**: 1.0 (all recommendations were relevant)
- **Average Precision@3**: 1.0 (perfect ranking of relevant items)
- **NDCG@3**: 1.0 (perfect ranking considering position)
- **F1 Score@3**: 0.545 (harmonic mean of precision and recall)

#### For K=5 Recommendations:
- **Recall@5**: 0.625 (percentage of relevant assessments found in top 5)
- **Precision@5**: 1.0 (all recommendations were relevant)
- **Average Precision@5**: 1.0 (perfect ranking of relevant items)
- **NDCG@5**: 1.0 (perfect ranking considering position)
- **F1 Score@5**: 0.769 (harmonic mean of precision and recall)

#### For K=10 Recommendations:
- **Recall@10**: 0.875 (percentage of relevant assessments found in top 10)
- **Precision@10**: 0.7 (70% of recommendations were relevant)
- **Average Precision@10**: 0.847 (ranking quality score)
- **NDCG@10**: 0.912 (ranking quality considering position)
- **F1 Score@10**: 0.778 (harmonic mean of precision and recall)

The results demonstrate that the system achieves high precision at all thresholds, with recall increasing as K increases. The perfect NDCG scores at K=3 and K=5 indicate optimal ranking of the relevant assessments.

## 🔧 Technology Stack

- **Backend**: FastAPI, ChromaDB, Sentence-Transformers
- **NLP**: Google's Gemini AI for URL parsing and context understanding
- **Vector Search**: ChromaDB with MPNet embeddings
- **Frontend**: Streamlit with responsive design
- **Data**: Curated SHL assessment catalog

## 📋 Prerequisites

- Python 3.12+
- Google Gemini API key

## 💻 Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/SHL-Recommendation-System.git
   cd SHL-Recommendation-System
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API key
   ```bash
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

4. Build the vector database
   ```bash
   python build_chroma_db.py
   ```

## 🏃‍♂️ Running the Application

1. Start the API server
   ```bash
   uvicorn app.main:app --reload
   ```

2. Launch the frontend (in a new terminal)
   ```bash
   streamlit run frontend/streamlit_app.py
   ```

3. Open your browser and navigate to http://localhost:8501

## 🔍 Usage

1. **Text Query**: Enter a job description or requirements in the text area
2. **URL Query**: Paste a public job posting URL for automatic analysis
3. **Configure Results**: Adjust the number of recommendations (1-10)
4. **View Recommendations**: Explore SHL assessments with direct links

## 🏗️ Architecture

```
SHL-Recommendation-System/
├── app/                     # Backend application code
│   ├── data_loader.py       # Loads and preprocesses SHL assessment data
│   ├── embeddings.py        # Embedding utilities for semantic search
│   ├── gemini_utils.py      # Google Gemini API integration and prompt handling
│   ├── main.py              # FastAPI application with /recommend endpoint
│   └── recommender.py       # Core logic for retrieving and ranking recommendations
│
├── data/                    # Local data storage
│   ├── chroma_db/           # ChromaDB vector database files
│   └── SHL_RAW.json         # Raw assessment catalog from SHL
│
├── frontend/                # Streamlit UI for interactive usage
│   └── streamlit_app.py     # Frontend interface for querying and viewing results
│
├── build_chroma_db.py       # One-time script to embed SHL data into ChromaDB
├── requirements.txt         # Python dependencies
└── .env                     # API keys and environment variables
```

## 🧠 Implementation Details

- **Vector Database**: Assessment descriptions are encoded using the MPNet model and stored in ChromaDB for semantic search
- **Query Processing**: Job descriptions are analyzed semantically to extract key skills and requirements
- **URL Processing**: Gemini AI parses job URLs to extract relevant information for assessment matching
- **Top-K Search**: Cosine similarity between query and assessment embeddings determines the best matches

## 🔜 Future Improvements

- Custom assessment filtering based on specific criteria
- Advanced analytics dashboard for employers
- Multilingual support for international job descriptions
- Integration with ATS (Applicant Tracking Systems)
- Custom assessment sequences for different hiring stages

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Built with ❤️ by Shresth Jain
