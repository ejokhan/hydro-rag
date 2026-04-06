"""
Create evaluation test set for Hydro-RAG
50 questions with expected answers and source papers.
"""
import json
import os

def create_test_questions():
    """
    Create test questions across domains.
    Each question has:
    - query: the question
    - domain: hydrology, healthcare, or llm
    - difficulty: easy, medium, hard
    - expected_keywords: words that MUST appear in retrieved chunks
    - expected_answer_keywords: words that should appear in the generated answer
    """
    questions = [
        # ---- HYDROLOGY: Easy (direct keyword match) ----
        {
            "id": "hydro_easy_01",
            "query": "What machine learning models are used for streamflow prediction?",
            "domain": "hydrology",
            "difficulty": "easy",
            "expected_keywords": ["streamflow", "prediction", "machine learning"],
            "expected_answer_keywords": ["LSTM", "neural network", "random forest", "streamflow"],
        },
        {
            "id": "hydro_easy_02",
            "query": "How is deep learning applied to flood forecasting?",
            "domain": "hydrology",
            "difficulty": "easy",
            "expected_keywords": ["flood", "forecasting", "deep learning"],
            "expected_answer_keywords": ["flood", "prediction", "neural"],
        },
        {
            "id": "hydro_easy_03",
            "query": "What methods are used for water quality anomaly detection?",
            "domain": "hydrology",
            "difficulty": "easy",
            "expected_keywords": ["water quality", "anomaly"],
            "expected_answer_keywords": ["anomaly", "detection", "water"],
        },
        {
            "id": "hydro_easy_04",
            "query": "How are neural networks used for rainfall runoff modeling?",
            "domain": "hydrology",
            "difficulty": "easy",
            "expected_keywords": ["rainfall", "runoff"],
            "expected_answer_keywords": ["rainfall", "runoff", "model"],
        },
        {
            "id": "hydro_easy_05",
            "query": "What is the role of LSTM in hydrological time series forecasting?",
            "domain": "hydrology",
            "difficulty": "easy",
            "expected_keywords": ["LSTM", "hydrological"],
            "expected_answer_keywords": ["LSTM", "time series", "forecast"],
        },
        # ---- HYDROLOGY: Medium (requires understanding) ----
        {
            "id": "hydro_med_01",
            "query": "What are the advantages of transformer models over LSTM for streamflow prediction?",
            "domain": "hydrology",
            "difficulty": "medium",
            "expected_keywords": ["transformer", "streamflow"],
            "expected_answer_keywords": ["transformer", "LSTM", "attention"],
        },
        {
            "id": "hydro_med_02",
            "query": "How can transfer learning be applied to ungauged basins?",
            "domain": "hydrology",
            "difficulty": "medium",
            "expected_keywords": ["transfer", "ungauged"],
            "expected_answer_keywords": ["transfer", "learning", "basin"],
        },
        {
            "id": "hydro_med_03",
            "query": "What preprocessing techniques improve groundwater level prediction accuracy?",
            "domain": "hydrology",
            "difficulty": "medium",
            "expected_keywords": ["groundwater", "prediction"],
            "expected_answer_keywords": ["groundwater", "prediction"],
        },
        {
            "id": "hydro_med_04",
            "query": "How do ensemble methods compare to single models for river discharge forecasting?",
            "domain": "hydrology",
            "difficulty": "medium",
            "expected_keywords": ["ensemble", "discharge"],
            "expected_answer_keywords": ["ensemble", "model"],
        },
        {
            "id": "hydro_med_05",
            "query": "What challenges exist in real-time flood prediction using machine learning?",
            "domain": "hydrology",
            "difficulty": "medium",
            "expected_keywords": ["flood", "real-time"],
            "expected_answer_keywords": ["challenge", "flood", "prediction"],
        },
        # ---- HYDROLOGY: Hard (multi-hop reasoning) ----
        {
            "id": "hydro_hard_01",
            "query": "How can variational mode decomposition improve LSTM-based streamflow forecasting?",
            "domain": "hydrology",
            "difficulty": "hard",
            "expected_keywords": ["variational", "decomposition", "LSTM"],
            "expected_answer_keywords": ["decomposition", "LSTM", "streamflow"],
        },
        {
            "id": "hydro_hard_02",
            "query": "What hybrid architectures combine physical models with deep learning for hydrology?",
            "domain": "hydrology",
            "difficulty": "hard",
            "expected_keywords": ["hybrid", "physical", "deep learning"],
            "expected_answer_keywords": ["hybrid", "physical", "model"],
        },
        {
            "id": "hydro_hard_03",
            "query": "How does attention mechanism help in multi-step ahead streamflow prediction?",
            "domain": "hydrology",
            "difficulty": "hard",
            "expected_keywords": ["attention", "streamflow", "prediction"],
            "expected_answer_keywords": ["attention", "multi-step"],
        },
        {
            "id": "hydro_hard_04",
            "query": "What evaluation metrics are most appropriate for comparing streamflow models across different climate regions?",
            "domain": "hydrology",
            "difficulty": "hard",
            "expected_keywords": ["evaluation", "streamflow", "climate"],
            "expected_answer_keywords": ["metric", "evaluation"],
        },
        {
            "id": "hydro_hard_05",
            "query": "How can graph neural networks capture spatial dependencies in watershed modeling?",
            "domain": "hydrology",
            "difficulty": "hard",
            "expected_keywords": ["graph", "neural", "watershed"],
            "expected_answer_keywords": ["graph", "spatial"],
        },
        # ---- HEALTHCARE: Easy ----
        {
            "id": "health_easy_01",
            "query": "How is NLP used for clinical decision support?",
            "domain": "healthcare",
            "difficulty": "easy",
            "expected_keywords": ["NLP", "clinical", "decision support"],
            "expected_answer_keywords": ["NLP", "clinical"],
        },
        {
            "id": "health_easy_02",
            "query": "What machine learning methods are used for electronic health record analysis?",
            "domain": "healthcare",
            "difficulty": "easy",
            "expected_keywords": ["electronic health record", "machine learning"],
            "expected_answer_keywords": ["EHR", "health record"],
        },
        {
            "id": "health_easy_03",
            "query": "How can text mining extract information from medical documents?",
            "domain": "healthcare",
            "difficulty": "easy",
            "expected_keywords": ["text mining", "medical"],
            "expected_answer_keywords": ["text", "medical", "extraction"],
        },
        {
            "id": "health_easy_04",
            "query": "What are biomedical question answering systems?",
            "domain": "healthcare",
            "difficulty": "easy",
            "expected_keywords": ["biomedical", "question answering"],
            "expected_answer_keywords": ["biomedical", "question", "answer"],
        },
        {
            "id": "health_easy_05",
            "query": "How is machine learning applied to medical document retrieval?",
            "domain": "healthcare",
            "difficulty": "easy",
            "expected_keywords": ["medical", "document", "retrieval"],
            "expected_answer_keywords": ["medical", "retrieval"],
        },
        # ---- HEALTHCARE: Medium ----
        {
            "id": "health_med_01",
            "query": "What challenges exist in applying NLP to clinical notes?",
            "domain": "healthcare",
            "difficulty": "medium",
            "expected_keywords": ["NLP", "clinical notes"],
            "expected_answer_keywords": ["clinical", "challenge"],
        },
        {
            "id": "health_med_02",
            "query": "How can deep learning improve patient outcome prediction from EHR data?",
            "domain": "healthcare",
            "difficulty": "medium",
            "expected_keywords": ["deep learning", "patient", "EHR"],
            "expected_answer_keywords": ["patient", "prediction"],
        },
        {
            "id": "health_med_03",
            "query": "What role does named entity recognition play in biomedical text processing?",
            "domain": "healthcare",
            "difficulty": "medium",
            "expected_keywords": ["entity recognition", "biomedical"],
            "expected_answer_keywords": ["entity", "recognition", "biomedical"],
        },
        {
            "id": "health_med_04",
            "query": "How are transformers being applied to healthcare NLP tasks?",
            "domain": "healthcare",
            "difficulty": "medium",
            "expected_keywords": ["transformer", "healthcare"],
            "expected_answer_keywords": ["transformer", "healthcare"],
        },
        {
            "id": "health_med_05",
            "query": "What methods improve the accuracy of clinical text classification?",
            "domain": "healthcare",
            "difficulty": "medium",
            "expected_keywords": ["clinical", "text classification"],
            "expected_answer_keywords": ["classification", "clinical"],
        },
        # ---- LLM/RAG: Easy ----
        {
            "id": "llm_easy_01",
            "query": "What is retrieval augmented generation?",
            "domain": "llm",
            "difficulty": "easy",
            "expected_keywords": ["retrieval", "augmented", "generation"],
            "expected_answer_keywords": ["retrieval", "generation", "LLM"],
        },
        {
            "id": "llm_easy_02",
            "query": "How are vector databases used in RAG systems?",
            "domain": "llm",
            "difficulty": "easy",
            "expected_keywords": ["vector", "database", "RAG"],
            "expected_answer_keywords": ["vector", "retrieval"],
        },
        {
            "id": "llm_easy_03",
            "query": "What embedding models are used for document retrieval?",
            "domain": "llm",
            "difficulty": "easy",
            "expected_keywords": ["embedding", "retrieval"],
            "expected_answer_keywords": ["embedding", "model"],
        },
        {
            "id": "llm_easy_04",
            "query": "How do large language models handle hallucination in generated text?",
            "domain": "llm",
            "difficulty": "easy",
            "expected_keywords": ["hallucination", "language model"],
            "expected_answer_keywords": ["hallucination"],
        },
        {
            "id": "llm_easy_05",
            "query": "What evaluation metrics are used for RAG systems?",
            "domain": "llm",
            "difficulty": "easy",
            "expected_keywords": ["evaluation", "RAG"],
            "expected_answer_keywords": ["evaluation", "metric"],
        },
        # ---- LLM/RAG: Medium ----
        {
            "id": "llm_med_01",
            "query": "How does retrieval quality affect the accuracy of RAG generated answers?",
            "domain": "llm",
            "difficulty": "medium",
            "expected_keywords": ["retrieval", "quality", "RAG"],
            "expected_answer_keywords": ["retrieval", "quality", "answer"],
        },
        {
            "id": "llm_med_02",
            "query": "What chunking strategies work best for scientific document retrieval?",
            "domain": "llm",
            "difficulty": "medium",
            "expected_keywords": ["chunking", "retrieval"],
            "expected_answer_keywords": ["chunk", "strategy"],
        },
        {
            "id": "llm_med_03",
            "query": "How can reranking improve retrieval precision in RAG pipelines?",
            "domain": "llm",
            "difficulty": "medium",
            "expected_keywords": ["reranking", "retrieval"],
            "expected_answer_keywords": ["rerank", "precision"],
        },
        {
            "id": "llm_med_04",
            "query": "What is the difference between dense and sparse retrieval methods?",
            "domain": "llm",
            "difficulty": "medium",
            "expected_keywords": ["dense", "sparse", "retrieval"],
            "expected_answer_keywords": ["dense", "sparse"],
        },
        {
            "id": "llm_med_05",
            "query": "How do embedding model choice and chunk size interact in RAG performance?",
            "domain": "llm",
            "difficulty": "medium",
            "expected_keywords": ["embedding", "chunk", "RAG"],
            "expected_answer_keywords": ["embedding", "chunk"],
        },
        # ---- CROSS-DOMAIN: Hard ----
        {
            "id": "cross_hard_01",
            "query": "How can RAG systems be applied to environmental monitoring data analysis?",
            "domain": "cross",
            "difficulty": "hard",
            "expected_keywords": ["RAG", "environmental"],
            "expected_answer_keywords": ["retrieval", "environmental"],
        },
        {
            "id": "cross_hard_02",
            "query": "What are the benefits of combining time series models with NLP for sensor data interpretation?",
            "domain": "cross",
            "difficulty": "hard",
            "expected_keywords": ["time series", "NLP", "sensor"],
            "expected_answer_keywords": ["time series", "natural language"],
        },
        {
            "id": "cross_hard_03",
            "query": "How can foundation models trained on scientific data generalize across environmental domains?",
            "domain": "cross",
            "difficulty": "hard",
            "expected_keywords": ["foundation model", "scientific", "generalize"],
            "expected_answer_keywords": ["foundation", "transfer", "domain"],
        },
        {
            "id": "cross_hard_04",
            "query": "What approaches combine document retrieval with numerical data analysis for scientific research?",
            "domain": "cross",
            "difficulty": "hard",
            "expected_keywords": ["retrieval", "numerical", "scientific"],
            "expected_answer_keywords": ["retrieval", "data", "analysis"],
        },
        {
            "id": "cross_hard_05",
            "query": "How can automated literature review systems help identify research gaps in hydrology?",
            "domain": "cross",
            "difficulty": "hard",
            "expected_keywords": ["literature review", "hydrology"],
            "expected_answer_keywords": ["literature", "review", "research"],
        },
    ]
    return questions

def main():
    questions = create_test_questions()
    
    os.makedirs("data/evaluation", exist_ok=True)
    with open("data/evaluation/test_questions.json", "w") as f:
        json.dump(questions, f, indent=2)
    
    # Print stats
    domains = {}
    difficulties = {}
    for q in questions:
        domains[q["domain"]] = domains.get(q["domain"], 0) + 1
        difficulties[q["difficulty"]] = difficulties.get(q["difficulty"], 0) + 1
    
    print("=" * 50)
    print(f"Test set created: {len(questions)} questions")
    print(f"Domains: {domains}")
    print(f"Difficulties: {difficulties}")
    print(f"Saved to: data/evaluation/test_questions.json")
    print("=" * 50)

if __name__ == "__main__":
    main()
