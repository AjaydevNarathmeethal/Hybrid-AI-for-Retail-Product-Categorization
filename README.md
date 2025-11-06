# Hybrid AI for Retail Product Categorization


A hybrid LLM + rule-based system for large-scale retail product classification, focused on the Food sector. This project combines open-source LLaMA3, DeBERTa, and rule-based methods to categorize over 2 million product entries into Food and Non-Food categories, and further into subcategories, with real-time insights via a Streamlit interface.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Data Sources](#data-sources)  
3. [Data Preprocessing](#data-preprocessing)  
4. [Level 1 Food Classification](#level-1-food-classification)  
5. [Level 1 DeBERTa Learning & Inference](#level-1-deberta-learning--inference)  
6. [Level 2 Classification using LLaMA3](#level-2-classification-using-llama3)  
7. [Interactive Streamlit Interface](#interactive-streamlit-interface)  
8. [Data Insights & Visualization](#data-insights--visualization)  
9. [Key Lessons & Future Directions](#key-lessons--future-directions)  
10. [Requirements](#requirements)  
11. [Getting Started](#getting-started)  
12. [License](#license)  

---

## Project Overview

Classifying retail product data, especially in the food sector, is challenging due to ambiguous product names, inconsistent labeling, and missing brand cues. This project:  

- Categorizes 2M+ Amazon product entries into Food/Non-Food categories and subcategories.  
- Uses a hybrid approach combining LLM reasoning (LLaMA3), fine-tuned DeBERTa, and rule-based refinements.  
- Provides a Streamlit web demo for real-time inspection of predictions and model decisions.  

The approach leverages human-in-the-loop rules, GPU-based inference with open-source LLMs, and a multi-level taxonomy for scalable, production-grade classification.

---

## Data Sources

The system integrates product data from multiple U.S. retailers:

- **Amazon:** Primary source for taxonomy and category hierarchy  
- **Target & Walmart:** Added for diversity and generalizability  

**Key extracted features:**

- `product_name`  
- `search_terms`  
- Shopper demographics (age, income, etc.)

**Taxonomy:**  

- **Level 1:** Food / Non-Food  
- **Level 2:** 8 Food subcategories (e.g., Beverages, Snacks, Breakfast, Pantry Staples)

---

## Data Preprocessing

- Raw shopper logs merged with demographic data.  
- Missing values and duplicates removed.  
- Session segmentation applied (30 min for searches, 60 min for other events).  
- Search terms linked to product views using cosine similarity and time decay.  
- Data grouped into session-level units for modeling.

---

## Level 1 Food Classification

- **Rule-based filtering** using keyword matching: `"juice", "cookie", "snack", "shampoo", "detergent"` etc.  
- Lightweight and effective first-pass categorization: `Food`, `Non-Food`, or `Unknown`.  
- Results saved as CSV for downstream processing.

---

## Level 1 DeBERTa Learning & Inference

- **Fine-tuned DeBERTa-v3** for hierarchical classification (Level 1 and Level 2).  
- Multi-task learning with separate heads for Level 1 and Level 2.  
- Optimizations: mixed precision, gradient accumulation, warmup scheduling.  
- Classifies new shopper entries with high precision (>96% for Level 1).

---

## Level 2 Classification using LLaMA3

- LLaMA3 integrated via **Ollama API** for Level 2 Food subcategories.  
- Adaptive batch inference for GPU/CPU memory optimization.  
- Key-based caching to avoid redundant inference.  
- Few-shot prompt engineering for edge-case handling.  
- Fault-tolerant with checkpointing and async saving.  
- Output: `search_term`, `product_name`, `level1_category`, `predicted_level2_category`.

---

## Interactive Streamlit Interface

- Real-time product description input and prediction.  
- Level 1 classification: DeBERTa  
- Level 2 classification: DeBERTa or LLaMA3  
- Debug mode with API tracing and prediction history logging.  

**Run:**  
```bash
streamlit run 7_Food_Categorizer_Streamlit_App.py
```

## Data Insights & Visualization

Key observations from shopper & demographic data:

- Lower-income groups (Income ID 1) dominate online shopping.
- Female shoppers spend more time browsing Food categories like Canned Food, Breakfast, Beverages.
- Seasonal spike in activity from September to November.
- Most purchased categories: Beverages > Pantry Staples > Snacks.
- Frequent itemsets (Apriori algorithm): `"Canned & Packaged Foods" + "Pantry Staples"`, `"Beverages" + "Pantry Staples"`
- Age 30–45 are the most frequent shoppers of Foods & Beverages.


## Key Lessons & Future Directions

- Hybrid models combining LLM reasoning and rule-based refinements offer the best performance.
- Prompt refinement for LLM can improve Level 1 accuracy beyond 96%.
- BERT-based models could enhance Level 2 subcategory prediction.
- Cloud deployment can scale the system for faster processing of larger datasets.


## Requirements

- Python >= 3.8
- Libraries: `pandas`, `numpy`, `torch`, `scikit-learn`, `re`, `requests`, `tqdm`, `psutil`, `streamlit`
- GPU recommended (RTX 3070+)
- Ollama API server running locally (port 11434)

