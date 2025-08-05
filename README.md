
 Resume2JobMatch - R2JM

An intelligent job matching system that **scores how well a resume fits a job description** — powered by **SBERT embeddings** and **unsupervised XGBoost**.

## USP

Instead of relying on basic keyword overlaps, this system uses **contextual embeddings** to deeply understand resumes and job postings. It then applies **XGBoost** to compute how compatible they are — giving you a smart relevance score.

## WORKING

1. **Preprocessing**
   Extract and clean text from resumes and job descriptions.

2. **Embedding Generation**
   Use `all-MiniLM-L6-v2` (from `sentence-transformers`) to convert each text into semantic vectors.

3. **Scoring**
   Concatenate resume and job vectors, and pass them to a **ranking model trained with XGBoost** (unsupervised learning based on similarity scoring).

4. **Prediction**
   Upload a resume → Fetch job listings → Generate match scores → Sorted results.

##  Tech Stack

* **Python 3**
* **SBERT** (`sentence-transformers`)
* **XGBoost** (unsupervised ranking)
* **pandas**, **NumPy**, **scikit-learn**
* **Google Colab / Jupyter Notebook**
* **Git + GitHub**

## 📁 Key Files

* `data/`: Contains resume-job dataset
* `embeddings.npz`: Precomputed SBERT vectors
* `xgboost_model.pkl`: Trained model
* `train_model.py`: Script to train ranking model
* `predict.py`: Inference script for match scoring
* `README.md`: This file
