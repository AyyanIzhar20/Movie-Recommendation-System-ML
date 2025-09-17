# 🎬 Movie Recommendation System

A machine learning–based **Movie Recommender System** built on the **MovieLens 100K dataset**.  
This project demonstrates collaborative filtering using **SVD (Singular Value Decomposition)** and **Item-based Collaborative Filtering**, along with deployment via **Streamlit**.

---

## 📊 Dataset

We use the **MovieLens 100K dataset**, which contains:
- 100,000 ratings  
- 943 users  
- 1,682 movies  

Dataset files:
- `u.data` → user–movie–rating records  
- `u.item` → movie metadata  
- `u.genre` → movie genres  
- `u.info` → dataset info  

---

## ⚙️ Steps in the Project

### 1. Data Loading & Cleaning
- Loaded `u.data`, `u.item`, and other files.
- Performed **EDA** (distribution of ratings, sparsity check).
- Built **user–item matrix**.

### 2. Data Preparation
- Normalized ratings.  
- Train/Test split (80/20).  

### 3. Model Training
- **SVD (Surprise library)**  
- **Item-based Collaborative Filtering** (cosine similarity).  

### 4. Recommendations
- Generate **Top-N unseen movie recommendations** for any user.  
- Functions for:
  - `movies_recommend_svd(user_id, n)`
  - `recommend_item_based(user_id, n)`

### 5. Evaluation
- Used **Precision@K** for performance comparison.  
- Visualized model comparison with bar charts.

### 6. Deployment
- Saved models with `joblib`:  
  - `svd_model.pkl`  
  - `item_similarity.pkl`  
  - `pred_df.pkl` (optional pre-computed predictions)  
- Built a **Streamlit web app**:
  - Choose user ID
  - Select model (SVD / Item-based CF)
  - Display Top-N recommendations

---

## 🚀 Running the Project

### 1. Clone Repo
```bash
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender


