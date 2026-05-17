# Youri AI Service || DBS DICODING BOOTCAMP CAPSTONE TEAM PROJECT

FastAPI AI microservice for **Youri - Your Smart Cooking Assistant**.

This service handles:

- `GET /api/v1/health`
- `POST /api/v1/match`
- `POST /api/v1/substitute`

Core ML approach:

1. TF-IDF lexical matching for recipe ingredients.
2. Hard categorical filtering using `food_group`.
3. Cosine similarity over flavor profile vectors.
4. Local substitution knowledge-base for MVP substitution suggestions.

No external Generative AI API is used for the core matching logic.

---

## Tech Stack

- Python 3.11
- FastAPI 0.111.0
- Scikit-learn 1.4.2
- NumPy
- SciPy
- TensorFlow 2.15.0
- Joblib

---

## Project Structure

```txt
YOURI-AI-SERVICE/
├─ api/
├─ core/
├─ data/
├─ models/
├─ scripts/
├─ tests/
├─ .env.example
├─ .gitignore
├─ README.md
└─ requirements.txt
```
