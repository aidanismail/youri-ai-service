# Youri AI Service

FastAPI microservice untuk fitur AI pada **Youri - Smart Cooking Assistant**. Service ini membantu aplikasi memberi rekomendasi resep berdasarkan bahan yang dimiliki user dan memberi saran substitusi bahan yang hilang.

## Tujuan Project

- Mencocokkan bahan user dengan resep yang paling relevan.
- Menghindari rekomendasi yang terlalu longgar dari payload anomali.
- Memberi substitusi bahan yang lebih realistis berdasarkan fungsi kuliner dan kemiripan rasa.
- Menjadi layer AI backend yang bisa diintegrasikan dengan PWA Youri.
- Menyediakan fondasi untuk pengembangan Deep Learning sesuai checklist capstone.

## Status Implementasi

| Area | Status | Catatan |
|---|---:|---|
| FastAPI service | Done | Endpoint internal sudah tersedia |
| Recipe matching | Done | TF-IDF + ingredient normalization + scoring guard |
| AI substitution baseline | Done | Functional group + cosine similarity |
| API key protection | Done | Header `x-internal-api-key` |
| DL model | Done | Dilanjutkan di `train_dl.py`, `custom_layers.py`, `substitution_dl.py` |
| TensorBoard/evaluation/export DL | Done | Bagian lanjutan untuk checklist AI |
| AI Substitution Engine | Done | Integrasi Hybrid: Functional Group Matrix + Cosine Similarity |
| Deep Learning Ranker | Done | Keras Functional API dengan Custom Layer FlavorInteractionLayer |
| Context-Aware Inference | Done | Model sukses mengadu bumbu surplus vs Rata-Rata Vektor Rasa Kuali |


## Tech Stack

- Python 3.11
- FastAPI
- Scikit-learn
- TensorFlow 2.15
- NumPy
- SciPy
- Joblib
- Google Generative AI SDK

## Project Structure

```txt
youri-ai-service/
|-- main.py
|-- requirements.txt
|-- README.md
|-- api/
|   |-- dependencies.py
|   |-- schemas.py
|   `-- routers/
|       `-- cooking.py
|-- data/
|   `-- dataset_opsi_1_final_with_images.json  
|-- ml_engine/
|   |-- information_retrieval.py               
|   |-- ingredient_normalization.py            
|   |-- custom_layers.py                       
|   |-- substitution_dl.py                     
|   `-- weights/
|       |-- tfidf_vectorizer.pkl
|       |-- recipe_matrix.pkl
|       |-- recipe_metadata.pkl
|       |-- ingredient_master.pkl              
|       |-- substitution_ranker.keras          
|       `-- substitution_ranker_metrics.json   
|-- scripts/
|   |-- train_tfidf.py                         
|   `-- train_dl.py                            
`-- logs/
    `-- tensorboard/                           
```

## Setup

```powershell
cd "\youri-ai-service"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set API key internal:

```powershell
$env:INTERNAL_API_KEY="your-development-key"
```

Jika env var tidak diset, service memakai default development key:

```txt
youri-super-secret-key-2026
```

## Generate Artifact TF-IDF

Jalankan ulang command ini jika dataset atau normalization berubah:

```powershell
python scripts\train_tfidf.py
```

Output artifact:

```txt
ml_engine/weights/tfidf_vectorizer.pkl
ml_engine/weights/recipe_matrix.pkl
ml_engine/weights/recipe_metadata.pkl
ml_engine/weights/ingredient_master.pkl
```

## Generate Artifact Deep Learning

Jalankan perintah ini untuk melatih model jaringan saraf tiruan (Keras) dalam mengenali interaksi kombinasi rasa bumbu di dalam kuali:

```powershell
python scripts/train_dl.py
```

Output artifact:

```txt
ml_engine/weights/substitution_ranker.keras
ml_engine/weights/substitution_ranker_metrics.json
```

## Run Service

```powershell
uvicorn main:app --reload
```

Base URL lokal:

```txt
http://127.0.0.1:8000
```

Swagger UI:

```txt
http://127.0.0.1:8000/docs
```

Health check:

```txt
GET /health
```

## Endpoints

Semua endpoint AI membutuhkan header:

```txt
x-internal-api-key: your-development-key
```

### Match Recipes

```txt
POST /v1/ai/cooking/match
```

Contoh payload:

```json
{
  "ingredients": [
    {
      "name": "ayam",
      "is_valid": true
    },
    {
      "name": "bawang putih",
      "is_valid": true
    },
    {
      "name": "kecap manis",
      "is_valid": true
    }
  ],
  "tools": []
}
```

### AI Substitution

```txt
POST /v1/ai/cooking/ai-substitution
```

Contoh payload:

```json
{
  "recipe_id": "pepes-tahu-jamur-tiram",
  "missing_ingredients": [
    {
      "id": "ING-0001",
      "name": "bawang merah",
      "is_valid": true
    }
  ],
  "surplus_ingredients": [
    {
      "id": "ING-0002",
      "name": "bawang putih",
      "is_valid": true
    }
  ]
}
```

## Flow AI Saat Ini

### Recipe Matching

```txt
User Payload Ingredients (Kulkas)
  │
  ├──► Filter komponen 'is_valid' & eliminasi bumbu umum (Garampenyedap, Air, Minyak) via COMMON_INGREDIENTS
  ├──► Normalisasi string teks menjadi token kanonikal murni
  ├──► Transformasi matriks via TF-IDF Vectorizer
  ├──► Penyaringan Guard: Validasi ketersediaan bahan utama resep (Anchor Ingredients)
  ├──► Kalkulasi rasio cakupan resep (Recipe Coverage) & presisi bahan pengguna (User Precision)
  └──► Penggabungan metadata langkah memasak ('steps') via dinamika algoritma Local Slugify
```

### AI Substitution

```txt
Bahan Hilang (Missing) + Bahan Sisa (Surplus)
  │
  ├──► Validasi gerbang Layer 1: Kompatibilitas fungsional rumpun makanan (Substitution Compatibility Matrix)
  ├──► Validasi gerbang Layer 2: Perhitungan Cosine Similarity antara matriks profil rasa 7-D bahan masakan
  ├──► Validasi gerbang Layer 3 (Deep Learning): Evaluasi bumbu surplus melawan Rata-Rata Konteks Rasa Resep 
  │    menggunakan model jaringan saraf tiruan (Average Pooling Context Vector)
  ├──► Kombinasi Bobot Final Skor: (50% DL Score + 30% Flavor Cosine + 20% Functional Filter)
  └──► Output: Pemetaan bahan substitusi terbaik dilengkapi dialog respon emosional karakter asisten masis
```

