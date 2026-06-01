import os
import sys

# Jalur agar bisa membaca modul ml_engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml_engine.information_retrieval import SmartSubstitutionEngine
from ml_engine.substitution_dl import SubstitutionRankerModel

def run_stress_test():
    print("=" * 60)
    print("🔬 STARTING HYPER CONTEXT-AWARE STRESS TEST (20 CASES)")
    print("=" * 60)
    
    engine = SmartSubstitutionEngine()
    dl_ranker = SubstitutionRankerModel(engine_instance=engine)
    
    if not dl_ranker.is_ready:
        print("[ERROR] Model DL atau Master DB gagal dimuat.")
        return

    test_cases = [
        # ==========================================
        # 🟢 10 SKENARIO EXPECTED SUCCESS (LOLOS)
        # ==========================================
        {
            "name": "Skenario 1: Tumis Ayam (Bawang Putih -> Bawang Merah)",
            "missing_item": "bawang putih", "candidate_item": "bawang merah",
            "recipe_ingredients": ["daging ayam", "bawang bombay", "merica", "garam", "daun salam"]
        },
        {
            "name": "Skenario 2: Sambal Bawang (Cabe Rawit -> Cabe Merah)",
            "missing_item": "cabe rawit", "candidate_item": "cabe merah",
            "recipe_ingredients": ["bawang merah", "bawang putih", "garam", "minyak goreng", "gula pasir"]
        },
        {
            "name": "Skenario 3: Sup Daging Warm Aromatics (Jahe -> Lengkuas)",
            "missing_item": "jahe", "candidate_item": "lengkuas",
            "recipe_ingredients": ["daging sapi", "bawang putih", "pala bubuk", "daun bawang", "seledri"]
        },
        {
            "name": "Skenario 4: Nasi Goreng Kampung (Kecap Manis -> Gula Merah)",
            "missing_item": "kecap manis", "candidate_item": "gula merah",
            "recipe_ingredients": ["nasi putih", "telor", "bawang merah", "bawang putih", "garam"]
        },
        {
            "name": "Skenario 5: Sayur Lodeh Gurih (Santan -> Kemiri)",
            "missing_item": "santan", "candidate_item": "kemiri",
            "recipe_ingredients": ["manisa", "kacang panjang", "bawang merah", "bawang putih", "daun salam"]
        },
        {
            "name": "Skenario 6: Opor Ayam Lezat (Kemiri -> Kacang Mede)",
            "missing_item": "kemiri", "candidate_item": "kacang mede",
            "recipe_ingredients": ["daging ayam", "santan", "bawang putih", "ketumbar", "serai"]
        },
        {
            "name": "Skenario 7: Sambal Terasi Umami (Terasi -> Petis Udang)",
            "missing_item": "terasi", "candidate_item": "petis udang",
            "recipe_ingredients": ["cabe rawit", "tomat", "bawang merah", "garam", "gula pasir"]
        },
        {
            "name": "Skenario 8: Tumis Kangkung Belacan (Saus Tiram -> Kecap Asin)",
            "missing_item": "saus tiram", "candidate_item": "kecap asin",
            "recipe_ingredients": ["kangkung", "bawang putih", "bawang merah", "cabe merah", "tauco"]
        },
        {
            "name": "Skenario 9: Soto Ayam Kuning (Kunyit -> Daun Kunyit)",
            "missing_item": "kunyit", "candidate_item": "daun kunyit",
            "recipe_ingredients": ["daging ayam", "bawang putih", "serai", "daun jeruk", "kemiri"]
        },
        {
            "name": "Skenario 10: Semur Daging Karamel (Gula Merah -> Madu)",
            "missing_item": "gula merah", "candidate_item": "madu",
            "recipe_ingredients": ["daging sapi", "kecap manis", "bawang merah", "pala bubuk", "merica"]
        },

        # ==========================================
        # 🔴 10 SKENARIO EXPECTED FAILURE (DITOLAK)
        # ==========================================
        {
            "name": "Skenario 11: Kolak Pisang Tradisional (Santan -> Minyak Wijen)",
            "missing_item": "santan", "candidate_item": "minyak wijen",
            "recipe_ingredients": ["pisang kepok", "gula merah", "daun pandan", "kolang kaling", "singkong"]
        },
        {
            "name": "Skenario 12: Es Dawet Ayu Manis (Gula Merah -> Saus Tiram)",
            "missing_item": "gula merah", "candidate_item": "saus tiram",
            "recipe_ingredients": ["cendol", "santan", "nangka", "es batu", "daun pandan"]
        },
        {
            "name": "Skenario 13: Sambal Rawit Ekstrem (Cabe Rawit -> Gula Pasir)",
            "missing_item": "cabe rawit", "candidate_item": "gula pasir",
            "recipe_ingredients": ["bawang merah", "bawang putih", "garam", "minyak goreng"]
        },
        {
            "name": "Skenario 14: Sup Kaldu Ayam Bening (Bawang Putih -> Buah Apel)",
            "missing_item": "bawang putih", "candidate_item": "buah apel",
            "recipe_ingredients": ["daging ayam", "wortel", "kentang", "merica", "garam", "seledri"]
        },
        {
            "name": "Skenario 15: Tumis Tempe Pedas Gurih (Bawang Merah -> Susu UHT)",
            "missing_item": "bawang merah", "candidate_item": "susu uht",
            "recipe_ingredients": ["tempe", "cabe hijau", "kecap manis", "bawang putih", "lengkuas"]
        },
        {
            "name": "Skenario 16: Rendang Daging Minang (Santan -> Cuka Makan)",
            "missing_item": "santan", "candidate_item": "cuka makan",
            "recipe_ingredients": ["daging sapi", "bawang putih", "cabe merah", "lengkuas", "serai"]
        },
        {
            "name": "Skenario 17: Sayur Asem Segar (Asam Jawa -> Tepung Terigu)",
            "missing_item": "asam jawa", "candidate_item": "tepung terigu",
            "recipe_ingredients": ["jagung manis", "melinjo", "kacang tanah", "labu siam", "daun salam"]
        },
        {
            "name": "Skenario 18: Ayam Goreng Ketumbar (Ketumbar -> Buah Pisang)",
            "missing_item": "ketumbar", "candidate_item": "buah pisang",
            "recipe_ingredients": ["daging ayam", "bawang putih", "kunyit", "lengkuas", "garam"]
        },
        {
            "name": "Skenario 19: Bubur Sumsum Lembut (Garam -> Cabe Rawit)",
            "missing_item": "garam", "candidate_item": "cabe rawit",
            "recipe_ingredients": ["tepung beras", "santan", "daun pandan", "air putih"]
        },
        {
            "name": "Skenario 20: Pindang Patin Sumatra (Asam Jawa -> Mayones)",
            "missing_item": "asam jawa", "candidate_item": "mayones",
            "recipe_ingredients": ["ikan patin", "tomat hijau", "serai", "daun kunyit", "cabe merah"]
        }
    ]
    
    # 3. Jalankan Inferensi Evaluasi dengan Threshold Adaptif 0.04
    for idx, case in enumerate(test_cases, 1):
        score = dl_ranker.predict_rank_score_context(
            candidate_item=case["candidate_item"],
            all_recipe_ingredients=case["recipe_ingredients"]
        )
        
        print(f"\n[{idx}] {case['name']}")
        print(f"  - Bahan Hilang : {case['missing_item']} -> Pengganti: {case['candidate_item']}")
        print(f"  --> 🧠 DL Confidence Score: {score:.4f}")
        
        if score >= 0.0400:
            print("  --> 🟢 Kesimpulan: Model setuju, kombinasi rasa harmoni dengan isi kuali!")
        else:
            print("  --> 🔴 Kesimpulan: Model menolak/menurunkan skor karena merusak nuansa kuali!")

    print("\n" + "=" * 60)
    print("🏁 HYPER STRESS TEST SELESAI SUKSES")
    print("=" * 60)

if __name__ == "__main__":
    run_stress_test()