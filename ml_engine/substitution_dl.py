import os
import tensorflow as tf
import numpy as np
from ml_engine.custom_layers import FlavorInteractionLayer

class SubstitutionRankerModel:
    def __init__(self, engine_instance=None):
        self.engine = engine_instance
        self.model_path = os.path.join(os.path.dirname(__file__), "weights", "substitution_ranker.keras")
        self.model = None
        self.is_ready = False
        self.load_model_safe()

    def load_model_safe(self):
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(
                    self.model_path,
                    custom_objects={"FlavorInteractionLayer": FlavorInteractionLayer}
                )
                self.is_ready = True
                print("Deep Learning Model Ranker berhasil dimuat!")
            except Exception:
                self.is_ready = False
        else:
            self.is_ready = False

    def predict_rank_score_context(self, candidate_item: str, all_recipe_ingredients: list) -> float:
        """
        Inference Baru: Mengadu bumbu surplus melawan Rata-Rata Konteks Rasa Resep.
        Sukses membaca flavor_vector 7-D langsung dari objek memory asli Aidan (ingredient_kb).
        """
        if not self.is_ready or self.model is None or self.engine is None:
            return 0.72
            
        c_data = self.engine.ingredient_kb.get(candidate_item)
        if not c_data:
            return 0.0
            
        # 1. Tarik vektor rasa 7D milik bumbu surplus (Kandidat) langsung dari properti data
        raw_c = c_data.get("flavor_vector")
        vec_c = np.asarray(raw_c, dtype=np.float32).reshape(1, -1) if raw_c is not None else np.zeros((1, 7), dtype=np.float32)
        
        # 2. HITUNG CONTEXT: Ambil semua herba resep, cari rata-rata vektor rasanya
        context_vectors = []
        for ing_name in all_recipe_ingredients:
            if ing_name == candidate_item: 
                continue # Jangan masukkan bahan yang sedang diuji ke dalam konteks kuali
                
            ing_data = self.engine.ingredient_kb.get(ing_name)
            if ing_data:
                raw_i = ing_data.get("flavor_vector")
                # 🛡️ FIX DI SINI: Gunakan 'is not None' bukan 'if raw_i'
                if raw_i is not None:
                    context_vectors.append(raw_i)
                    
        # Jika resep kosong, kasih default zeros, jika ada isinya kita hitung rata-ratanya (Average Pooling)
        if context_vectors:
            mean_context = np.mean(context_vectors, axis=0).astype(np.float32).reshape(1, -1)
        else:
            mean_context = np.zeros((1, 7), dtype=np.float32)
            
        # 3. Lempar ke model .keras lu: [Kandidat, Rata-rata Kuali Resep]
        try:
            preds = self.model([vec_c, mean_context], training=False)
            return float(preds[0][0])
        except Exception:
            return 0.72  # Jaminan keselamatan runtime jika terjadi kendala aljabar tensor
    
    def predict_rank_score(self, missing_item: str, candidate_item: str) -> float:
        """
        Jembatan alias (Backward Compatibility) agar router lama Aidan tidak break.
        Secara otomatis menggunakan bahan pengganti dan mengasumsikan resep dasar 
        diambil dari knowledge base sebagai representasi konteks awal.
        """
        # Kita panggil fungsi konteks baru dengan melemparkan candidate_item, 
        # dan sebagai fallback context, masukkan missing_item ke dalam list herba resep
        return self.predict_rank_score_context(
            candidate_item=candidate_item, 
            all_recipe_ingredients=[missing_item]
        )