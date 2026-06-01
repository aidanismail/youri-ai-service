import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="YouriAI")
class FlavorInteractionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FlavorInteractionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        """
        inputs: List tensor [missing_flavor_vector, candidate_flavor_vector]
        Masing-masing berdimensi (Batch, 5) sesuai rancangan pkl Aidan
        """
        vec_missing, vec_candidate = inputs
        
        # 1. Selisih absolut (Jarak semantik)
        abs_diff = tf.abs(tf.subtract(vec_missing, vec_candidate))
        
        # 2. Perkalian elemen (Dot-interaction element-wise)
        elementwise_prod = tf.multiply(vec_missing, vec_candidate)
        
        # 3. Concatenate fitur gabungan
        combined_features = tf.concat([vec_missing, vec_candidate, abs_diff, elementwise_prod], axis=-1)
        return combined_features

    def get_config(self):
        return super(FlavorInteractionLayer, self).get_config()