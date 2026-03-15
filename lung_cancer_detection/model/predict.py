import tensorflow as tf

_model_cache = None


def load_model_once(model_path):
    global _model_cache
    if _model_cache is None:
        _model_cache = tf.keras.models.load_model(model_path)
    return _model_cache


def predict_image(model, processed_image):
    preds = model.predict(processed_image, verbose=0)
    return preds[0]
