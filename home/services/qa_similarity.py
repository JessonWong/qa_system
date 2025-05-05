import codecs
import numpy as np
import tensorflow as tf  # Import tensorflow
from tensorflow import keras  # Keep this if needed elsewhere, or use tf.keras
from keras_bert import Tokenizer, load_trained_model_from_checkpoint

# --- Constants ---
maxlen = 200
config_path = "bert_config.json"
checkpoint_path = "bert_model.ckpt"
dict_path = "vocab.txt"

# --- Tokenizer ---
token_dict = {}
try:
    with codecs.open(dict_path, "r", "utf8") as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
except FileNotFoundError:
    print(f"Error: Dictionary file not found at {dict_path}")
    # Handle error appropriately, maybe raise or exit
except Exception as e:
    print(f"Error reading dictionary file {dict_path}: {e}")


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append("[unused1]")  # space类用未经训练的[unused1]表示
            else:
                R.append("[UNK]")  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict) if token_dict else None

# --- Similarity Model ---
bert_model2 = None
simSentModel = None


def create_model():
    """Creates the sentence similarity model."""
    global bert_model2  # Ensure we load the model into the global scope
    if bert_model2 is None:
        try:
            bert_model2 = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
            # Use a more descriptive variable name than 'l'
            for layer in bert_model2.layers:
                layer.trainable = True
            print("Similarity BERT model loaded successfully.")
        except Exception as e:
            print(f"Error loading BERT model for similarity: {e}")
            return None  # Return None if model loading fails

    x1_in = tf.keras.layers.Input(shape=(None,))
    x2_in = tf.keras.layers.Input(shape=(None,))

    x = bert_model2([x1_in, x2_in])
    x = tf.keras.layers.Lambda(lambda x: x[:, 0])(x)
    simp = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    sim_model = tf.keras.Model([x1_in, x2_in], simp)
    sim_model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(1e-5),  # Use tf.keras.optimizers
        metrics=["accuracy"],
    )
    return sim_model


def load_similarity_model_weights():
    """Loads weights for the similarity model."""
    global simSentModel
    if simSentModel is None:
        simSentModel = create_model()
        if simSentModel:  # Only load weights if model creation was successful
            try:
                simSentModel.load_weights("ChineseSimSentWeights.hdf5")
                print("Sentence similarity model weights loaded successfully.")
            except Exception as e:
                print(f"Error loading sentence similarity model weights: {e}")
                simSentModel = None  # Reset if weights loading fails


# Load weights when the module is imported
load_similarity_model_weights()


def predict_similar_text(senttext1, senttext2):
    """Predicts the similarity score between two sentences."""
    if simSentModel is None or tokenizer is None:
        print("Error: Similarity model or tokenizer not initialized.")
        return 0.0  # Return a default value or raise an error

    # 利用BERT进行tokenize
    senttext1 = senttext1[:maxlen]
    senttext2 = senttext2[:maxlen]

    try:
        x1, x2 = tokenizer.encode(first=senttext1, second=senttext2)
        # Padding
        X1 = x1 + [0] * (maxlen - len(x1)) if len(x1) < maxlen else x1[:maxlen]  # Ensure maxlen truncation if needed
        X2 = x2 + [0] * (maxlen - len(x2)) if len(x2) < maxlen else x2[:maxlen]  # Ensure maxlen truncation if needed

        # 模型预测并输出预测结果
        predicted = simSentModel.predict([np.array([X1]), np.array([X2])], verbose=0)  # Suppress verbose output
        y1 = predicted[0][0]  # Get the scalar value
        # print(f"Comparing '{senttext1}' and '{senttext2}': Score = {y1}") # Optional: for debugging
        return float(y1)  # Ensure return type is float
    except Exception as e:
        print(f"Error during similarity prediction: {e}")
        return 0.0  # Return default value on error


# Optional: Pre-run prediction for model initialization (if needed, similar to views.py)
# try:
#     print("Initializing similarity model...")
#     predict_similar_text("你好", "你好吗")
#     print("Similarity model initialized.")
# except Exception as e:
#     print(f"Error during similarity model initialization: {e}")
