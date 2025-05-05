import json
import jieba
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

# --- Constants and Global Variables ---
VOCAB_SIZE = 2100  # sql编程
SEQUENCE_LENGTH = 300  # sql编程

ALG_OP_TEXT_EN = ["=", "<>", ">=", "<=", ">", "<", "like"]
ALG_OP_TEXT_ZH = ["等于", "不等于", "大于等于", "小于等于", "大于", "小于", "类似于"]
CATE_OP_EN = ["="]
CATE_OP_ZH = ["为"]

# This will store table/attribute mapping during standardization
sample = {}

# --- Helper Functions ---


def find_all(sub, s):
    """Finds all occurrences of a substring."""
    index_list = []
    index = s.find(sub)
    while index != -1:
        index_list.append(index)
        index = s.find(sub, index + 1)
    return index_list if index_list else [-1]


def textReplace(text, s):
    """Replaces table/attribute/value names with standardized placeholders."""
    newtext = text
    global sample  # Ensure we modify the global sample dictionary

    if not sample or "tablezh" not in sample or "tableen" not in sample or "attribute" not in sample:
        print("Warning: 'sample' dictionary not properly initialized in textReplace.")
        return newtext  # Avoid errors if sample isn't ready

    if s > 0:  # requirement preprocessing sql
        posv = find_all(sample["tablezh"], newtext)
        if posv != [-1]:
            pos1 = posv[-1]
            newtext = newtext[:pos1] + newtext[pos1:].replace(sample["tablezh"], "table")

        latt = len(sample["attribute"])
        for i in range(latt):
            pos1 = newtext.find(sample["attribute"][i]["zh"])
            if pos1 >= 0:
                # Use a placeholder that includes the original index 'i'
                newtext = newtext[:pos1] + newtext[pos1:].replace(sample["attribute"][i]["zh"], f"att{i}")

        # calculate OP processing
        algOpNum = len(ALG_OP_TEXT_ZH)
        for i in range(algOpNum):
            posv = find_all(ALG_OP_TEXT_ZH[i], newtext)
            if posv == [-1]:
                continue
            else:
                offset = 0
                li = len(ALG_OP_TEXT_ZH[i])
                for k in range(len(posv)):  # Use k instead of j to avoid conflict
                    pos1 = posv[k] + offset
                    # Check if the operator is surrounded by potential attribute and value
                    # This logic might need refinement based on actual sentence structures
                    # Simple check: look for 'attX' before and digit/quote after
                    prev_att_match = tf.strings.regex_full_match(
                        newtext[max(0, pos1 - 5) : pos1], r".*att\d+\s*$"
                    )  # Check before
                    next_val_match = tf.strings.regex_full_match(
                        newtext[pos1 + li : min(len(newtext), pos1 + li + 10)], r"^\s*(\d+|'.*').*"
                    )  # Check after

                    if prev_att_match and next_val_match:
                        # Extract value and associated attribute index more robustly
                        try:
                            # Find the 'attX' just before the operator
                            search_start = max(0, pos1 - 10)  # Search window
                            att_search_text = newtext[search_start:pos1]
                            att_match = tf.strings.regex_search(att_search_text, r"att(\d+)\s*$")
                            if att_match:
                                index_str = att_match.numpy().decode("utf-8").split("att")[-1].split()[0]
                                index = int(index_str)

                                # Extract the value after the operator
                                val_search_text = newtext[pos1 + li :]
                                val_match = tf.strings.regex_search(val_search_text, r"^\s*(\d+|'.+?')")
                                if val_match:
                                    digitValue = val_match.numpy().decode("utf-8").strip()
                                    placeholder = f"value{index}"
                                    # Replace the original value with the placeholder
                                    newtext = newtext[: pos1 + li] + val_search_text.replace(digitValue, placeholder, 1)
                                    offset += len(placeholder) - len(digitValue)
                                    # Store the original value in the sample dictionary
                                    if index < len(sample["attribute"]):
                                        sample["attribute"][index]["value"] = digitValue
                                    else:
                                        print(
                                            f"Warning: Attribute index {index} out of bounds during value replacement."
                                        )

                        except Exception as e:
                            print(f"Error during value processing for '{ALG_OP_TEXT_ZH[i]}': {e}")
                            # Continue processing other parts even if one fails

        # category OP processing (similar robust logic needed)
        cateOpNum = len(CATE_OP_ZH)
        for i in range(cateOpNum):
            posv = find_all(CATE_OP_ZH[i], newtext)
            if posv == [-1]:
                continue
            else:
                offset = 0
                li = len(CATE_OP_ZH[i])
                for k in range(len(posv)):
                    pos1 = posv[k] + offset
                    # Similar logic to find preceding 'attX' and following quoted value
                    try:
                        search_start = max(0, pos1 - 10)
                        att_search_text = newtext[search_start:pos1]
                        att_match = tf.strings.regex_search(att_search_text, r"att(\d+)\s*$")
                        if att_match:
                            index_str = att_match.numpy().decode("utf-8").split("att")[-1].split()[0]
                            index = int(index_str)

                            val_search_text = newtext[pos1 + li :]
                            val_match = tf.strings.regex_search(
                                val_search_text, r"^\s*'(.+?)'"
                            )  # Match content within quotes
                            if val_match:
                                cateValue = val_match.numpy().decode("utf-8").strip()[1:-1]  # Get value without quotes
                                placeholder = f"value{index}"
                                original_quoted_value = f"'{cateValue}'"
                                # Replace the original quoted value
                                newtext = newtext[: pos1 + li] + val_search_text.replace(
                                    original_quoted_value, placeholder, 1
                                )
                                offset += len(placeholder) - len(original_quoted_value)
                                if index < len(sample["attribute"]):
                                    sample["attribute"][index]["value"] = f"'{cateValue}'"  # Store with quotes
                                else:
                                    print(
                                        f"Warning: Attribute index {index} out of bounds during category value replacement."
                                    )

                    except Exception as e:
                        print(f"Error during category value processing for '{CATE_OP_ZH[i]}': {e}")

    else:  # sql statement preprocessing
        pos1 = newtext.find(sample["tableen"])
        if pos1 >= 0:
            newtext = newtext.replace(sample["tableen"], "table")

        latt = len(sample["attribute"])
        for i in range(latt):
            # Replace attribute english name with att{i}
            pos1 = newtext.find(sample["attribute"][i]["en"])
            if pos1 >= 0:
                newtext = newtext.replace(sample["attribute"][i]["en"], f"att{i}")

        # Replace stored values with value{i} placeholders
        # Iterate through attributes that have a stored 'value'
        for i in range(latt):
            if "value" in sample["attribute"][i]:
                original_value = sample["attribute"][i]["value"]
                placeholder = f"value{i}"
                # Need careful replacement to avoid replacing parts of other words
                # Use regex with word boundaries or spaces if possible
                # Simple replacement for now, might need refinement
                newtext = newtext.replace(original_value, placeholder)

    return newtext


def standarizeRequirement(text):
    """Standardizes the natural language requirement text."""
    newtext = "在表格"
    global sample  # Ensure we initialize/modify the global sample dictionary
    sample = {}  # Reset sample for each new requirement

    try:
        textlist = text.split(",")
        text_part = textlist[0].strip()
        pos1 = text_part.find("表格")
        pos2 = text_part.find("(")
        pos3 = text_part.find(")")
        if pos1 == -1 or pos2 == -1 or pos3 == -1 or pos2 <= pos1 or pos3 <= pos2:
            raise ValueError("Table definition format error in first part.")

        sample["tablezh"] = text_part[pos1 + 2 : pos2]
        sample["tableen"] = text_part[pos2 + 1 : pos3]
        newtexttable = "table" + text_part[pos3 + 1 :] + ","  # Keep text after table definition

        attnum = len(textlist)
        att = []
        newtextatt = ""
        if attnum < 2:
            raise ValueError("Missing attribute definitions.")

        for i in range(1, attnum - 1):
            attelem = {}
            text_part = textlist[i].strip()
            if i == 1:
                pos1_attr = text_part.find("属性有")
                if pos1_attr != -1:
                    text_part = text_part[pos1_attr + 3 :]  # Start after "属性有"

            # Find split between Chinese name and English name/type
            pos2_attr = 0
            for j in range(len(text_part)):
                if text_part[j].isascii() and text_part[j] != "(":  # Stop at first ASCII char (excluding '(')
                    break
                else:
                    pos2_attr += 1

            if pos2_attr == 0 or pos2_attr == len(text_part):
                raise ValueError(f"Cannot split attribute name/type in: '{textlist[i]}'")

            attelem["zh"] = text_part[:pos2_attr].strip()
            temptext = text_part[pos2_attr:].strip()
            pos3_attr = temptext.find("(")
            pos4_attr = temptext.find(")")
            if pos3_attr == -1 or pos4_attr == -1 or pos4_attr <= pos3_attr:
                raise ValueError(f"Cannot find type definition '(type)' in: '{textlist[i]}'")

            attelem["en"] = temptext[:pos3_attr].strip()
            attelem["type"] = temptext[pos3_attr + 1 : pos4_attr].strip()
            attelem["order"] = i - 1  # Use 0-based index internally
            att.append(attelem)
            newtextatt += f"att{i - 1},"  # Use 0-based index for placeholder

        sample["attribute"] = att

        # Process the last part (the actual query)
        text_part = textlist[attnum - 1].strip()
        if text_part.find("sql语句") > 0:
            # Pass 1 to indicate requirement preprocessing
            newlasttext = textReplace(text_part, 1)
            # Combine standardized parts
            newtext = newtext + newtexttable + newtextatt + newlasttext
        else:
            raise ValueError("Last part does not contain 'sql语句'.")

        return newtext

    except Exception as e:
        print(f"Error standardizing requirement: {e}")
        print(f"Original text: {text}")
        sample = {}  # Clear sample on error
        return "输入文本不符合规范"  # Return error message


def custom_standardization(input_string):
    """Custom standardization for TextVectorization."""
    lowercase = tf.strings.lower(input_string)
    # Remove [ and ] but keep other characters for jieba
    return tf.strings.regex_replace(lowercase, "[\[\]]", "")


# --- Vectorization Setup ---

source_vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE, output_mode="int", standardize=custom_standardization, output_sequence_length=SEQUENCE_LENGTH
)

target_vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    standardize=custom_standardization,
    output_sequence_length=SEQUENCE_LENGTH + 1,  # For teacher forcing
)

# --- Load Vocabularies ---
# Ensure these files are accessible from where the Django app runs
# Consider using absolute paths or paths relative to the project root
SOURCE_VOCAB_FILE = "source_vocab.json"
TARGET_VOCAB_FILE = "target_vocab.json"

try:
    with open(SOURCE_VOCAB_FILE, "r", encoding="utf-8") as json_file:
        source_vocab = json.load(json_file)
    source_vectorization.set_vocabulary(source_vocab)

    with open(TARGET_VOCAB_FILE, "r", encoding="utf-8") as json_file:
        target_vocab = json.load(json_file)
    target_vectorization.set_vocabulary(target_vocab)
    print("SQL Generator vocabularies loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading vocabulary file: {e}. Ensure '{SOURCE_VOCAB_FILE}' and '{TARGET_VOCAB_FILE}' exist.")
    # Handle error appropriately, maybe raise an exception or disable this service
except Exception as e:
    print(f"An error occurred loading vocabularies: {e}")

target_index_lookup = dict(zip(range(len(target_vocab)), target_vocab)) if "target_vocab" in locals() else {}
MAX_DECODED_SENTENCE_LENGTH = 60

# --- Transformer Model Definition ---


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
        else:
            padding_mask = None  # Handle case where mask is None

        attention_output = self.attention(query=inputs, value=inputs, key=inputs, attention_mask=padding_mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "dense_dim": self.dense_dim,
            }
        )
        return config


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        # Corrected: input_dim should be sequence_length for position indices
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        # Ensure positions don't exceed sequence_length used for embedding table size
        positions = tf.range(start=0, limit=length, delta=1)
        positions = tf.minimum(positions, self.sequence_length - 1)  # Clamp positions

        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        # Mask is based on input tokens (0 means padding)
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "sequence_length": self.sequence_length,
                "vocab_size": self.vocab_size,
            }
        )
        return config


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(latent_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "latent_dim": self.latent_dim,
            }
        )
        return config

    def call(self, inputs, encoder_outputs, mask=None, encoder_mask=None):  # Added encoder_mask
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            # Decoder padding mask (for self-attention)
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            # Combine causal mask and padding mask for self-attention
            combined_mask_1 = tf.minimum(padding_mask, causal_mask)
        else:
            combined_mask_1 = causal_mask

        # Cross-attention mask (using encoder's padding mask)
        if encoder_mask is not None:
            cross_attn_mask = tf.cast(encoder_mask[:, tf.newaxis, :], dtype="int32")
        else:
            cross_attn_mask = None

        attention_output_1 = self.attention_1(query=inputs, value=inputs, key=inputs, attention_mask=combined_mask_1)
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=cross_attn_mask,  # Use encoder mask here
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)


# --- Build and Load Model ---

embed_dim = 256
latent_dim = 2048
num_heads = 8


def build_transformer_model():
    """Builds the full Transformer model."""
    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="chinese")
    # Explicitly compute mask for encoder input
    encoder_input_mask = PositionalEmbedding(SEQUENCE_LENGTH, VOCAB_SIZE, embed_dim).compute_mask(encoder_inputs)
    x = PositionalEmbedding(SEQUENCE_LENGTH, VOCAB_SIZE, embed_dim)(encoder_inputs)
    # Pass the computed mask to the encoder
    encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x, mask=encoder_input_mask)

    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="sql")
    # Explicitly compute mask for decoder input
    decoder_input_mask = PositionalEmbedding(SEQUENCE_LENGTH, VOCAB_SIZE, embed_dim).compute_mask(decoder_inputs)
    x = PositionalEmbedding(SEQUENCE_LENGTH, VOCAB_SIZE, embed_dim)(decoder_inputs)
    # Pass both decoder mask (for self-attention) and encoder mask (for cross-attention)
    x = TransformerDecoder(embed_dim, latent_dim, num_heads)(
        x, encoder_outputs, mask=decoder_input_mask, encoder_mask=encoder_input_mask
    )
    x = layers.Dropout(0.5)(x)
    decoder_outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(x)

    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# Load weights - ensure the path is correct
MODEL_WEIGHTS_PATH = "TransformerSQLModelWeights2.h5"
transformer_model = None
try:
    # It's generally safer to build the model first then load weights
    transformer_model = build_transformer_model()
    transformer_model.load_weights(MODEL_WEIGHTS_PATH)
    print(f"Transformer model weights loaded successfully from {MODEL_WEIGHTS_PATH}.")
except FileNotFoundError:
    print(f"Error: Model weights file not found at {MODEL_WEIGHTS_PATH}. SQL generation will not work.")
except Exception as e:
    print(f"An error occurred loading transformer model weights: {e}")


# --- Decoding Functions ---


def decode_sequence(input_sentence):
    """Decodes an input sentence into a SQL query."""
    if transformer_model is None or not target_index_lookup:
        print("Error: Transformer model or vocab not loaded. Cannot decode.")
        return "Error: Model not ready."

    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"  # Use the start token defined in vocab/training
    for i in range(MAX_DECODED_SENTENCE_LENGTH):
        tokenized_target_sentence = target_vectorization([decoded_sentence])[:, :-1]

        try:
            # Ensure model is called correctly
            next_token_predictions = transformer_model.predict(
                [tokenized_input_sentence, tokenized_target_sentence],
                verbose=0,  # Suppress verbose output during prediction loop
            )

            # Get the token index with the highest probability at the current step 'i'
            sampled_token_index = np.argmax(next_token_predictions[0, i, :])
            sampled_token = target_index_lookup.get(sampled_token_index, "[UNK]")  # Use .get for safety

            decoded_sentence += " " + sampled_token

            # Check for end token
            if sampled_token == "[end]" or sampled_token.find("endend") >= 0:  # Use the end token from vocab/training
                break
        except Exception as e:
            print(f"Error during prediction at step {i}: {e}")
            return "Error during decoding."

    return decoded_sentence


def deStandarize(text):
    """Converts standardized SQL back to original table/attribute/value names."""
    global sample
    if not sample or "tableen" not in sample or "attribute" not in sample:
        print("Warning: 'sample' dictionary not ready for de-standardization.")
        return text  # Return text as is if sample is missing

    # Replace 'table' placeholder with original table name
    text = text.replace("from table ", f"from {sample['tableen']} ")
    text = text.replace("(table.", f"({sample['tableen']}.")  # Handle cases like (table.att0

    try:
        # Replace 'attX' placeholders
        # Iterate backwards to handle indices correctly if lengths change
        for i in range(len(sample["attribute"]) - 1, -1, -1):
            placeholder = f"att{i}"
            original_name = sample["attribute"][i]["en"]
            # Use regex for safer replacement (word boundaries)
            text = tf.strings.regex_replace(text, r"\b" + placeholder + r"\b", original_name).numpy().decode("utf-8")

        # Replace 'valueX' placeholders
        # Iterate backwards
        for i in range(len(sample["attribute"]) - 1, -1, -1):
            if "value" in sample["attribute"][i]:
                placeholder = f"value{i}"
                original_value = sample["attribute"][i]["value"]
                # Use regex for safer replacement
                text = (
                    tf.strings.regex_replace(text, r"\b" + placeholder + r"\b", original_value).numpy().decode("utf-8")
                )

    except Exception as e:
        print(f"Error during de-standardization: {e}")
        # Fallback or return partially processed text

    return text


# --- Main Service Function ---


def generate_sql(natural_language_query):
    """
    Takes a natural language query, standardizes it, generates SQL,
    and de-standardizes the result.
    """
    global sample
    try:
        # Standardize the input requirement
        standardized_text = standarizeRequirement(natural_language_query)
        if standardized_text == "输入文本不符合规范":
            return "无法解析输入文本，请检查格式。"

        # Tokenize using jieba (as done during training)
        splits = jieba.cut(standardized_text.strip(), cut_all=False)
        jieba_tokenized_text = " ".join(splits)

        # Decode sequence using the Transformer model
        decoded_standardized_sql = decode_sequence(jieba_tokenized_text)

        # Clean up start/end tokens
        decoded_standardized_sql = decoded_standardized_sql.replace("[start]", "").strip()
        end_token_pos = decoded_standardized_sql.find("[end]")  # Or "endend"
        if end_token_pos != -1:
            decoded_standardized_sql = decoded_standardized_sql[:end_token_pos].strip()

        if not decoded_standardized_sql or "Error" in decoded_standardized_sql:
            return f"模型生成SQL时出错: {decoded_standardized_sql}"

        # De-standardize the generated SQL
        final_sql = deStandarize(decoded_standardized_sql)

        # Final cleanup (e.g., replace 'select all' with 'select *')
        final_sql = final_sql.replace("select all ", "select * ")

        return final_sql

    except Exception as e:
        print(f"Error in generate_sql function: {e}")
        return "处理请求时发生内部错误。"
    finally:
        sample = {}  # Clear sample after processing
