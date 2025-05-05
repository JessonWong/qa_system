import numpy as np
import tensorflow as tf  # Import tensorflow
from tensorflow import keras  # Keep this if needed elsewhere, or use tf.keras
from keras_bert import load_trained_model_from_checkpoint
from .qa_similarity import tokenizer, maxlen  # Import tokenizer and maxlen from similarity module

# --- Constants ---
config_path = "bert_config.json"
checkpoint_path = "bert_model.ckpt"
# dict_path is handled in qa_similarity

# --- QA Model Components ---
bert_model = None
QAmodel = None


class SquadExample:
    """Processes a single SQuAD example."""

    def __init__(self, question, context, start_char_idx, answer_text, all_answers):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.start_token_idx = 0
        self.end_token_idx = 0
        self.context_token_to_char = []  # Store original context tokens
        self.skip = False

    def preprocess(self):
        if tokenizer is None:
            print("Error: Tokenizer not initialized in SquadExample.")
            self.skip = True
            return

        context = " ".join(str(self.context).split())
        question = " ".join(str(self.question).split())
        answer = " ".join(str(self.answer_text).split())

        end_char_idx = self.start_char_idx + len(answer)
        if end_char_idx >= len(context):
            self.skip = True
            return

        # Tokenize context and question
        try:
            tokenized_context, _ = tokenizer.encode(context)  # Use the imported tokenizer
            tokenized_question, _ = tokenizer.encode(question)
        except Exception as e:
            print(f"Error during tokenization in SquadExample: {e}")
            self.skip = True
            return

        self.context_token_to_char = tokenized_context  # Store token ids

        # Find token indices corresponding to the answer span
        context_tokens_str = tokenizer.tokenize(context)
        answer_tokens_str = tokenizer.tokenize(answer)

        start_token = -1
        end_token = -1
        current_char_pos = 0
        token_char_spans = []
        for token in context_tokens_str:
            start = context.find(token.replace("##", ""), current_char_pos)
            if start == -1:
                start = current_char_pos  # Approximate position
            end = start + len(token.replace("##", ""))
            token_char_spans.append((start, end))
            current_char_pos = end

        # Find tokens overlapping with the answer character span
        ans_token_indices = []
        for i, (start, end) in enumerate(token_char_spans):
            if start < end_char_idx and end > self.start_char_idx:
                ans_token_indices.append(i)

        if not ans_token_indices:
            try:
                first_answer_token = answer_tokens_str[0]
                start_token = context_tokens_str.index(first_answer_token)
                end_token = start_token + len(answer_tokens_str) - 1
                ans_token_indices = list(range(start_token, end_token + 1))
            except ValueError:
                print(f"Warning: Could not find answer tokens in context for: '{answer}'")
                self.skip = True

        if not self.skip and ans_token_indices:
            self.start_token_idx = ans_token_indices[0]
            self.end_token_idx = ans_token_indices[-1]
        else:
            self.start_token_idx = 0
            self.end_token_idx = 0

        # Create inputs, ensuring lengths and handling truncation/padding
        input_ids_unpadded = tokenized_context + tokenized_question[1:]
        token_type_ids_unpadded = [0] * len(tokenized_context) + [1] * len(tokenized_question[1:])

        if len(input_ids_unpadded) > maxlen:
            q_len = len(tokenized_question[1:])
            max_context_len = maxlen - q_len
            if self.end_token_idx < max_context_len:
                input_ids_truncated = input_ids_unpadded[:maxlen]
                token_type_ids_truncated = token_type_ids_unpadded[:maxlen]
            else:
                print("Warning: Truncation might affect answer span.")
                input_ids_truncated = input_ids_unpadded[:maxlen]
                token_type_ids_truncated = token_type_ids_unpadded[:maxlen]
                if self.start_token_idx >= maxlen:
                    print("Warning: Answer start token truncated.")
                    self.start_token_idx = maxlen - 1
                    self.end_token_idx = maxlen - 1
                elif self.end_token_idx >= maxlen:
                    self.end_token_idx = maxlen - 1

            self.input_ids = input_ids_truncated
            self.token_type_ids = token_type_ids_truncated
            self.attention_mask = [1] * maxlen

        else:
            padding_length = maxlen - len(input_ids_unpadded)
            self.input_ids = input_ids_unpadded + [0] * padding_length
            self.token_type_ids = token_type_ids_unpadded + [0] * padding_length
            self.attention_mask = [1] * len(input_ids_unpadded) + [0] * padding_length


def create_inputs_targets(squad_examples):
    """Creates model inputs and targets from processed SquadExamples."""
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in squad_examples:
        if not item.skip:
            dataset_dict["input_ids"].append(item.input_ids)
            dataset_dict["token_type_ids"].append(item.token_type_ids)
            dataset_dict["attention_mask"].append(item.attention_mask)
            dataset_dict["start_token_idx"].append(item.start_token_idx)
            dataset_dict["end_token_idx"].append(item.end_token_idx)

    for key in dataset_dict:
        try:
            dataset_dict[key] = np.array(dataset_dict[key])
        except ValueError as e:
            print(f"Error converting {key} to numpy array: {e}")
            return None, None

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y


def create_QA_model():
    """Creates the Question Answering model."""
    global bert_model
    if bert_model is None:
        try:
            bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
            for layer in bert_model.layers:
                layer.trainable = True
            print("QA BERT model loaded successfully.")
        except Exception as e:
            print(f"Error loading BERT model for QA: {e}")
            return None

    input_ids = tf.keras.layers.Input(shape=(maxlen,), dtype="int32", name="input_ids")
    token_type_ids = tf.keras.layers.Input(shape=(maxlen,), dtype="int32", name="token_type_ids")
    attention_mask = tf.keras.layers.Input(shape=(maxlen,), dtype="int32", name="attention_mask")

    embedding = bert_model([input_ids, token_type_ids])

    start_logits = tf.keras.layers.Dense(1, name="start_logit", use_bias=False)(embedding)
    start_logits = tf.keras.layers.Lambda(lambda x: x[..., 0], name="start_slice")(start_logits)

    end_logits = tf.keras.layers.Dense(1, name="end_logit", use_bias=False)(embedding)
    end_logits = tf.keras.layers.Lambda(lambda x: x[..., 0], name="end_slice")(end_logits)

    mask_adder = tf.keras.layers.Lambda(lambda x: (1.0 - tf.cast(x, "float32")) * -1e10, name="mask_adder")
    start_logits = tf.keras.layers.Add(name="start_add")([start_logits, mask_adder(attention_mask)])
    end_logits = tf.keras.layers.Add(name="end_add")([end_logits, mask_adder(attention_mask)])

    start_probs = tf.keras.layers.Activation(tf.keras.activations.softmax, name="start_softmax")(start_logits)
    end_probs = tf.keras.layers.Activation(tf.keras.activations.softmax, name="end_softmax")(end_logits)

    train_model = tf.keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[start_probs, end_probs],
    )

    qaloss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    qaoptimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    train_model.compile(optimizer=qaoptimizer, loss=[qaloss, qaloss])

    return train_model


def load_qa_model_weights():
    """Loads weights for the QA model."""
    global QAmodel
    if QAmodel is None:
        QAmodel = create_QA_model()
        if QAmodel:
            try:
                QAmodel.load_weights("ChineseQAweights.hdf5")
                print("QA model weights loaded successfully.")
            except Exception as e:
                print(f"Error loading QA model weights: {e}")
                QAmodel = None


load_qa_model_weights()


def predict_single_text(context, question):
    """Predicts the answer to a question given a context."""
    if QAmodel is None or tokenizer is None:
        print("Error: QA model or tokenizer not initialized.")
        return "模型未准备好。"

    pred_ans = ""
    squad_examples = []

    start_char_idx = 0
    answer_text = "a"
    all_answers = []

    squad_eg = SquadExample(question, context, start_char_idx, answer_text, all_answers)
    squad_eg.preprocess()

    if squad_eg.skip:
        print("Warning: Skipping prediction due to preprocessing issues.")
        return "无法处理输入。"

    squad_examples.append(squad_eg)

    x_test, _ = create_inputs_targets(squad_examples)
    if x_test is None:
        return "创建模型输入时出错。"

    try:
        pred_start, pred_end = QAmodel.predict(x_test, verbose=0)

        start_idx = np.argmax(pred_start[0])
        end_idx = np.argmax(pred_end[0])

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        tokenized_context_ids, _ = tokenizer.encode(context)
        context_tokens = tokenizer.convert_ids_to_tokens(tokenized_context_ids)

        if start_idx >= len(context_tokens) or end_idx >= len(context_tokens):
            print(
                f"Warning: Predicted indices ({start_idx}, {end_idx}) out of bounds for {len(context_tokens)} tokens."
            )
            return "无法在文本中定位答案。"

        answer_tokens = context_tokens[start_idx : end_idx + 1]
        pred_ans = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
        pred_ans = pred_ans.replace(" ", "").replace("[UNK]", "?")

    except Exception as e:
        print(f"Error during QA prediction or answer extraction: {e}")
        return "预测答案时出错。"

    return pred_ans if pred_ans else "未能提取答案。"
