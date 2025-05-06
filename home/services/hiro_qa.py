import json
import torch
import numpy as np
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import os
from .hiro_es_search import ESSearcher

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def preprocess_data(data):
    question_data = []

    for item in data:
        question_text = item["question"]["question_context"]
        question_id = item["question"]["question_id"]
        answers = item["answers"]

        question_data.append({"question_id": question_id, "question_text": question_text, "answers": answers})

    return question_data


class BertEncoder:
    def __init__(self, model_name="bert-base-chinese"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, texts, batch_size=8):
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # 对文本进行编码
            encoded_input = self.tokenizer(
                batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512
            )

            # 获取BERT输出
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # 使用[CLS]标记的输出作为整个句子的表示
            sentence_embeddings = model_output.last_hidden_state[:, 0, :].numpy()
            all_embeddings.append(sentence_embeddings)

        return np.vstack(all_embeddings)


class LLMEnhancer:
    """使用本地Llama模型增强答案质量的类"""

    def __init__(self, model_path="meta-llama/Llama-2-7b-chat-hf", device="cuda"):
        """
        初始化Llama增强器

        参数:
            model_path: 本地Llama模型路径或Hugging Face模型名称
            device: 使用的设备，'cuda'或'cpu'
        """
        # self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        # print(f"正在加载Llama模型到{self.device}设备...")

        # 使用半精度加载模型以节省内存
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            # low_cpu_mem_usage=True,
            # device_map=self.device
            # load_in_4bit = True
        )

        # print("Llama模型加载完成")

    def enhance_answer(self, query, original_answer, max_new_tokens=1000):
        # 构建系统提示和用户提示
        system_prompt = "你是一个专业的数据库知识问答助手，擅长解释数据库相关概念。"

        user_prompt = f"""
用户问题: {query}

现有答案: {original_answer}

请对上述答案进行补充和完善，使其更加全面、准确和易于理解。
回答要详实专业，可以适当添加例子或相关知识，但不要过度冗长。
        """
        # print(user_prompt)
        # 构建完整的提示模板（根据Llama-2-chat的格式）
        # prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
        template = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        # 编码输入
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        inputs = self.tokenizer.apply_chat_template(
            template, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )

        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                inputs, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9, do_sample=True
            )

        # 解码并提取生成的文本
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取模型生成的回答部分（去除提示部分）
        enhanced_answer = full_response.split("assistant")[-1].strip()
        # print("增强结果：",enhanced_answer)
        return enhanced_answer


class BertQuestionAnswering:
    def __init__(
        self,
        es=False,
        data_path=None,
        use_llm_enhancement=False,
        llm_model_path="meta-llama/Llama-2-7b-chat-hf",
        device="cuda",
    ):
        if es is False:
            self.data = load_dataset(data_path)
            pass
        else:
            self.es = ESSearcher(es_hosts=["localhost:9200"], index_name="courseqa")
            if not self.es.is_connected():
                raise ConnectionError("无法连接到Elasticsearch服务器。")
            self.data = self.load_es_data()
        self.question_data = preprocess_data(self.data)
        self.questions = [item["question_text"] for item in self.question_data]
        self.encoder = BertEncoder()
        self.question_embeddings = self.encoder.encode(self.questions)

        # 新增: 大模型增强功能开关和初始化
        self.use_llm_enhancement = use_llm_enhancement
        if use_llm_enhancement:
            self.llm_enhancer = LLMEnhancer(model_path=llm_model_path)

    def get_answers(self, query, top_k_questions=3, enhance_top_answer=True):
        query_embedding = self.encoder.encode([query])
        similarities = cosine_similarity(query_embedding, self.question_embeddings)[0]
        top_indices = similarities.argsort()[-top_k_questions:][::-1]

        all_candidate_answers = []

        for idx in top_indices:
            question_similarity = similarities[idx]
            question_info = self.question_data[idx]

            for answer in question_info["answers"]:
                answer_type_weight = self._get_answer_type_weight(answer["answer_type"])
                confidence = question_similarity * answer_type_weight

                all_candidate_answers.append(
                    {
                        "question_id": question_info["question_id"],
                        "question_text": question_info["question_text"],
                        "question_similarity": question_similarity,
                        "answer_id": answer["answer_id"],
                        "answer_type": answer["answer_type"],
                        "answer_text": answer["answer_context"],
                        "answer_type_weight": answer_type_weight,
                        "confidence": confidence,
                        "enhanced": False,  # 标记是否已增强
                        "enhanced_answer": None,  # 存储增强后的答案
                    }
                )

        all_candidate_answers.sort(key=lambda x: x["confidence"], reverse=True)

        # 如果启用大模型增强且需要增强最佳答案
        if self.use_llm_enhancement and enhance_top_answer and all_candidate_answers:
            top_answer = all_candidate_answers[0]
            enhanced_answer = self.llm_enhancer.enhance_answer(query, top_answer["answer_text"])
            top_answer["enhanced"] = True
            top_answer["enhanced_answer"] = enhanced_answer

        return all_candidate_answers

    def _get_answer_type_weight(self, answer_type):
        """根据答案类型分配权重"""
        weights = {"综合性回答": 1.0, "普通回答": 0.8, "特殊回答": 0.7}
        return weights.get(answer_type, 0.5)  # 默认权重为0.5

    def load_es_data(self):
        """从Elasticsearch加载数据并预处理"""
        if not self.es.is_connected():
            raise ConnectionError("无法连接到Elasticsearch服务器。")

        print(f"从Elasticsearch索引 '{self.es.index_name}' 加载所有原始文档...")
        raw_sources = self.es.scroll_all_raw_sources()
        question_data = []

        for item in raw_sources:
            question = item["question"]
            question_text = question["context"]
            question_id = question["id"]
            answers = item["answer"]

            for answer in answers:
                answer["answer_context"] = answer["context"]
                answer["answer_id"] = answer["id"]
                answer["answer_type"] = answer["type"]
            
            question_data.append(
                {
                    # "question_id": question_id,
                    # "question_text": question_text,
                    "question":{
                        "question_id": question_id,
                        "question_context": question_text,
                    },
                    "answers": answers,
                }
            )

        return question_data


def main():
    # 创建问答系统，启用本地Llama模型增强
    qa_system = BertQuestionAnswering(
        "data_database.json",
        use_llm_enhancement=False,
        # llm_model_path="Qwen/Qwen2.5-0.5B-Instruct",  # 替换为你的本地模型路径
        device="cuda",  # 如果没有GPU，可以设置为"cpu"
    )

    queries = ["数据库事务是什么？", "索引有什么用？", "什么是数据库范式？"]

    for query in queries:
        print(f"\n查询: {query}")
        print("=" * 70)

        candidate_answers = qa_system.get_answers(query, top_k_questions=5)

        for i, answer in enumerate(candidate_answers[:10]):
            print(f"候选答案 #{i + 1}:")
            print(f"问题: {answer['question_text']}")
            print(f"问题相似度: {answer['question_similarity']:.4f}")
            print(f"答案类型: {answer['answer_type']} (权重: {answer['answer_type_weight']:.2f})")
            print(f"综合置信度: {answer['confidence']:.4f}")

            # 如果有增强的答案，则显示增强答案
            if answer["enhanced"] and answer["enhanced_answer"]:
                print("原始答案:", answer["answer_text"])
                print("\n增强后的答案:", answer["enhanced_answer"])
            else:
                print(f"答案: {answer['answer_text']}")

            print("-" * 70)


if __name__ == "__main__":
    main()
