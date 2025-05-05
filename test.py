import json
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def preprocess_data(data):
    question_data = []
    
    for item in data:
        question_text = item['question']['question_context']
        question_id = item['question']['question_id']
        answers = item['answers']
        
        question_data.append({
            'question_id': question_id,
            'question_text': question_text,
            'answers': answers
        })
    
    return question_data

class BertEncoder:
    def __init__(self, model_name='bert-base-chinese'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        
    def encode(self, texts, batch_size=8):
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 对文本进行编码
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                          return_tensors='pt', max_length=512)
            
            # 获取BERT输出
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                
            # 使用[CLS]标记的输出作为整个句子的表示
            sentence_embeddings = model_output.last_hidden_state[:, 0, :].numpy()
            all_embeddings.append(sentence_embeddings)
            
        return np.vstack(all_embeddings)


class BertQuestionAnswering:
    def __init__(self, data_path):

        self.data = load_dataset(data_path)
        self.question_data = preprocess_data(self.data)

        self.questions = [item['question_text'] for item in self.question_data]

        self.encoder = BertEncoder()
        
        self.question_embeddings = self.encoder.encode(self.questions)
    
    def get_answers(self, query, top_k_questions=3):
        
        query_embedding = self.encoder.encode([query])

        similarities = cosine_similarity(query_embedding, self.question_embeddings)[0]
        
        top_indices = similarities.argsort()[-top_k_questions:][::-1]

        all_candidate_answers = []
        
        for idx in top_indices:
            question_similarity = similarities[idx]
            question_info = self.question_data[idx]
            
            for answer in question_info['answers']:
                answer_type_weight = self._get_answer_type_weight(answer['answer_type'])
                
                confidence = question_similarity * answer_type_weight
                
                all_candidate_answers.append({
                    'question_id': question_info['question_id'],
                    'question_text': question_info['question_text'],
                    'question_similarity': question_similarity,
                    'answer_id': answer['answer_id'],
                    'answer_type': answer['answer_type'],
                    'answer_text': answer['answer_context'],
                    'answer_type_weight': answer_type_weight,
                    'confidence': confidence
                })
        
        all_candidate_answers.sort(key=lambda x: x['confidence'], reverse=True)
        
        return all_candidate_answers
    
    def _get_answer_type_weight(self, answer_type):
        """根据答案类型分配权重"""
        weights = {
            '综合性回答': 1.0,
            '普通回答': 0.8,
            '特殊回答': 0.7
        }
        return weights.get(answer_type, 0.5)  # 默认权重为0.5

def main():
    qa_system = BertQuestionAnswering('data_database.json')
    
    queries = [
        "数据库事务是什么？",
        "索引有什么用？",
        "什么是数据库范式？"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        print("=" * 70)
        
        candidate_answers = qa_system.get_answers(query, top_k_questions=5)
        
        for i, answer in enumerate(candidate_answers[:10]):
            print(f"候选答案 #{i+1}:")
            print(f"问题: {answer['question_text']}")
            print(f"问题相似度: {answer['question_similarity']:.4f}")
            print(f"答案类型: {answer['answer_type']} (权重: {answer['answer_type_weight']:.2f})")
            print(f"综合置信度: {answer['confidence']:.4f}")
            print(f"答案: {answer['answer_text']}")
            print("-" * 70)

if __name__ == "__main__":
    main()