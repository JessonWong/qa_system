#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from elasticsearch import Elasticsearch
import time

def import_data_to_es():
    # 连接到Elasticsearch实例
    print("连接到Elasticsearch...")
    es = Elasticsearch(hosts=["localhost:9200"])
    
    # 检查Elasticsearch连接状态
    if not es.ping():
        print("无法连接到Elasticsearch服务器！")
        return False
    
    # 检查索引是否存在，如果存在则删除（可选）
    if es.indices.exists(index="courseqa"):
        print("删除现有courseqa索引...")
        es.indices.delete(index="courseqa")
    
    # 创建索引及映射，使用简单映射以确保兼容性
    print("创建courseqa索引...")
    mapping = {
        "mappings": {
            "properties": {
                "question": {"type": "text"},
                "answer": {"type": "text"}
            }
        }
    }
    
    try:
        es.indices.create(index="courseqa", body=mapping)
    except Exception as e:
        print(f"创建索引失败: {str(e)}")
        return False
    
    # 加载数据
    print("加载courseqa.json文件...")
    try:
        with open("courseqa.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"加载数据失败: {str(e)}")
        return False
    
    # 导入数据到Elasticsearch
    print(f"开始导入{len(data)}条问答数据到Elasticsearch...")
    success_count = 0
    error_count = 0
    
    for i, qa_pair in enumerate(data):
        try:
            # 检查每个问题是否有答案
            if not qa_pair.get("answer"):
                print(f"跳过问题 {i+1} - 没有找到答案")
                error_count += 1
                continue
            
            # 找到最全面的答案或综合性回答
            best_answer = None
            for ans in qa_pair["answer"]:
                if "综合性回答" in ans.get("answer_type", ""):
                    best_answer = ans["answer_context"]
                    break
            
            # 如果没有找到综合性回答，选择第一个答案
            if best_answer is None and qa_pair["answer"]:
                best_answer = qa_pair["answer"][0]["answer_context"]
            
            # 创建文档，符合项目查询格式
            doc = {
                "question": qa_pair["question"]["question_context"],
                "answer": best_answer
            }
            
            # 索引文档
            es.index(index="courseqa", id=qa_pair["question"]["question_id"], body=doc)
            success_count += 1
            
            # 每100条打印进度
            if (i+1) % 100 == 0 or i+1 == len(data):
                print(f"已处理 {i+1}/{len(data)} 条记录")
                
        except Exception as e:
            print(f"导入问题 {i+1} 失败: {str(e)}")
            error_count += 1
    
    # 刷新索引使数据可搜索
    es.indices.refresh(index="courseqa")
    
    print(f"\n导入完成! 成功: {success_count}, 失败: {error_count}")
    print("尝试搜索第一条记录...")
    
    # 尝试搜索第一条记录，验证导入是否成功
    try:
        body = {"_source": ["question", "answer"], "query": {"match": {"question": data[0]["question"]["question_context"]}}}
        result = es.search(index="courseqa", body=body)
        if result["hits"]["total"]["value"] > 0:
            print("搜索测试成功! 找到记录.")
            print(f"问题: {result['hits']['hits'][0]['_source']['question']}")
            print(f"答案: {result['hits']['hits'][0]['_source']['answer']}")
        else:
            print("搜索测试失败! 未找到记录.")
    except Exception as e:
        print(f"搜索测试失败: {str(e)}")
    
    return True

if __name__ == "__main__":
    print("开始导入courseqa.json数据到Elasticsearch...")
    start_time = time.time()
    success = import_data_to_es()
    end_time = time.time()
    elapsed = end_time - start_time
    
    if success:
        print(f"导入完成! 耗时: {elapsed:.2f} 秒")
    else:
        print(f"导入失败! 耗时: {elapsed:.2f} 秒") 