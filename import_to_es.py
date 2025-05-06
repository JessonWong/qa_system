#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from elasticsearch import Elasticsearch
import time


def perform_search_test(es_instance, index_name):
    """
    对指定索引执行一个测试搜索。
    """
    print(f"尝试在索引 '{index_name}' 中搜索记录...")
    try:
        # 示例查询：查找 answer.id 为 "1" 的文档
        # 您可以根据需要修改此查询
        # body = {
        #     "_source": ["question", "answer"],
        #     "query": {
        #         "nested": {
        #             "path": "answer",
        #             "query": {"match": {"answer.id": "1"}},  # 假设 answer 对象中的 id 字段名为 "id"
        #             # 根据您的映射，这里应该是 "answer.id"
        #         }
        #     },
        # }

        body = {
            "query": {
                "nested": {
                    "path": "answer",
                    "query": {"term": {"answer.id": 1}},
                    "inner_hits": {"_source": True},
                }
            }
        }
        # 如果您的 answer id 字段在映射中是 answer_id，则使用：
        # body = {
        #     "_source": ["question", "answer"],
        #     "query": {"nested": {"path": "answer", "query": {"match": {"answer.answer_id": "1"}}}},
        # }

        # 或者，如果您想搜索导入的第一条记录（假设data变量在此作用域不可用，
        # 您可能需要一种不同的方式来获取第一个文档的标识符，或者传递它）
        # 例如，通过 question.id 来搜索（假设第一个文档的 question.id 是已知的或可获取的）
        # first_question_id = "some_known_id" # 您需要替换为实际的ID
        # body = {
        #     "_source": ["question", "answer"],
        #     "query": {"match": {"question.id": first_question_id}}
        # }

        result = es_instance.search(index=index_name, body=body)
        if result["hits"]["total"]["value"] > 0:
            print("搜索测试成功! 找到记录.")
            # 打印时，question现在是一个对象
            for i in range(len(result["hits"]["hits"])):
                print(f"记录 {i + 1}:")
                print(f"问题ID: {result['hits']['hits'][i]['_source']['question']['id']}")
                print(f"问题上下文: {result['hits']['hits'][i]['_source']['question']['context']}")
                print(f"答案ID: {result['hits']['hits'][i]['inner_hits']['answer']['hits']['hits'][0]['_source']['id']}")
                print(f"答案上下文: {result['hits']['hits'][i]['inner_hits']['answer']['hits']['hits'][0]['_source']['context']}")
                print(f"答案类型: {result['hits']['hits'][i]['inner_hits']['answer']['hits']['hits'][0]['_source']['type']}")
            # print(f"问题对象: {result['hits']['hits'][0]['_source']['question']}")
            # print(f"答案: {result['hits']['hits'][0]['_source']['answer']}")
        else:
            print("搜索测试失败! 未找到记录.")
    except Exception as e:
        print(f"搜索测试失败: {str(e)}")


def import_data_to_es():
    # 连接到Elasticsearch实例
    print("连接到Elasticsearch...")
    es = Elasticsearch(hosts=["localhost:9200"])

    # 检查Elasticsearch连接状态
    if not es.ping():
        print("无法连接到Elasticsearch服务器！")
        return False

    index_name = "courseqa"

    # 检查索引是否存在，如果存在则删除（可选）
    if es.indices.exists(index=index_name):
        print(f"删除现有 {index_name} 索引...")
        es.indices.delete(index=index_name)

    # 创建索引及映射，使用简单映射以确保兼容性
    print(f"创建 {index_name} 索引...")
    mapping = {
        "mappings": {
            "properties": {
                "question": {  # question字段现在是一个对象
                    "type": "object",
                    "properties": {
                        "id": {"type": "keyword"},  # question对象的id字段
                        "context": {"type": "text"},  # question对象的context字段
                    },
                },
                "answer": {
                    "type": "nested",
                    "properties": {
                        "id": {"type": "keyword"},  # answer 对象的 id 字段
                        "context": {"type": "text"},
                        "type": {"type": "keyword"},
                    },
                },
            }
        }
    }

    try:
        es.indices.create(index=index_name, body=mapping)
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
    first_question_id_for_test = None  # 用于后续搜索测试

    for i, qa_pair in enumerate(data):
        try:
            # 检查每个问题是否有答案
            if not qa_pair.get("answer") or not isinstance(qa_pair["answer"], list) or not qa_pair["answer"]:
                print(f"跳过问题 {i + 1} - 没有找到答案或答案格式不正确")
                error_count += 1
                continue

            # 收集所有答案
            answers_list = []
            for ans in qa_pair["answer"]:
                answers_list.append(
                    {
                        "id": ans.get("answer_id"),  # 确保这里是 "id" 以匹配映射
                        "context": ans.get("answer_context"),
                        "type": ans.get("answer_type"),
                    }
                )

            current_question_id = qa_pair["question"]["question_id"]
            if i == 0:  # 保存第一个问题的ID用于测试
                first_question_id_for_test = current_question_id

            # 创建文档，符合项目查询格式
            doc = {
                "question": {  # 构建嵌套的question对象
                    "id": current_question_id,
                    "context": qa_pair["question"]["question_context"],
                },
                "answer": answers_list,
            }

            # 索引文档
            es.index(index=index_name, id=current_question_id, body=doc)
            success_count += 1

            # 每100条打印进度
            if (i + 1) % 100 == 0 or i + 1 == len(data):
                print(f"已处理 {i + 1}/{len(data)} 条记录")

        except Exception as e:
            print(f"导入问题 {i + 1} 失败: {str(e)}")
            error_count += 1

    # 刷新索引使数据可搜索
    es.indices.refresh(index=index_name)

    print(f"\n导入完成! 成功: {success_count}, 失败: {error_count}")

    # 执行搜索测试
    perform_search_test(es, index_name)  # 调用新的搜索函数

    return True


if __name__ == "__main__":
    print("开始导入courseqa.json数据到Elasticsearch...")
    start_time = time.time()
    import_successful = import_data_to_es()  # 修改变量名以更清晰
    # perform_search_test(es, index_name)  # 调用新的搜索函数
    end_time = time.time()
    elapsed = end_time - start_time

    if import_successful:  # 根据导入函数的返回值判断
        print(f"数据处理流程完成! 总耗时: {elapsed:.2f} 秒")
    else:
        print(f"数据处理流程失败! 总耗时: {elapsed:.2f} 秒")
