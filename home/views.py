from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import HttpResponse
from django.http import JsonResponse

import numpy as np
from tensorflow import keras
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs
from keras.layers import *
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from elasticsearch import Elasticsearch
import requests

from .services.sql_generator import generate_sql

# ES access
host = "localhost:9200"
es = Elasticsearch(host, maxsize=15)


def es_search_body(value, key):
    body = {"_source": ["question", "answer"], "query": {"match": {key: value}}}
    return body


def checkSimilarQuestion(quesstr):
    returnText = "在问答库中没有找到答案"
    key = "question"
    senttext1 = quesstr
    bd = es_search_body(senttext1, key)
    try:
        results = es.search(body=bd, index="courseqa")
        hits = results.get("hits", {}).get("hits", [])
        for hit in hits:
            if "_source" in hit and "question" in hit["_source"]:
                senttext2 = hit["_source"]["question"]
                returnText = hit["_source"]["answer"] + f" （来自问答：{senttext2}）"
                break
    except Exception as e:
        print(f"Error during Elasticsearch search: {e}")
        returnText = "在问答库中检索时发生错误。"
    return returnText


def outputStr(instr):
    text = instr.strip()
    rstr = "请把问题再描述详细一点。"

    sqlpos = text.find("sql语句")
    tablepos = text.find("在表格")
    if sqlpos >= 0 and tablepos >= 0:
        try:
            rstr = generate_sql(text)
        except Exception as e:
            print(f"Error in SQL generation: {e}")
            rstr = "处理SQL生成请求时出错。"
        return rstr

    pos = text.find("问答库")
    if pos >= 0:
        try:
            rstr = checkSimilarQuestion(text)
        except Exception as e:
            print(f"Error in Elasticsearch QA: {e}")
            rstr = "在问答库中检索时出错。"
        return rstr

    return rstr


@csrf_exempt
def getanswer(request):
    try:
        post_content = json.loads(request.body, encoding="utf-8")["content"]
        response_content = outputStr(post_content)
        return HttpResponse(response_content, content_type="text/plain; charset=utf-8")
    except json.JSONDecodeError:
        return HttpResponse("无效的请求格式。", status=400, content_type="text/plain; charset=utf-8")
    except KeyError:
        return HttpResponse("请求缺少 'content' 字段。", status=400, content_type="text/plain; charset=utf-8")
    except Exception as e:
        print(f"Error in getanswer view: {e}")
        return HttpResponse("服务器内部错误。", status=500, content_type="text/plain; charset=utf-8")


def index(request):
    data = {}
    data["name"] = "Tanch"
    data["message"] = "你好"
    return render(request, "./index.html", data)


def book_list(request):
    return HttpResponse("book content")
