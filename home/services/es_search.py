from elasticsearch import Elasticsearch

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
