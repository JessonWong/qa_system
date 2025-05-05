from elasticsearch import Elasticsearch
from .qa_similarity import predict_similar_text  # Import the similarity function

# ES access
host = "localhost:9200"
es = Elasticsearch(host, maxsize=15)
SIMILARITY_THRESHOLD = 0.8  # Define a threshold for similarity


def es_search_body(value, key):
    body = {"_source": ["question", "answer"], "query": {"match": {key: value}}}
    return body


def checkSimilarQuestion(quesstr):
    """Searches Elasticsearch and uses BERT similarity to find the best match."""
    returnText = "在问答库中没有找到答案"
    key = "question"
    # Remove "问答库" prefix if present, as it's likely not part of the actual question
    senttext1 = quesstr.replace("问答库", "").strip()
    if not senttext1:
        return "请输入有效的问题。"

    bd = es_search_body(senttext1, key)
    try:
        results = es.search(body=bd, index="courseqa")
        hits = results.get("hits", {}).get("hits", [])
        best_score = -1.0
        best_match_answer = ""
        best_match_question = ""

        print(f"ES Search for '{senttext1}' returned {len(hits)} hits.")  # Debugging

        for hit in hits:
            if "_source" in hit and "question" in hit["_source"] and "answer" in hit["_source"]:
                senttext2 = hit["_source"]["question"]
                answer = hit["_source"]["answer"]

                # Calculate similarity score
                similarity_score = predict_similar_text(senttext1, senttext2)
                print(f"Comparing with '{senttext2}': Score = {similarity_score}")  # Debugging

                if similarity_score >= SIMILARITY_THRESHOLD and similarity_score > best_score:
                    best_score = similarity_score
                    best_match_answer = answer
                    best_match_question = senttext2
                    print(f"New best match found: Score={best_score}")  # Debugging

        if best_match_answer:
            returnText = f"{best_match_answer} （来自问答：{best_match_question}）"
            print(f"Final Answer: {returnText}")  # Debugging
        else:
            print("No sufficiently similar question found.")  # Debugging

    except Exception as e:
        print(f"Error during Elasticsearch search or similarity check: {e}")
        returnText = "在问答库中检索时发生错误。"

    return returnText
