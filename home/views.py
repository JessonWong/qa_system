from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import HttpResponse
# Removed unused imports and Elasticsearch related code

from .services.sql_generator import generate_sql
from .services.es_search import checkSimilarQuestion  # Import from the new module


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
            # Call the imported function
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
