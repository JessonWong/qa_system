# from django.shortcuts import render
# from django.views.decorators.csrf import csrf_exempt
# import json
# from django.http import HttpResponse
# # Removed unused imports and Elasticsearch related code

# from .services.sql_generator import generate_sql
# from .services.es_search import checkSimilarQuestion  # Import from the new module


# def outputStr(instr):
#     text = instr.strip()
#     rstr = "请把问题再描述详细一点。"

#     sqlpos = text.find("sql语句")
#     tablepos = text.find("在表格")
#     if sqlpos >= 0 and tablepos >= 0:
#         try:
#             rstr = generate_sql(text)
#         except Exception as e:
#             print(f"Error in SQL generation: {e}")
#             rstr = "处理SQL生成请求时出错。"
#         return rstr

#     pos = text.find("问答库")
#     if pos >= 0:
#         try:
#             # Call the imported function
#             rstr = checkSimilarQuestion(text)
#         except Exception as e:
#             print(f"Error in Elasticsearch QA: {e}")
#             rstr = "在问答库中检索时出错。"
#         return rstr

#     return rstr


# @csrf_exempt
# def getanswer(request):
#     try:
#         post_content = json.loads(request.body, encoding="utf-8")["content"]
#         response_content = outputStr(post_content)
#         return HttpResponse(response_content, content_type="text/plain; charset=utf-8")
#     except json.JSONDecodeError:
#         return HttpResponse("无效的请求格式。", status=400, content_type="text/plain; charset=utf-8")
#     except KeyError:
#         return HttpResponse("请求缺少 'content' 字段。", status=400, content_type="text/plain; charset=utf-8")
#     except Exception as e:
#         print(f"Error in getanswer view: {e}")
#         return HttpResponse("服务器内部错误。", status=500, content_type="text/plain; charset=utf-8")


# def index(request):
#     data = {}
#     data["name"] = "Tanch"
#     data["message"] = "你好"
#     return render(request, "./index.html", data)


# def book_list(request):
#     return HttpResponse("book content")


# # # BERT model
# # maxlen = 200

# # config_path = "bert_config.json"
# # checkpoint_path = "bert_model.ckpt"
# # dict_path = "vocab.txt"

# # token_dict = {}

# # with codecs.open(dict_path, "r", "utf8") as reader:
# #     for line in reader:
# #         token = line.strip()
# #         token_dict[token] = len(token_dict)


# # class OurTokenizer(Tokenizer):
# #     def _tokenize(self, text):
# #         R = []
# #         for c in text:
# #             if c in self._token_dict:
# #                 R.append(c)
# #             elif self._is_space(c):
# #                 R.append("[unused1]")  # space类用未经训练的[unused1]表示
# #             else:
# #                 R.append("[UNK]")  # 剩余的字符是[UNK]
# #         return R


# # tokenizer = OurTokenizer(token_dict)


# # class SquadExample:
# #     def __init__(self, question, context, start_char_idx, answer_text, all_answers):
# #         self.question = question
# #         self.context = context
# #         self.start_char_idx = start_char_idx
# #         self.answer_text = answer_text
# #         self.all_answers = all_answers
# #         self.input_ids = []
# #         self.skip = False

# #     def preprocess(self):
# #         context = self.context
# #         question = self.question
# #         answer_text = self.answer_text
# #         start_char_idx = self.start_char_idx

# #         # Clean context, answer and question
# #         context = " ".join(str(context).split())
# #         question = " ".join(str(question).split())
# #         answer = " ".join(str(answer_text).split())

# #         # if ("五" in answer_text):
# #         #     i=1

# #         # Find end character index of answer in context
# #         end_char_idx = start_char_idx + len(answer)
# #         if end_char_idx >= len(context):
# #             self.skip = True
# #             return

# #         # Mark the character indexes in context that are in answer
# #         is_char_in_ans = [0] * len(context)
# #         for idx in range(start_char_idx, end_char_idx):
# #             is_char_in_ans[idx] = 1

# #         # Tokenize context
# #         tokenized_context = tokenizer.encode(context)[0]
# #         self.context_token_to_char = tokenized_context
# #         # Find tokens that were created from answer characters
# #         ans_token_idx = []
# #         for i in range(start_char_idx, end_char_idx):
# #             ans_token_idx.append(i)
# #         # for idx in tokenized_context:
# #         #     if sum(is_char_in_ans[start_char_idx:end_char_idx]) > 0:
# #         #         ans_token_idx.append(idx)

# #         if len(ans_token_idx) == 0:
# #             self.skip = True
# #             return

# #         # Find start and end token index for tokens from answer
# #         self.start_token_idx = ans_token_idx[0]
# #         self.end_token_idx = ans_token_idx[-1]

# #         # Tokenize question
# #         tokenized_question = tokenizer.encode(question)[0]

# #         # Create inputs
# #         self.input_ids = tokenized_context + tokenized_question[1:]
# #         self.token_type_ids = [0] * len(tokenized_context) + [1] * len(tokenized_question[1:])
# #         self.attention_mask = [1] * len(self.input_ids)
# #         if len(self.input_ids) > maxlen:
# #             dist = 20
# #             a = maxlen - len(tokenized_question[1:])
# #             if a > self.start_token_idx + dist and a > self.end_token_idx + dist:
# #                 self.input_ids = tokenized_context[: a - 1] + tokenized_question[1:]
# #                 self.token_type_ids = [0] * len(tokenized_context[0 : a - 1]) + [1] * len(tokenized_question[1:])
# #                 self.attention_mask = [1] * len(self.input_ids)

# #         # Pad and create attention masks.
# #         # Skip if truncation is needed

# #         padding_length = maxlen - len(self.input_ids)
# #         if padding_length > 0:  # pad
# #             self.input_ids = self.input_ids + [0] * padding_length
# #             self.attention_mask = self.attention_mask + [0] * padding_length
# #             self.token_type_ids = self.token_type_ids + [0] * padding_length
# #             return
# #         elif padding_length < 0:  # skip
# #             self.skip = True
# #             return


# # def create_inputs_targets(squad_examples):
# #     dataset_dict = {
# #         "input_ids": [],
# #         "token_type_ids": [],
# #         "attention_mask": [],
# #         "start_token_idx": [],
# #         "end_token_idx": [],
# #     }
# #     for item in squad_examples:
# #         if item.skip == False:
# #             for key in dataset_dict:
# #                 dataset_dict[key].append(getattr(item, key))
# #     for key in dataset_dict:
# #         dataset_dict[key] = np.array(dataset_dict[key])

# #     x = [
# #         dataset_dict["input_ids"],
# #         dataset_dict["token_type_ids"],
# #     ]
# #     y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
# #     return x, y


# # bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
# # for l in bert_model.layers:
# #     l.trainable = True


# # ##QA model
# # def create_QA_model():
# #     input_ids = Input(shape=(None,))  # 待识别句子输入
# #     token_type_ids = Input(shape=(None,))  # 待识别句子输入
# #     attention_mask = Input(shape=(None,))  # 实体左边界（标签）
# #     # embedding = bert_model([input_ids, token_type_ids, attention_mask])
# #     x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), "float32"))(input_ids)

# #     embedding = bert_model([input_ids, token_type_ids])

# #     start_logits = Dense(1, name="start_logit", use_bias=False)(embedding)
# #     # start_logits = Flatten()(start_logits)
# #     start_logits = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([start_logits, x_mask])

# #     end_logits = Dense(1, name="end_logit", use_bias=False)(embedding)
# #     # end_logits = Flatten()(end_logits)
# #     end_logits = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([end_logits, x_mask])

# #     start_probs = Activation(keras.activations.softmax)(start_logits)
# #     end_probs = Activation(keras.activations.softmax)(end_logits)

# #     train_model = keras.Model(
# #         inputs=[input_ids, token_type_ids],
# #         outputs=[start_probs, end_probs],
# #     )
# #     qaloss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# #     qaoptimizer = keras.optimizers.Adam(lr=5e-5)
# #     train_model.compile(optimizer=qaoptimizer, loss=[qaloss, qaloss])
# #     return train_model


# # keras.backend.clear_session()
# # QAmodel = create_QA_model()
# # QAmodel.load_weights("ChineseQAweights.hdf5")


# # bert_model2 = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
# # for l in bert_model2.layers:
# #     l.trainable = True


# # # similar sentence
# # def create_model():
# #     x1_in = Input(shape=(None,))
# #     x2_in = Input(shape=(None,))

# #     x = bert_model2([x1_in, x2_in])
# #     x = Lambda(lambda x: x[:, 0])(x)
# #     simp = Dense(1, activation="sigmoid")(x)

# #     sim_model = keras.Model([x1_in, x2_in], simp)
# #     sim_model.compile(
# #         loss="binary_crossentropy",
# #         optimizer=Adam(1e-5),  # 用足够小的学习率
# #         metrics=["accuracy"],
# #     )
# #     return sim_model


# # keras.backend.clear_session()
# # simSentModel = create_model()
# # simSentModel.load_weights("ChineseSimSentWeights.hdf5")


# # def predict_similar_text(senttext1, senttext2):
# #     # 利用BERT进行tokenize
# #     senttext1 = senttext1[:maxlen]
# #     senttext2 = senttext2[:maxlen]

# #     x1, x2 = tokenizer.encode(first=senttext1, second=senttext2)
# #     X1 = x1 + [0] * (maxlen - len(x1)) if len(x1) < maxlen else x1
# #     X2 = x2 + [0] * (maxlen - len(x2)) if len(x2) < maxlen else x2

# #     # 模型预测并输出预测结果
# #     predicted = simSentModel.predict([np.array([X1]), np.array([X2])])[0]
# #     y1 = predicted[0]
# #     print(senttext2 + ":" + str(y1))
# #     return y1


# # # 对单句话进行预测
# # def predict_single_text(text1, text2):
# #     # 利用BERT进行tokenize
# #     pred_ans = ""
# #     squad_examples = []
# #     context = text1
# #     question = text2
# #     print(question)
# #     answer_text = "answer"
# #     all_answers = ["answers1", "answers1"]
# #     start_char_idx = 1
# #     squad_eg = SquadExample(question, context, start_char_idx, answer_text, all_answers)
# #     squad_eg.preprocess()
# #     if squad_eg.skip == False:
# #         squad_examples.append(squad_eg)

# #     x_test, y_test = create_inputs_targets(squad_examples)
# #     pred_start, pred_end = QAmodel.predict(x_test)
# #     for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
# #         squad_eg = squad_examples[idx]
# #         offsets = squad_eg.context_token_to_char
# #         start = np.argmax(start)
# #         end = np.argmax(end)
# #         print(start)
# #         print(end)
# #         # adjust the answer scope
# #         if start > end:
# #             temp = end
# #             end = start
# #             start = temp - 1
# #         else:
# #             end += 1

# #         if start >= len(offsets):
# #             continue
# #         # pred_char_start = offsets[start][0]
# #         if end < len(offsets):
# #             #    pred_char_end = offsets[end][1]
# #             #     pred_ans = squad_eg.context[pred_char_start:pred_char_end]
# #             pred_ans = squad_eg.context[start:end]
# #         else:
# #             #    pred_ans = squad_eg.context[pred_char_start:]
# #             pred_ans = squad_eg.context[start:]
# #     return pred_ans


# # def es_search_body(value, key):
# #     """
# #     将传参封装为es查询的body，可根据实际需求，判断是否新增此函数
# #     :param value:
# #     :param key:
# #     :return:
# #     """
# #     body = {"_source": ["question", "answer"], "query": {"match": {key: value}}}
# #     return body


# # def checkSimilarQuestion(quesstr):
# #     returnText = "在问答库中没有找到答案"
# #     key = "question"
# #     senttext1 = quesstr  # "解释exists查询的作用"
# #     bd = es_search_body(senttext1, key)
# #     print(senttext1)
# #     results = es.search(body=bd, index="courseqa")
# #     l = len(results["hits"]["hits"])
# #     print(l)
# #     for i in range(l):
# #         senttext2 = results["hits"]["hits"][i]["_source"]["question"]
# #         y = predict_similar_text(senttext1, senttext2)
# #         if y >= 0.8:
# #             print(senttext2 + "  score: " + str(y))
# #             returnText = results["hits"]["hits"][i]["_source"]["answer"] + "  （来自问答：" + senttext2 + " )"
# #             print(returnText)
# #             break
# #     return returnText


# # def outputStr(instr):
# #     text = instr.strip()
# #     rstr = "请把问题再描述详细一点。"
# #     sqlpos = text.find("sql语句")
# #     tablepos = text.find("在表格")
# #     print(sqlpos)
# #     if sqlpos >= 0 and tablepos >= 0:
# #         rstr = testtext(text)
# #         return rstr
# #     pos = text.find("问答库")
# #     print(pos)
# #     if pos >= 0:
# #         rstr = checkSimilarQuestion(text)
# #         return rstr
# #     if len(text) > 2:
# #         text2 = text.split("问")
# #         ques = text2[-1]
# #         text1 = text2[:-1]
# #         pos = text1[0].rfind("。")
# #         context = text1[0][0 : pos + 1]
# #         rstr = predict_single_text(context, ques)
# #         print(context + " : " + ques)
# #     return rstr


# # # load models in the first time
# # text = "解释exists查询的作用"
# # rstr = checkSimilarQuestion(text)
# # text1 = "在Transformer中，多头自注意力用于表示输入序列和输出序列，不过解码器必须通过掩蔽机制来保留自回归属性。Transformer的优点是并行性非常好，符合目前的硬件（主要指GPU）环境。"
# # text2 = "Transformer有什么优点？"
# # rstr = predict_single_text(text1, text2)


# # # Create your views here.


# # @csrf_exempt
# # def getanswer(request):
# #     print("start getanswer:")
# #     post_content = json.loads(request.body, encoding="utf-8")["content"]
# #     # post_content = json.loads(request.body, encoding='utf-8')['content']
# #     print(type(post_content))
# #     post_content = outputStr(post_content)
# #     print(post_content)
# #     # return JsonResponse({'content1': 'post请求'+post_content})
# #     return HttpResponse(post_content)


# # def index(request):
# #     # name = "Hello DTL!"
# #     data = {}
# #     data["name"] = "Tanch"
# #     data["message"] = "你好"
# #     # return render(request,"模板文件路径",context={字典格式:要在客户端中展示的数据})
# #     # context是个字典
# #     # return render(request,"./index.html",context={"name":name})
# #     return render(request, "./index.html", data)


# # def book_list(request):
# #     return HttpResponse("book content")


from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse  # Import JsonResponse
import json  # Import json for parsing request body

# Import services
from .services.sql_generator import generate_sql
from .services.es_search import checkSimilarQuestion
from .services.bert_qa_extractor import predict_single_text

# Removed unused outputStr function and commented-out BERT code


@csrf_exempt
def index(request):
    """Renders the main chat interface."""
    return render(request, "home/index.html")  # Assuming index.html is the main page


# Removed unused book_list function


@csrf_exempt
def getanswer(request):
    """
    Handles incoming questions, determines the type, calls the appropriate service,
    and returns the answer.
    """
    if request.method == "POST":
        try:
            # Assuming the question comes in JSON format: {'question': 'user question'}
            data = json.loads(request.body)
            question = data.get("question", "").strip()
            context = data.get("context", None)  # Optional context for text QA

            if not question:
                return JsonResponse({"answer": "请输入问题。"})

            answer = "抱歉，无法处理您的问题。"  # Default answer

            # --- Logic to determine which service to call ---
            if question.startswith("在表格"):
                # Assume it's a request for SQL generation
                answer = generate_sql(question)
            elif context:
                # If context is provided, use the BERT QA extractor
                answer = predict_single_text(context, question)
            else:
                # Default to searching the QA database
                # The checkSimilarQuestion function already handles the "问答库" prefix internally
                answer = checkSimilarQuestion(question)

            return JsonResponse({"answer": answer})

        except json.JSONDecodeError:
            return JsonResponse({"answer": "无效的请求格式。"}, status=400)
        except Exception as e:
            print(f"Error in getanswer view: {e}")  # Log the error
            return JsonResponse({"answer": "处理请求时发生服务器内部错误。"}, status=500)
    else:
        return JsonResponse({"answer": "仅支持POST请求。"}, status=405)


# Removed all the commented-out BERT model loading and SquadExample code
# as it's now encapsulated within the service modules (bert_qa_extractor, qa_similarity).