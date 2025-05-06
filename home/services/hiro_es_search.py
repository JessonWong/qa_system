from elasticsearch import Elasticsearch


class ESSearcher:
    def __init__(self, es_hosts=["localhost:9200"], index_name="courseqa"):
        """
        初始化Elasticsearch搜索器。

        参数:
            es_hosts: Elasticsearch主机列表。
            index_name: 要查询的索引名称。
        """
        self.es_hosts = es_hosts
        self.index_name = index_name
        self.es = None
        self._connect()

    def _connect(self):
        try:
            self.es = Elasticsearch(self.es_hosts)
            if not self.es.ping():
                print(f"无法连接到Elasticsearch服务器: {self.es_hosts}")
                self.es = None
        except Exception as e:
            print(f"连接Elasticsearch时发生错误: {e}")
            self.es = None

    def is_connected(self):
        return self.es is not None

    def scroll_all_raw_sources(self):
        """
        从Elasticsearch加载所有文档的原始 _source 内容。
        使用 scroll API 来高效获取所有文档。
        """
        if not self.is_connected():
            print("Elasticsearch未连接，无法加载数据。")
            return []

        print(f"从Elasticsearch索引 '{self.index_name}' 加载所有原始文档...")
        raw_sources = []
        try:
            query = {"query": {"match_all": {}}}
            # 初始请求，size可以根据情况调整，scroll API会处理后续分页
            resp = self.es.search(index=self.index_name, body=query, scroll="2m", size=100)

            scroll_id = resp.get("_scroll_id")
            hits = resp["hits"]["hits"]

            while scroll_id and len(hits) > 0:
                for hit in hits:
                    raw_sources.append(hit["_source"])  # 存储原始的 _source

                # 获取下一批结果
                resp = self.es.scroll(scroll_id=scroll_id, scroll="2m")
                scroll_id = resp.get("_scroll_id")
                hits = resp["hits"]["hits"]

            if scroll_id:  # 清除滚动上下文
                self.es.clear_scroll(scroll_id=scroll_id)

            print(f"成功从Elasticsearch加载了 {len(raw_sources)} 个原始文档。")
            return raw_sources

        except Exception as e:
            print(f"从Elasticsearch加载原始文档时发生错误: {e}")
            return []

    def fetch_all_questions(self):
        """
        从Elasticsearch加载所有文档的 'question' 字段。
        使用 _source filtering 和 scroll API。
        返回一个问题对象列表，例如：[{"id": "q1", "context": "text1"}, ...]
        """
        if not self.is_connected():
            print("Elasticsearch未连接，无法加载问题数据。")
            return []

        print(f"从Elasticsearch索引 '{self.index_name}' 加载所有问题数据 (仅question字段)...")
        questions_list = []
        try:
            query = {
                "_source": ["question"],  # 仅获取 question 字段
                "query": {"match_all": {}},
            }
            resp = self.es.search(index=self.index_name, body=query, scroll="2m", size=1000)

            scroll_id = resp.get("_scroll_id")
            hits = resp["hits"]["hits"]

            while scroll_id and len(hits) > 0:
                for hit in hits:
                    if "question" in hit["_source"] and hit["_source"]["question"]:
                        questions_list.append(hit["_source"]["question"])  # 添加 question 对象

                resp = self.es.scroll(scroll_id=scroll_id, scroll="2m")
                scroll_id = resp.get("_scroll_id")
                hits = resp["hits"]["hits"]

            if scroll_id:
                self.es.clear_scroll(scroll_id=scroll_id)

            print(f"成功从Elasticsearch加载了 {len(questions_list)} 个问题。")
            return questions_list

        except Exception as e:
            print(f"从Elasticsearch加载问题数据时发生错误: {e}")
            return []

    def search_documents_by_question_context(self, question_context_query, size=10):
        """
        根据问题上下文从Elasticsearch搜索文档并返回其原始 _source 内容列表。
        """
        if not self.is_connected():
            print("Elasticsearch未连接，无法搜索数据。")
            return []

        print(f"在Elasticsearch索引 '{self.index_name}' 中搜索问题上下文: '{question_context_query}'...")
        raw_sources = []
        try:
            query = {"query": {"match": {"question.context": question_context_query}}}
            # 对于特定查询，通常不需要scroll，除非预期结果集非常大且需要全部处理
            resp = self.es.search(index=self.index_name, body=query, size=size)

            hits = resp["hits"]["hits"]
            for hit in hits:
                raw_sources.append(hit["_source"])  # 存储原始的 _source

            print(f"查询 '{question_context_query}' 找到了 {len(raw_sources)} 个匹配的原始文档。")
            return raw_sources

        except Exception as e:
            print(f"根据问题上下文搜索文档时发生错误: {e}")
            return []

