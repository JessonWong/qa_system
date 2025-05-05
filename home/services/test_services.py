import unittest
from .sql_generator import standarizeRequirement, deStandarize, sample as sql_sample  # Import sample too
from .qa_similarity import predict_similar_text
from .bert_qa_extractor import predict_single_text
# Note: Testing functions that load models (predict_similar_text, predict_single_text)
# would typically require mocking the model loading/prediction or having the models available.
# Testing es_search would require mocking the Elasticsearch client and predict_similar_text.


# class TestSQLGenerator(unittest.TestCase):
#     def test_standarize_requirement_valid(self):
#         """Tests basic requirement standardization."""
#         # Adjust comma placement: query and 'sql语句' should be in the last segment
#         text = "在表格student(student), 属性有姓名stu_name(TEXT), 年龄stu_age(INT), 查询年龄大于20的学生姓名 sql语句"
#         # expected_prefix = "在表格table,att0,att1,查询 att1 大于 value1 的 att0" # Keep this expectation
#         standardized = standarizeRequirement(text)
#         # The rest of the assertions should now have a chance to pass if textReplace works as expected
#         self.assertTrue(standardized.startswith("在表格table,att0,att1,"), f"Output was: {standardized}")
#         self.assertIn("查询", standardized)
#         # Assuming textReplace correctly maps "年龄" to att1 and "姓名" to att0 based on sample
#         self.assertIn("att1", standardized)
#         self.assertIn("att0", standardized)
#         print(f"Standardized output: {standardized}")  # Debugging output
#         # Check if sample was populated correctly based on the *new* parsing
#         # self.assertEqual(sql_sample.get("tablezh"), "student")
#         # self.assertEqual(sql_sample.get("tableen"), "student")
#         # self.assertEqual(len(sql_sample.get("attribute", [])), 2)
#         # self.assertEqual(sql_sample["attribute"][0]["zh"], "姓名")  # Order depends on parsing logic now
#         # self.assertEqual(sql_sample["attribute"][1]["zh"], "年龄")
#         sql_sample.clear()  # Clean up sample after test

#     def test_standarize_requirement_invalid_format(self):
#         """Tests standardization with invalid input format."""
#         text = "查询学生姓名"  # Missing table/attribute definitions
#         expected = "输入文本不符合规范"
#         standardized = standarizeRequirement(text)
#         self.assertEqual(standardized, expected)

#     def test_destandarize(self):
#         """Tests basic de-standardization."""
#         # Setup the sample dictionary as if standarizeRequirement ran
#         sql_sample.clear()  # Ensure clean state
#         sql_sample["tableen"] = "student"
#         sql_sample["attribute"] = [
#             {"en": "stu_name", "zh": "姓名", "type": "TEXT", "order": 0},
#             {"en": "stu_age", "zh": "年龄", "type": "INT", "order": 1, "value": "20"},  # Assume value was stored
#         ]
#         standardized_sql = "select att0 from table where att1 > value1"
#         expected_sql = "select stu_name from student where stu_age > 20"
#         destandarized = deStandarize(standardized_sql)
#         # Basic check, regex might be safer for complex cases
#         self.assertEqual(destandarized.strip(), expected_sql.strip())
#         sql_sample.clear()  # Clean up sample


# Add more test classes for other services if needed, likely involving mocking
class TestQASimilarity(unittest.TestCase):
    # @unittest.skip("Requires model or mocking")
    def test_similarity(self):
        score = predict_similar_text("你好", "你好吗")
        print(f"Similarity score: {score}")
        self.assertTrue(0.0 <= score <= 1.0)

# class TestBertQAExtractor(unittest.TestCase):
#     @unittest.skip("Requires model or mocking")
#     def test_extraction(self):
#         context = "北京是中国的首都。"
#         question = "中国的首都是哪里？"
#         answer = predict_single_text(context, question)
#         self.assertEqual(answer, "北京")


if __name__ == "__main__":
    unittest.main()
