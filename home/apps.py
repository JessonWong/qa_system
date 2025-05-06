from django.apps import AppConfig
from .services.hiro_qa import BertQuestionAnswering


class HomeConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "home"


class HiroConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "home.hiro"

    qa_system = BertQuestionAnswering(
        es=True,
        use_llm_enhancement=False,
        # llm_model_path="Qwen/Qwen2.5-0.5B-Instruct",  # 替换为你的本地模型路径
        device="cuda",  # 如果没有GPU，可以设置为"cpu"
    )
