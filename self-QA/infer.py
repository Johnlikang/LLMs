from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 加载您训练好的模型
qa_pipeline = pipeline(
    task=Tasks.text_generation,
    model='您的模型路径',
    device='cuda'
)

# 输入问题生成回答
question = "Java是什么？"
result = qa_pipeline(question)
print(result['text'])