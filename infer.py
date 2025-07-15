import torch
from dataloader import model, tokenizer, device, max_length
from modelscope import AutoModelForSeq2SeqLM, AutoTokenizer

model_path = "./epoch_3_valid_bleu_0.71_model_weights.bin"
model.load_state_dict(torch.load(model_path))
model.eval()

# 输入问题生成回答
def generate_answer(context, question):
    input_seq = f"问题：{question}{tokenizer.sep_token}原文：{context}"
    inputs = tokenizer(
        input_seq,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            min_length=1,
            num_beams=4,  # 添加束搜索
            early_stopping=True,  # 找到有效EOS时停止
            no_repeat_ngram_size=2  # 防止重复
        )
    result = tokenizer.decode(output[0], skip_special_tokens=False).replace(" ", "")
    if result.startswith("<pad>答案:"):
        result = result[8:]
    if tokenizer.eos_token in result:
        result = result.split(tokenizer.eos_token)[0]
    return result

if __name__=="__main__":
    test_list = [
        {
            "context":"2025年3月13日至14日，将迎来近两年半以来的首次月全食，月亮将变成血红色，并持续65分钟！",
            "question":"两年半来首次月全食发生在什么时间？",
            "answer":"2025年3月13日至14日",
        },
        {
            "context": "如果说有一种食物能代表西安的味道，那一定是肉夹馍。",
            "question": "代表西安味道的食物是什么？",
            "answer": "肉夹馍",
        },
        {
            "context": "陕西联合足球俱乐部是一家位于陕西的职业足球俱乐部，主场位于渭南市体育中心体育场，现参加中国足球甲级联赛",
            "question": "陕西联合足球俱乐部的主场位于在哪？",
            "answer": "渭南市体育中心体育场",
        },
    ]
    for case in test_list:
        print(f"{case['context']}")
        print(f"{case['question']}")
        result = generate_answer(case['context'], case['question'])
        print(f"模型输出：{result}")
        print(f"参考答案：{case['answer']}")