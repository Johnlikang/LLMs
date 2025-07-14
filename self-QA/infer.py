import torch
from dataloader import model, tokenizer, device, max_length
from modelscope import AutoModelForSeq2SeqLM, AutoTokenizer

model_path = ""
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
    return result

if __name__=="__main__":
    result = generate_answer("中国首都是北京，位于华北平原。", "中国的首都是哪里？")
    print(pred)