import json
import torch
from torch.utils.data import DataLoader, Dataset
from modelscope import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("mengzi-t5-base")
tokenizer = AutoTokenizer.from_pretrained("Langboat/mengzi-t5-base")
max_length = 512
class QAData(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self,data_file):
        Data = {}
        with open(data_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        return self.data[item]

def collote_fn(batch_datas):
    input_sentence = []
    question_sentence = []
    answer_sentence = []
    for data in batch_datas:
        # T5做QA需要添加前缀指定任务类型
        input_text = "answer question: " + data['question'] + " context: " + data['context']
        input_sentence.append(input_text)
        answer_sentence.append(data['answer'])
    x = tokenizer(
        input_sentence,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    y = tokenizer(
        answer_sentence,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).input_ids
    # 一定要记得decoder_input_ids构建
    decoder_input_ids = tokenizer.pad_token_id * torch.ones_like(y)
    decoder_input_ids[1:] = y[:-1]
    collote_datas = {
            "input_ids": x["input_ids"],
            "attention_mask": x["attention_mask"],
            "labels": y,
            "decoder_input_ids": decoder_input_ids
        }
    return collote_datas

train_data = QAData("./dataset/train.json")
train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
valid_data = QAData("./dataset/dev.json")
valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=True, collate_fn=collote_fn)

if __name__ == "__main__":
    print(train_data[0])
    data = next(iter(train_dataloader))
    print(data)
