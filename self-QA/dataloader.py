import json
import torch
from torch.utils.data import DataLoader, Dataset
from modelscope import AutoModelForSeq2SeqLM, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForSeq2SeqLM.from_pretrained("mengzi-t5-base")
model = model.to(device)
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
        seq = self.data[item]
        # T5做QA需要添加前缀指定任务类型
        input_seq = f"问题:{seq['question']}{tokenizer.sep_token}原文:{seq['context']}"
        output_seq = f"答案:{seq['answer']}"
        return input_seq, output_seq

def collote_fn(batch_datas):
    input_sentence = []
    answer_sentence = []
    for (input_seq, output_seq) in batch_datas:
        input_sentence.append(input_seq)
        answer_sentence.append(output_seq)
    batch_data = tokenizer(
        input_sentence,
        text_target=answer_sentence,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    # 一定要记得decoder_input_ids构建
    batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(batch_data['labels'])
    # 处理标签
    end_token_index = torch.where(batch_data['labels'] == tokenizer.eos_token_id)[1]
    # 添加padding
    for idx, end_idx in enumerate(end_token_index):
        batch_data['labels'][idx][end_idx+1:] = -100
    return batch_data

if __name__ == "__main__":
    train_data = QAData("./dataset/train.json")
    dataloader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=collote_fn)
    # 分词测试
    token = tokenizer.tokenize("今天天气真热")
    print(token)
    # dataset测试
    print(train_data[0])
    # dataload测试
    data = next(iter(dataloader))
    print(data)
