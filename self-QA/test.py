import torch.cuda
from sacrebleu.metrics import BLEU
import dataloader
from tqdm.auto import tqdm
from modelscope import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("Langboat/mengzi-t5-base")
max_length = 512

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_loop(dataloader, model):
    preds, labels = [], []
    results = {}

    model.eval()
    for batch_data in tqdm(dataloader):
        with torch.no_grad():
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=max_length,
            ).cpu().numpy()
        label_tokens = batch_data["labels"].cpu().numpy()

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        preds += [pred.strip() for pred in decoded_preds]
        labels += [[label.strip()] for label in decoded_labels]
    for n in range(1, 5):
        bleu = BLEU(max_ngram_order=n)
        score = bleu.corpus_score(preds, labels)
        results[f"bleu{n}"] = score.score
    return results