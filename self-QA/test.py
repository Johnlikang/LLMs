import torch.cuda
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import dataloader
from tqdm.auto import tqdm
from modelscope import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("Langboat/mengzi-t5-base")
max_length = 512

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_loop(dataloader, model):
    total_bleu = [0.0] * 4
    smooth_fn = SmoothingFunction().method4
    count = 0
    preds, labels = [], []
    results = {}

    model.eval()
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=max_length,
            ).cpu().numpy()
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        label_tokens = batch_data["labels"].cpu().numpy()
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
        # BLEU
        for pred, label in zip(decoded_preds, decoded_labels):
            pred_tokens = list(pred.replace(" ", ""))
            ref_tokens = list(label.replace(" ", ""))
            total_bleu[0] += sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0),
                                           smoothing_function=smooth_fn)
            total_bleu[1] += sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0),
                                           smoothing_function=smooth_fn)
            total_bleu[2] += sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0),
                                           smoothing_function=smooth_fn)
            total_bleu[3] += sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                                           smoothing_function=smooth_fn)
            count += 1
        avg_bleu = [score / count for score in total_bleu]
        print(f"BLEU-1: {avg_bleu[0]:.4f}")
        print(f"BLEU-2: {avg_bleu[1]:.4f}")
        print(f"BLEU-3: {avg_bleu[2]:.4f}")
        print(f"BLEU-4: {avg_bleu[3]:.4f}")

        preds += [pred.strip() for pred in decoded_preds]
        labels += [[label.strip()] for label in decoded_labels]
    return avg_bleu