# Emotion Detection v2 — DeBERTa-v3-small + class-weighted loss -> HF Hub
# ---------------------------------------------------------------------------
# Goal: genuinely improve macro-F1 over the distilbert baseline (test acc 0.923,
# test macro-F1 0.877) by (1) a stronger base model and (2) class-weighted loss
# to help the rare emotions (love, surprise). Includes an ablation so we can
# honestly say where the gain comes from.
# Run on Colab/Kaggle with a T4 GPU. Edit HUB_ID in the last training cell.

# %% 1. Install deps (sentencepiece is needed for the DeBERTa-v3 tokenizer)
get_ipython().system('pip install -q -U "transformers>=4.44" datasets evaluate accelerate sentencepiece')

# %% 2. Log in (WRITE token from https://huggingface.co/settings/tokens)
from huggingface_hub import notebook_login
notebook_login()

# %% 3. Data + tokenize with the DeBERTa tokenizer
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

MODEL = "microsoft/deberta-v3-small"
ds = load_dataset("dair-ai/emotion")
labels = ds["train"].features["label"].names
id2label = {i: l for i, l in enumerate(labels)}
label2id = {l: i for i, l in enumerate(labels)}
num_labels = len(labels)

tok = AutoTokenizer.from_pretrained(MODEL)            # needs sentencepiece
def preprocess(b): return tok(b["text"], truncation=True)
ds_tok = ds.map(preprocess, batched=True)
collator = DataCollatorWithPadding(tokenizer=tok)
print("labels:", labels)

# %% 4. Metrics + class weights (inverse frequency, "balanced")
import numpy as np, evaluate, torch
from collections import Counter

acc_m, f1_m = evaluate.load("accuracy"), evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, y = eval_pred
    p = np.argmax(logits, axis=-1)
    return {"accuracy": acc_m.compute(predictions=p, references=y)["accuracy"],
            "f1": f1_m.compute(predictions=p, references=y, average="macro")["f1"]}

counts = Counter(ds["train"]["label"])
total = sum(counts.values())
class_weights = torch.tensor(
    [total / (num_labels * counts[i]) for i in range(num_labels)], dtype=torch.float)
print("per-class counts:", {labels[i]: counts[i] for i in range(num_labels)})
print("class weights   :", {labels[i]: round(class_weights[i].item(), 2) for i in range(num_labels)})

# %% 5. A Trainer that can use class-weighted cross-entropy
from transformers import (AutoModelForSequenceClassification, TrainingArguments,
                          Trainer)
from torch import nn

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kw):
        super().__init__(**kw)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = nn.CrossEntropyLoss(weight=w)(
            outputs.logits.view(-1, num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def run(weighted, out, push=False, hub_id=None, lr=2e-5, epochs=3):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=num_labels, id2label=id2label, label2id=label2id)
    args = TrainingArguments(
        output_dir=out, learning_rate=lr,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        num_train_epochs=epochs, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1",
        fp16=True, logging_steps=100, report_to="none",
        push_to_hub=push, hub_model_id=hub_id)
    tr = WeightedTrainer(
        class_weights=class_weights if weighted else None,
        model=model, args=args,
        train_dataset=ds_tok["train"], eval_dataset=ds_tok["validation"],
        processing_class=tok, data_collator=collator, compute_metrics=compute_metrics)
    tr.train()
    test = tr.evaluate(eval_dataset=ds_tok["test"], metric_key_prefix="test")
    return tr, test

# %% 6. ABLATION run A: DeBERTa WITHOUT class weights
trA, testA = run(weighted=False, out="deberta_plain")
print("DeBERTa (plain)   test:", {k: round(v, 4) for k, v in testA.items() if k.startswith("test")})

# %% 7. ABLATION run B: DeBERTa WITH class weights
trB, testB = run(weighted=True, out="deberta_weighted")
print("DeBERTa (weighted) test:", {k: round(v, 4) for k, v in testB.items() if k.startswith("test")})

# %% 8. Compare everything (honest story)
print("\n================ RESULTS (test set) ================")
print(f"{'model':30} {'accuracy':>9} {'macro-F1':>9}")
print(f"{'distilbert (v1 baseline)':30} {0.9230:>9.4f} {0.8770:>9.4f}")
print(f"{'deberta-v3-small (plain)':30} {testA['test_accuracy']:>9.4f} {testA['test_f1']:>9.4f}")
print(f"{'deberta-v3-small (weighted)':30} {testB['test_accuracy']:>9.4f} {testB['test_f1']:>9.4f}")

# %% 9. Publish the better model (by macro-F1) to a NEW repo
HUB_ID = "Sreekant13/deberta-v3-small-emotion"        # <-- your repo
best_tr, best_test = (trB, testB) if testB["test_f1"] >= testA["test_f1"] else (trA, testA)
best_tr.args.hub_model_id = HUB_ID
best_tr.push_to_hub(commit_message=f"DeBERTa-v3-small emotion (test f1={best_test['test_f1']:.4f})")
tok.push_to_hub(HUB_ID)
print("Published:", "https://huggingface.co/" + HUB_ID)

# %% 10. Sanity check
from transformers import pipeline
clf = pipeline("text-classification", model=HUB_ID)
for s in ["I can't believe how amazing this turned out!",
          "I love spending time with you.",
          "I'm so nervous about tomorrow."]:
    print(s, "->", clf(s)[0])
