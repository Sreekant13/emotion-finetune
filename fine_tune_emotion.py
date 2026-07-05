# Fine-tune DistilBERT for Emotion Detection -> publish to the Hugging Face Hub
# ---------------------------------------------------------------------------
# Run on Google Colab (Runtime > Change runtime type > T4 GPU) or Kaggle (GPU on).
# Each "# %%" marks a notebook cell. Paste cell-by-cell into Colab, or upload this
# file and run it. Only thing you MUST edit: HUB_ID (your username/repo) in cell 9.

# %% 1. Install dependencies
# (Colab has most of these; we just make sure they're current.)
get_ipython().system('pip install -q -U "transformers>=4.44" datasets evaluate accelerate')

# %% 2. Log in to the Hugging Face Hub
# Create a WRITE token at https://huggingface.co/settings/tokens, then paste it here.
from huggingface_hub import notebook_login
notebook_login()

# %% 3. Load and inspect the dataset (dair-ai/emotion: 6 emotions)
from datasets import load_dataset

ds = load_dataset("dair-ai/emotion")
print(ds)
labels = ds["train"].features["label"].names          # ['sadness','joy','love','anger','fear','surprise']
id2label = {i: l for i, l in enumerate(labels)}
label2id = {l: i for i, l in enumerate(labels)}
num_labels = len(labels)
print("labels:", labels)
print("example:", ds["train"][0])

# %% 4. Tokenize
from transformers import AutoTokenizer, DataCollatorWithPadding

MODEL = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(MODEL)

def preprocess(batch):
    return tok(batch["text"], truncation=True)

ds_tok = ds.map(preprocess, batched=True)
collator = DataCollatorWithPadding(tokenizer=tok)      # dynamic padding = faster

# %% 5. Metrics (accuracy + macro-F1)
import numpy as np, evaluate

acc_m = evaluate.load("accuracy")
f1_m = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, y = eval_pred
    p = np.argmax(logits, axis=-1)
    return {
        "accuracy": acc_m.compute(predictions=p, references=y)["accuracy"],
        "f1": f1_m.compute(predictions=p, references=y, average="macro")["f1"],
    }

# %% 6. Training helper (fresh model each run)
from transformers import (AutoModelForSequenceClassification, TrainingArguments,
                          Trainer)

def make_trainer(lr, epochs, out, push=False, hub_id=None):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=num_labels, id2label=id2label, label2id=label2id)
    args = TrainingArguments(
        output_dir=out,
        learning_rate=lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        eval_strategy="epoch",        # if your transformers is old: evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,                    # T4 speedup
        logging_steps=100,
        report_to="none",             # no wandb prompt
        push_to_hub=push,
        hub_model_id=hub_id,
    )
    return Trainer(
        model=model, args=args,
        train_dataset=ds_tok["train"], eval_dataset=ds_tok["validation"],
        processing_class=tok, data_collator=collator, compute_metrics=compute_metrics,
    )

# %% 7. BASELINE (default learning rate 2e-5)
baseline = make_trainer(lr=2e-5, epochs=3, out="baseline")
baseline.train()
baseline_val = baseline.evaluate()
print("BASELINE validation:", baseline_val)

# %% 8. HYPERPARAMETER SWEEP over learning rate (2 epochs each to stay quick)
results = {2e-5: baseline_val["eval_f1"]}   # reuse baseline's LR result
for lr in [1e-5, 3e-5, 5e-5]:
    t = make_trainer(lr=lr, epochs=2, out=f"sweep_{lr}")
    t.train()
    m = t.evaluate()
    results[lr] = m["eval_f1"]
    print(f"  lr={lr}:  val_f1={m['eval_f1']:.4f}  val_acc={m['eval_accuracy']:.4f}")

best_lr = max(results, key=results.get)
print("\nsweep results (lr -> val_f1):", {f"{k:.0e}": round(v, 4) for k, v in results.items()})
print("BEST learning rate:", best_lr)

# %% 9. FINAL training with the best LR (more epochs), then push to the Hub
HUB_ID = "Sreekant13/distilbert-emotion"     # <-- EDIT: your-username/repo-name

final = make_trainer(lr=best_lr, epochs=4, out="final", push=True, hub_id=HUB_ID)
final.train()

val_metrics = final.evaluate()
test_metrics = final.evaluate(eval_dataset=ds_tok["test"], metric_key_prefix="test")
print("FINAL validation:", val_metrics)
print("FINAL test:      ", test_metrics)
print(f"\nBaseline val_f1={baseline_val['eval_f1']:.4f}  ->  Tuned val_f1={val_metrics['eval_f1']:.4f}")

# %% 10. Push model + tokenizer + auto-generated model card (with metrics)
final.push_to_hub(commit_message=f"Emotion DistilBERT (test acc={test_metrics['test_accuracy']:.4f})")
tok.push_to_hub(HUB_ID)
print("Published:  https://huggingface.co/" + HUB_ID)

# %% 11. Sanity check: load your published model and run it
from transformers import pipeline
clf = pipeline("text-classification", model=HUB_ID)
for s in ["I can't believe how amazing this turned out!",
          "I'm so nervous about tomorrow.",
          "Why does this always happen to me, it's infuriating."]:
    print(s, "->", clf(s)[0])
