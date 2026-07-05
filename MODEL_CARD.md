---
language:
- en
license: apache-2.0
base_model: distilbert-base-uncased
library_name: transformers
pipeline_tag: text-classification
tags:
- text-classification
- emotion
- emotion-detection
- distilbert
- sentiment
datasets:
- dair-ai/emotion
metrics:
- accuracy
- f1
model-index:
- name: distilbert-emotion
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: Emotion
      type: dair-ai/emotion
      split: test
    metrics:
    - type: accuracy
      value: 0.923
      name: Accuracy
    - type: f1
      value: 0.877
      name: Macro F1
widget:
- text: "I am so grateful this finally worked out."
- text: "I am really nervous about the interview tomorrow."
- text: "I cannot stop smiling today."
- text: "I did not expect that to happen at all."
---

# distilbert-emotion

This is a fine-tuned version of [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) for single-label **emotion classification** on the [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) dataset. Given a short piece of English text, the model predicts one of six emotions: **sadness, joy, love, anger, fear, or surprise**.

Training code is available at [github.com/Sreekant13/emotion-finetune](https://github.com/Sreekant13/emotion-finetune).

## Results

The final model is the best checkpoint by validation macro-F1 (kept automatically via `load_best_model_at_end`).

| Split | Accuracy | Macro F1 |
|-------|:--------:|:--------:|
| Validation | 0.9425 | 0.9171 |
| Test (held out) | 0.9230 | 0.8770 |

Macro-F1 is lower than accuracy because the two rarest classes (love and surprise) are harder and count equally under macro averaging. This gap is the motivation for the follow-up experiment described at the bottom of this card.

## Labels

The model uses the standard `dair-ai/emotion` label order:

| ID | Emotion |
|----|---------|
| 0 | sadness |
| 1 | joy |
| 2 | love |
| 3 | anger |
| 4 | fear |
| 5 | surprise |

## How to use

With the `transformers` pipeline:

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="Sreekant13/distilbert-emotion")

classifier("I am so grateful this finally worked out.")
# [{'label': 'joy', 'score': 0.99...}]
```

To return scores for every emotion, pass `top_k=None`:

```python
classifier("I am really nervous about the interview tomorrow.", top_k=None)
```

With the model and tokenizer directly:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "Sreekant13/distilbert-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

inputs = tokenizer("I cannot stop smiling today.", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_id = logits.argmax(-1).item()
print(model.config.id2label[predicted_id])
```

## Intended uses and limitations

**Intended uses.** Classifying short, informal English text into one of the six emotions above. Reasonable fits include tagging user feedback, routing support messages, adding affect signals to a chatbot, or lightweight social-media analysis.

**Limitations.**

- The `dair-ai/emotion` data is drawn from English Twitter messages, so the model is tuned to short, informal, social-media style text. Expect weaker performance on long documents, formal writing, other domains, or languages other than English.
- The model predicts exactly one emotion per input. It does not handle mixed or multi-label emotions, and it has no explicit "neutral" class.
- Subtle phenomena such as sarcasm, irony, and negation can be misclassified.
- Per-class performance is uneven. The rare classes (love, surprise) score lower than the frequent ones, as reflected in the macro-F1 versus accuracy gap.
- This model should not be used on its own for high-stakes decisions about people.

## Training and evaluation data

[`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion), a dataset of English Twitter messages labeled with six emotions. The default split was used:

| Split | Examples |
|-------|:--------:|
| train | 16,000 |
| validation | 2,000 |
| test | 2,000 |

## Training procedure

The base learning rate was selected with a short sweep over {1e-5, 2e-5, 3e-5, 5e-5}, and 2e-5 gave the best validation macro-F1. The final model was then trained at that learning rate for four epochs, keeping the best checkpoint by validation macro-F1.

### Training hyperparameters

- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 32
- seed: 42
- optimizer: adamw_torch_fused (betas=(0.9, 0.999), epsilon=1e-08)
- lr_scheduler_type: linear
- num_epochs: 4
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy | Macro F1 |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:--------:|
| 0.2199 | 1.0 | 1000 | 0.1885 | 0.9260 | 0.8994 |
| 0.1317 | 2.0 | 2000 | 0.1548 | 0.9345 | 0.9067 |
| 0.0963 | 3.0 | 3000 | 0.1578 | 0.9425 | 0.9171 |
| 0.0664 | 4.0 | 4000 | 0.1587 | 0.9425 | 0.9155 |

The epoch 3 checkpoint was retained as the final model because it had the highest validation macro-F1 (0.9171). Training past that point began to overfit, with validation loss rising slightly and macro-F1 dipping at epoch 4.

### Framework versions

- Transformers 5.12.1
- PyTorch 2.11.0+cu128
- Datasets 5.0.0
- Tokenizers 0.22.2

## Follow-up work

To close the gap between accuracy and macro-F1 on the rare classes, a second experiment fine-tunes `deberta-v3-small` with class-weighted cross-entropy and a clean weighted-versus-plain ablation. The code lives in the same repository: [github.com/Sreekant13/emotion-finetune](https://github.com/Sreekant13/emotion-finetune).

## Citation

Dataset (CARER):

```bibtex
@inproceedings{saravia-etal-2018-carer,
  title = "{CARER}: Contextualized Affect Representations for Emotion Recognition",
  author = "Saravia, Elvis and Liu, Hsien-Chi Toby and Huang, Yen-Hao and Wu, Junlin and Chen, Yi-Shin",
  booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  year = "2018",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/D18-1404",
  pages = "3687--3697"
}
```

Base model (DistilBERT):

```bibtex
@article{sanh2019distilbert,
  title = {DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author = {Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  journal = {arXiv preprint arXiv:1910.01108},
  year = {2019}
}
```
