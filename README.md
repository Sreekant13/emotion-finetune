# Emotion Classification with Transformers

Fine-tuning transformer encoders for **6-class emotion detection** on the [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) dataset (sadness, joy, love, anger, fear, surprise), with a published model on the Hugging Face Hub.

🤗 **Published model:** [Sreekant13/distilbert-emotion](https://huggingface.co/Sreekant13/distilbert-emotion)

---

## What's here

Two self-contained training scripts, written as notebook-style cells (`# %%`) so they paste straight into Google Colab or Kaggle with a free T4 GPU:

| Script | Base model | Idea |
|--------|-----------|------|
| [`fine_tune_emotion.py`](fine_tune_emotion.py) | `distilbert-base-uncased` | Baseline. Learning-rate sweep, then a final run at the best LR, pushed to the Hub with an auto-generated model card. |
| [`fine_tune_emotion_v2.py`](fine_tune_emotion_v2.py) | `microsoft/deberta-v3-small` | Improvement experiment. A stronger base model plus **class-weighted cross-entropy** to help the rare emotions (love, surprise), with a clean **ablation** (weighted vs. plain) so the gain is attributable. |

---

## Results

**v1: DistilBERT baseline** (`distilbert-base-uncased`, 4 epochs, mixed-precision AMP)

| Split | Accuracy | Macro-F1 |
|-------|:--------:|:--------:|
| Validation | **0.9425** | **0.9171** |
| Test | 0.9230 | 0.8770 |

The gap between accuracy (0.92) and macro-F1 (0.88) on the test set is the whole point of v2: accuracy is dominated by the frequent classes, while macro-F1 exposes weaker performance on the rare ones.

**v2: DeBERTa-v3-small + class-weighted loss**

An ablation that trains DeBERTa-v3-small twice, with and without inverse-frequency class weights, and reports both against the v1 baseline on the test set. Run [`fine_tune_emotion_v2.py`](fine_tune_emotion_v2.py) to reproduce the comparison table; the better model by macro-F1 is pushed to the Hub.

---

## How to run

1. Open [Google Colab](https://colab.research.google.com) (`Runtime > Change runtime type > T4 GPU`) or a Kaggle notebook with GPU on.
2. Create a **write** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Paste a script in cell-by-cell (or upload and run it). Edit `HUB_ID` to your own `username/repo` before the final training cell.

Dependencies are installed by the first cell:

```bash
pip install -U "transformers>=4.44" datasets evaluate accelerate sentencepiece
```

---

## Method notes

- **Dynamic padding** via `DataCollatorWithPadding` keeps training fast instead of padding every batch to a fixed length.
- **Metrics:** accuracy and **macro-F1** (macro so every emotion counts equally, regardless of frequency).
- **Best-model selection:** `load_best_model_at_end` with `metric_for_best_model="f1"`.
- **v2 class weights:** inverse-frequency (`total / (num_classes * count_c)`), applied inside a custom `WeightedTrainer.compute_loss`.

---

## Stack

`PyTorch` · `Transformers` · `Datasets` · `Evaluate` · `Hugging Face Hub` · `DistilBERT` · `DeBERTa-v3`

## License

MIT. See [LICENSE](LICENSE).
