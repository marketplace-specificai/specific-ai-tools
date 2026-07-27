# embedding_heads

Run SpecificAI **classification heads** on top of GGUF embeddings from
**Lemonade Server** (HTTP) or **llama-cpp-python** (in-process).

When BERT (and similar) models are converted to GGUF, the pooler and classifier
layers are dropped. This package loads those head layers (from `.npy` artifacts
or Hugging Face `safetensors`), takes CLS embeddings from an embedding backend,
and applies the same post-processing used on the SpecificAI platform
(single / multi-label, calibrated temperature, per-label thresholds).

## Install

```bash
pip install specific-ai-tools

# Optional: in-process GGUF via llama-cpp-python
pip install "specific-ai-tools[llamacpp]"
```

## Quick start — Lemonade Server (HTTP)

```python
from specific_ai_tools.embedding_heads import LemonadeEmbeddingClassifier

classifier = LemonadeEmbeddingClassifier(
    lemonade_model_name="user.email-agent-triage",
    checkpoint="specific-AI/email-agent-triage:bert-base-only.gguf",
    lemonade_base_url="http://localhost:13305",
)

predictions = classifier.predict(["Please escalate this ticket to billing"])
print(predictions[0].predicted_labels, predictions[0].predicted_confidences)
print(predictions[0].all_confidences)
```

On construction, `LemonadeEmbeddingClassifier`:

1. `GET /v1/models` — if the model is missing or not configured with
   `--embd-normalize "-1" --pooling cls`, it
   `POST /api/v1/pull` then `POST /api/v1/load` (`save_options: true`).
2. Resolves head/tokenizer artifacts from the Hugging Face model card in
   `checkpoint` (the segment before `:`).
3. At predict time, HF-tokenizes text and calls `POST /v1/embeddings` with
   token ids as `input`.

Override the head source with `model=` (local path or HF id).

> **Note:** Lemonade may have slightly better embedding performance than
> in-process `llama-cpp-python`, because it drives the native llama.cpp
> (`llama-server`) C++ path over HTTP rather than the Python bindings.

## Quick start — llama-cpp-python (GGUF in the model dir)

```python
from specific_ai_tools.embedding_heads import LlamaCppEmbeddingClassifier

classifier = LlamaCppEmbeddingClassifier(
    model="specific-AI/email-agent-triage",  # or local dir with head + GGUF
    gguf_filename="bert-base-only.gguf",           # file name inside that model dir
)

predictions = classifier.predict(["Please escalate this ticket to billing"])
```

Uses CLS pooling and `normalize=False` so embeddings stay raw for the head.

## Split model → head `.npy` + GGUF

```bash
pip install "specific-ai-tools[split]"

.venv/bin/python scripts/split_model.py /path/to/hf-classification-model
# optional: --outtype f32  (otherwise taken from config.json torch_dtype)
# optional: --llamacpp-repo URL  --llamacpp-dir ~/.cache/specific-ai-tools/llama.cpp
```

Writes `pooler_*.npy` / `classifier_*.npy` (for BERT) and `bert-base-only.gguf`
into the model directory.

## Compare transformers vs Lemonade

```bash
.venv/bin/python scripts/compare_models_results.py \\
    specific-AI/email-agent-phishing-detection \\
    path/to/inputs.json \\
    --lemonade-model-name user.specific-ai-phishing-detection
```

Runs each `example` through Hugging Face transformers and
`LemonadeEmbeddingClassifier`, then reports label / probability agreement
(default atol ``0.05``). The HF card must already be split.

## Custom embedding backends

Subclass `EmbeddingClassifier` and implement `get_embeddings`:

```python
from specific_ai_tools.embedding_heads import EmbeddingClassifier
import numpy as np

class MyClassifier(EmbeddingClassifier):
    def get_embeddings(
        self,
        input_ids_batch: list[list[int]],
        texts: list[str],
    ) -> np.ndarray:
        # Use input_ids_batch and/or texts; return shape (batch, hidden_size)
        ...
```

## Head weights

For each model the package looks for:

1. Pre-split NumPy files (`pooler_w.npy`, `pooler_b.npy`, `classifier_w.npy`,
   `classifier_b.npy`) next to `config.json`, **or**
2. Head tensors inside `model.safetensors` (keys depend on architecture strategy).

`config.json` drives labels and problem type (`single_label_classification` /
`multi_label_classification`). Optional calibration fields may be supplied via
`metadata.json` or constructor overrides (`calibration_temperature`,
`selected_thresholds`, `rejection_label_name`, `confidence_rejection_enabled`).

## Prediction shape

```python
prediction.predicted_labels       # selected labels
prediction.predicted_confidences  # confidences for selected labels only
prediction.all_confidences        # confidences for every class
```

## Supported architectures (v0.1)

| Architecture | Strategy | Status |
|---|---|---|
| BERT (`BertForSequenceClassification`) | pooler dense + tanh → classifier | Supported |
| DistilBERT | pre_classifier (ReLU) → classifier | Planned |
| RoBERTa | — | Out of scope (no GGUF support) |
