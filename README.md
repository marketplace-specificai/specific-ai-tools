# specific-ai-tools

Python toolkit for SpecificAI — edge inference helpers, partner/platform
adapters, and utilities for running Specific AI models outside the platform
(Lemonade, llama.cpp, and more as topics land).

## Install

```bash
pip install specific-ai-tools

# Optional extras
pip install "specific-ai-tools[llamacpp]"  # local GGUF via llama-cpp-python
pip install "specific-ai-tools[split]"     # split_model + compare_models_results (torch)
pip install "specific-ai-tools[all]"       # torch + llama-cpp-python
```

## Packages

| Package | Description |
|---|---|
| [`embedding_heads`](src/specific_ai_tools/embedding_heads/README.md) | Run classification heads on CLS embeddings from Lemonade Server (HTTP) or llama-cpp-python when those layers are stripped by GGUF conversion |

```python
from specific_ai_tools.embedding_heads import LemonadeEmbeddingClassifier

classifier = LemonadeEmbeddingClassifier(
    lemonade_model_name="user.email-agent-triage",
    checkpoint="specific-AI/email-agent-triage:bert-base-only.gguf",
    lemonade_base_url="http://localhost:13305",
)
```

See [`embedding_heads` README](src/specific_ai_tools/embedding_heads/README.md) for
Lemonade and llama-cpp details. More topics will land here as siblings of
`embedding_heads` over time.

## About Us

**[Specific AI](https://specific.ai)** is the automatic SLM distillation platform
that turns task prompts into production-grade small language models in days —
not weeks — so your subject matter experts can ship models without waiting on
scarce data-science bandwidth.

We help enterprises move agentic AI from prototype to production with SLMs that
are typically **1,000×–10,000× smaller** than teacher LLMs, run in
**milliseconds** on CPUs or edge devices, and deliver the same or better
task quality at a fraction of the cost — self-hosted on your cloud or
downloaded for your own inference stack.

**Prompt → Distill → Deploy.** Bring your prompt and data, drop them into
Specific AI, and get a validated small model ready to test and ship.

Ready to create SLMs at scale? Visit **[specific.ai](https://specific.ai)**.

## License

MIT — see [LICENSE](LICENSE).

Copyright (C) 2026 Specific AI Inc. All rights reserved.
