# godot-docs-to-csv

Converts the Godot Engine HTML documentation into an Alpaca-style CSV dataset
(`instruction,input,output,category,source`) for fine-tuning LLMs (Qwen, Gemma, Llama, ...) with Unsloth.

## Build and run

```bash
mvn package
java -jar target/godot-docs-to-csv.jar <docsDir> <output.csv> [--max-chars 4000]
```

`docsDir` is the extracted docs folder (the one containing `classes/`, `tutorials/`, ...).

## What it generates

| Source pages | Rows |
| --- | --- |
| `classes/class_*.html` | One per class overview, property, method, signal, enum, constant, constructor, operator, theme property and annotation, plus "what properties/methods does X have" lists. Undocumented members are skipped. |
| Everything else (tutorials, getting started, ...) | One per page section, with the section heading turned into a question. |

Outputs are Markdown: inline code in backticks, GDScript/C#/GLSL samples in fenced code blocks.
Outputs longer than `--max-chars` are split into `(part i/n)` rows without breaking code blocks.
`input` is always empty (kept for Alpaca compatibility).

## Loading in Unsloth

Pass `keep_default_na=False`, otherwise the empty `input` column is read as `None`.

```python
from datasets import load_dataset

dataset = load_dataset("csv", data_files="godot_docs_dataset.csv", split="train", keep_default_na=False)

def to_chat(batch):
    texts = []
    for question, answer in zip(batch["instruction"], batch["output"]):
        messages = [{"role": "user", "content": question},
                    {"role": "assistant", "content": answer}]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return {"text": texts}

dataset = dataset.map(to_chat, batched=True)  # then train with SFTTrainer on the "text" field
```

The dataset generated from the Godot 4.7 docs (24,631 rows) is published on Hugging Face as
[arunsrajan/godot-docs-dataset](https://huggingface.co/datasets/arunsrajan/godot-docs-dataset).

## Colab notebooks

| Notebook | What it does | GPU |
| --- | --- | --- |
| [Docs assistant](notebooks/Qwen3_5_2B_Godot_Docs_Tool_Calling.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arunsrajan/godot-docs-to-csv/blob/main/notebooks/Qwen3_5_2B_Godot_Docs_Tool_Calling.ipynb) | Gives Qwen3.5-2B a `search_godot_docs` tool over the dataset. `ask_with_docs()` always searches before answering. Optionally fine-tunes the model with LoRA on tool-calling conversations built from the dataset. | Free T4 |
| [Coding assistant](notebooks/Qwen3_5_Godot_Coding_Assistant.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arunsrajan/godot-docs-to-csv/blob/main/notebooks/Qwen3_5_Godot_Coding_Assistant.ipynb) | Qwen3.5-4B in thinking mode with `search_godot_docs` and `check_gdscript`, which checks the model's GDScript with Godot's own parser (`godot --headless --check-only`) so it can fix errors before answering. No training. | Free T4 |
| [Generate reasoning examples](notebooks/Qwen3_5_9B_Generate_Reasoning_Examples.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arunsrajan/godot-docs-to-csv/blob/main/notebooks/Qwen3_5_9B_Generate_Reasoning_Examples.ipynb) | Step 1 of 2. Qwen3.5-9B (with vLLM) solves coding tasks and API questions in thinking mode with both tools. Only examples whose final GDScript passes Godot's checker are kept and pushed to a private Hugging Face dataset. | A100, or L4 with FP8 |
| [Thinking fine-tune](notebooks/Qwen3_5_2B_Thinking_Finetune.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arunsrajan/godot-docs-to-csv/blob/main/notebooks/Qwen3_5_2B_Thinking_Finetune.ipynb) | Step 2 of 2. Fine-tunes Qwen3.5-2B on those examples (80% reasoning) and measures, before and after, whether it finishes thinking, uses the tools and writes GDScript that passes the checker. | Free T4 |

Where to start:

- **Godot API questions:** the docs assistant. Try `ask_with_docs()` before training: searching first fixes most
  answers a small model gets wrong from memory.
- **Writing GDScript:** the coding assistant.
- **A small model that reasons and checks its own code:** the two thinking notebooks, in order.

## License

MIT, see [LICENSE](LICENSE). The generated dataset is derived from the Godot documentation, which is licensed under CC BY 3.0.
