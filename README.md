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
