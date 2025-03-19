# GCCL-Medical-LLM2-FlowerTune

---

### Evaluation for Medical challenge (Result of 10th checkpoint out of 10 total rounds)

|        | PubMedQA | MedMCQA | MedQA |  Avg  |
| :-----: | :------: | :-----: | :---: | :---: |
| Acc (%) |  65.80  |  60.38  | 68.57 | 64.91 |

#### Changes from baseline

**- We add formatter on previous version ([GCCL-Medical-LLM-Flowertune](https://github.com/gachon-CCLab/GCCL-Medical-LLM-FlowerTune))**

**- Formatter has function to add Llama 3 Style.**

<pyproject.toml>

`model.name = "ContactDoctor/Bio-Medical-Llama-3-8B"`

`num-server-rounds = 10`

<dataset.py>

```
def formatting_prompts_func(example):
    """Llama 3 Style"""
    output_texts = []
    sys_prompt = "You are an expert trained on healthcare and biomedical reasoning."

    for i in range(len(example["instruction"])):
        text = (
            f"<s>[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n"
            f"{example['instruction'][i]}\n[/INST]\n"
            f"{example['response'][i]} </s>"
        )
        output_texts.append(text)
    return output_texts
```

#### Evaluation Command

```
python eval.py \
--base-model-name-path=ContactDoctor/Bio-Medical-Llama-3-8B \
--peft-path= \ # PEFT PATH - Checkpoint 10
--run-name= \ # RUN NAME
--batch-size=16 \
--quantization=4 \
--datasets=pubmedqa,medmcqa,medqa
```

#### Path to check evaluation results

```
GCCL-Medical-LLM2-FlowerTune/flowertune-eval-medical/benchmarks/

# GCCL_medical_peft10
```

#### Checkpoint Download

[Link](https://drive.google.com/drive/folders/1Nley5gPpxvtD-eLt8nH4SSLCf_Ap2jHx?usp=sharing)
