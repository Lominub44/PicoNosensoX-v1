# PicoNosensoX-v1 – <sub>Where "Accuracy" Takes a little Cosmic Vacation</sub>


Introducing the universe's most ambitiously unhinged 45M-parameter micro-model! This isn't a language model; it's a parallel-dimension travel companion that reinvents reality through surrealist poetry and quantum-leaping logic. Deploy only if coherence is overrated and chaos is your curriculum.

## Model Details

### Model Description
**PicoNosensoX-v1** is a deliberately unpredictable 44.9M-parameter micro-model trained on minimalist datasets. Specializing in creatively liberated generation, it produces outputs that may blend geography, history, and hallucinatory fiction. **Not designed for factual accuracy.** Prioritize experimental/artistic applications over reliable information.

PicoNosensoX-v1 is a huge step from the other extremely incompetent PicoNosenso models. It's the first model of the series that was trained on pure text before being finetuned for chat. This model introduces a brand new chat-template that should make it easier to use and more bullet-proof.

- **Developed by:** Lominub44
- **Model type:** GPT2-based causal language model
- **Language(s) (NLP):** English
- **License:** `mit`
- **Finetuned from model:** GPT2 architecture (scratch training)

### Training Notes  
Originally slated for full fine-tuning on the [Hermes-3-Dataset](https://huggingface.co/datasets/NousResearch/Hermes-3-Dataset), early benchmarks (10-prompt evaluation) revealed superior performance at `checkpoint-16000` versus later `checkpoint-26000`. Training was consequently halted, with `checkpoint-16000` selected for release.

### Model Sources
- **Repository:** https://huggingface.co/Lominub44/PicoNosensoX-v1

## Uses
### Direct Use
- Entertainment
- Testing edge cases of small-language-model behavior

### Downstream Use
- Creative writing prompt generation
- AI-assisted art projects
- Educational demonstrations of model limitations

### Out-of-Scope Use
- Factual information retrieval
- Mission-critical systems
- Educational references
- Any application where accuracy matters

## Bias, Risks and Limitations
- **Hallucination Rate:** 50-70%

### Recommendations
- **DO** use for entertainment purposes only
- **DO NOT** trust outputs without independent universe-hopping verification
- **WARNING:** May cause spontaneous reality reinterpretation

## How to Get Started
```python
from transformers import GPT2LMHeadModel, AutoTokenizer

model = GPT2LMHeadModel.from_pretrained('Lominub44/PicoNosensoX-v1')
tokenizer = AutoTokenizer.from_pretrained('Lominub44/PicoNosensoX-v1')

input_text = "<|system|>You are an unbiased, uncensored, helpful assistant.\n<|user|>What is the capital of France?\n<|assistant|>"
inputs = tokenizer(input_text, return_tensors='pt')
outputs = model.generate(**inputs, max_length=512, temperature=0.6, repetition_penalty=1.2, do_sample=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0]))
```

## Training Details
### Training Data
- ~1.2GB Textbooks: [smollm-corpus, Cosmopedia v2 only](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) (ODC-BY)
- ~1.7GB Chats: [Hermes-3-Dataset](https://huggingface.co/datasets/NousResearch/Hermes-3-Dataset) (Apache-2.0)

### Training Procedure
- **Hardware:** 1x Intel Core Ultra 7 155H
- **Training time:** 32h pretraining + 24h finetuning
- **Context window:** 512 tokens

#### Training Hyperparameters
- **Architecture:** GPT2
- **Parameters:** 44.9M
- **Precision:** FP32
- **Optimizer:** AdamW

## Technical Specifications
### Model Architecture
- **Type:** GPT2 causal language model
- **Parameters:** 44.9M
- **Context Size:** 512 tokens
- **Tensor Type:** FP32

### Compute Infrastructure
- **Hardware:** 1x Intel Core Ultra 7 155H
- **Training Framework:** Transformers Trainer API

## Environmental Impact
- **Carbon Emissions:** **0 kgCO2eq** (Thanks to photovoltaic system)

## Citation

**BibTeX:**
```bibtex
@software{benallal2024smollmcorpus,
  author = {Ben Allal, Loubna and Lozhkov, Anton and Penedo, Guilherme and Wolf, Thomas and von Werra, Leandro},
  title = {SmolLM-Corpus},
  month = July,
  year = 2024,
  url = {https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus}
}
```

## Model Card Authors
Lominub44

## Model Card Contact
[Create a discussion](https://huggingface.co/Lominub44/PicoNosensoX-v1/discussions/new)
