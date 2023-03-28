![alt text](https://huggingface.co/blog/assets/86_bloom_megatron_deepspeed/bloom-banner.png)
## 🌲🤏 BLOOM-LoRA: Low-Rank Adaptation for InstructorDoctor-200Kd ataset.

**We try to reimplement BLOOM-LoRA (much less restricted BLOOM license here https://huggingface.co/spaces/bigscience/license) using Chatdoctor-LoRA with this [**InstructorDoctor-200k dataset**](https://drive.google.com/file/d/1lyfqIwlLSClhgrCutWuEe_IACNq6XNUt/view?usp=sharing) "MedDialog: a large-scale medical dialogue dataset" or this [**InstructorDoctor-5k dataset**](https://drive.google.com/file/d/1nDTKZ3wZbZWTkFMBkxlamrzbNz0frugg/view?usp=sharing) following a paper, namely [**ChatDoctor: A Medical Chat Model Fine-tuned on LLaMA Model using Medical Domain Knowledge**](https://arxiv.org/pdf/2303.14070.pdf)**

**Why do we try to finetune these BLOOM models? Because the BLOOM licence seems to be more relax with [The BigScience RAIL License](https://bigscience.huggingface.co/blog/the-bigscience-rail-license)! Moreover, BLOOM models were trained on the dataset having [59 Languages (46 natural and 13 programing languages](https://huggingface.co/bigscience/bloom) including [2.7% Vietnamese (7^th in total 46 natural languages)](https://huggingface.co/bigscience/bloom)**

**For example, you can try our finetuned BLOOMZ-b71-mt-chatdoctor-200k (`LinhDuong/doctorwithbloom`) model out on Colab [here](https://colab.research.google.com/drive/1LY5Ds6qyr_Drpp9WSdt-ZEMvvrFICdEx#scrollTo=X_pz8MuY84Qh)!!!!!**

We try to reimplement BLOOM-LoRA using a variety of sources such as [the original LLaMA](https://github.com/facebookresearch/llama), [Stanford-Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [Alpaca-LoRA](https://github.com/tloen/alpaca-lora), [BLOOMZ](https://github.com/NouamaneTazi/bloomz.cpp), and a name to few. These datasets for finetuning tasks can be found at [the original source](https://drive.google.com/file/d/1lyfqIwlLSClhgrCutWuEe_IACNq6XNUt/view?usp=sharing) or [my HuggingFace Hub](https://huggingface.co/LinhDuong).

In addition to the training code, which runs on a single RTX 4090 takes a week or runs on 8 GPUs A100 takes around 8 hours using the InstructorDoctor-200k dataset. On the other hand, you can use the InstructorDoctor-5k dataset and run on a single RTX 3090/4090, it takes overnight.
We now publish a script for downloading and inference on the foundation model and LoRA,
as well as the resulting [LoRA weights themselves](https://huggingface.co/LinhDuong/doctorwithbloom/tree/main).
To fine-tune cheaply and efficiently, we use Hugging Face's [PEFT](https://github.com/huggingface/peft)
as well as Tim Dettmers' [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).

Without hyperparameter tuning, the LoRA model produces outputs comparable to the Stanford Alpaca model. (Please see the outputs included below.) Further tuning might be able to achieve better performance; I invite interested users to give it a try and report their results.

## Setup

1. Install dependencies

    ```bash
    pip install -r requirements.txt
    ```

1. Set environment variables, or modify the files referencing `BASE_MODEL`:

    ```bash
    # Files referencing `BASE_MODEL`
    # export_hf_checkpoint.py
    # export_state_dict_checkpoint.py
    export BASE_MODEL=bigscience/7b1
    ```

    Both `finetune.py` and `generate.py` use `--base_model` flag as shown further below.

1. If bitsandbytes doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).

### Training (`finetune.py`)

This file contains a straightforward application of PEFT to the Bloomz-7b1 model,
as well as some code related to prompt construction and tokenization.
PRs adapting this code to support larger models are always welcome.

Example usage:

```bash
python finetune.py \
    --base_model 'bigscience/bloomz-7b1' \
    --data_path 'LinhDuong/chatdoctor-200k' \
    --output_dir './lora-chatdoctor-200k'
```

We can also tweak our hyperparameters:

```bash
python finetune.py \
    --base_model 'bigscience/bloomz-7b1' \
    --data_path 'LinhDuong/chatdoctor-200k' \
    --output_dir './lora-chatdoctor-200k' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length
```
If you want to finetune with multi-GPU, for example with 8 GPUs A100:

```bash
WORLD_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
torchrun --nproc_per_node=8 --master_port=1234 
    finetune.py \
    --base_model 'bigscience/bloomz-7b1' \
    --data_path 'LinhDuong/chatdoctor-200k' \
    --output_dir './lora-chatdoctor-200k' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length
```

### Inference (`generate.py`)

This file reads the foundation model from the Hugging Face model hub and the LoRA weights from `tloen/alpaca-lora-7b`, and runs a Gradio interface for inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.

Example usage:

```bash
python generate.py \
    --load_8bit \
    --base_model 'bigscience/bloomz-7b1' \
    --lora_weights 'LinhDuong/doctorwithbloom'
```

### Or you can build a ChatDoctor model on your own machine and communicate with it.
 
 ```python
python chat.py
 ```

### Checkpoint export (`export_*_checkpoint.py`)

These files contain scripts that merge the LoRA weights back into the base model
for export to Hugging Face format and to PyTorch `state_dicts`.
They should help users
who want to run inference in projects like [llama.cpp](https://github.com/ggerganov/llama.cpp)
or [alpaca.cpp](https://github.com/antimatter15/alpaca.cpp).

### Notes

- We can likely improve our model performance significantly if we had a better dataset. Consider supporting the [LAION Open Assistant](https://open-assistant.io/) effort to produce a high-quality dataset for supervised fine-tuning (or bugging them to release their data).
- We're continually fixing bugs and conducting training runs, and the weights on the Hugging Face Hub are being updated accordingly. In particular, those facing issues with response lengths should make sure that they have the latest version of the weights and code.
- Users with multiple GPUs should take a look [here](https://github.com/tloen/alpaca-lora/issues/8#issuecomment-1477490259).

 
 
 ## Introduction
The development of instruction-following large language models (LLMs) such as ChatGPT has garnered significant attention due to their remarkable success in instruction understanding and human-like response generation.
These auto-regressive LLMs are pre-trained over web-scale natural languages by predicting the next token and then fine-tuned to follow large-scale human instructions.
Also, they have shown strong performances over a wide range of NLP tasks and generalizations to unseen tasks, demonstrating their potential as a unified solution for various problems such as natural language understanding, text generation, and conversational AI.
However, the exploration of such general-domain LLMs in the medical field remains relatively untapped, despite the immense potential they hold for transforming healthcare communication and decision-making.
The specific reason is that the existing models do not learn the medical field in detail, resulting in the models often giving wrong diagnoses and wrong medical advice when playing the role of a doctor. By fine-tuning the large language dialogue model on the data of doctor-patient conversations, the application of the model in the medical field can be significantly improved. Especially in areas where medical resources are scarce, ChatDoctor can be used for initial diagnosis and triage of patients, significantly improving the operational efficiency of existing hospitals.

Since large language models such as ChatGPT are in a non-open source state, they used Meta's LLaMA and first trained a generic conversation model using 52K instruction-following data provided by Stanford Alpaca, and then fine-tuned the model on their collected physician-patient conversation dataset.
The main contributions of our method are three-fold:
1) They designed a process framework for fine-tuning large language models in the medical domain.
2) They collected a training data with 5,000 doctor-patient conversations for fine-tuning the large language model.
3) They validate that the fine-tuned bigrams with medical domain knowledge have real potential for clinical application.
 
 ## Physician and patient conversation dataset</h2>
The first step in building a physician-patient conversation dataset is to collect the disease database that serves as the gold standard. Therefore, they collected and organized a database of diseases, which contains about 700 diseases with their relative symptoms, medical tests, and recommended medications. To train high-quality conversation models on an academic budget, we input each message from the disease database separately as a prompt into the ChatGPT API to automatically generate instruction data. It is worth noting that their prompts to the ChatGPT API contain the gold standard of diseases and symptoms, and drugs, so our fine-tuned ChatDoctor is not only able to achieve ChatGPT's conversational fluency but also higher diagnostic accuracy compared to ChatGPT. They finally collected 5K doctor-patient conversation instructions and named it InstructorDoctor-5K.



## Limitations
We emphasize that ChatDoctor is for academic research only and any commercial use and clinical use is prohibited. There are three factors in this decision: First, ChatDoctor is based on LLaMA and has a non-commercial license, so we necessarily inherited this decision. Second, our model is not licensed for healthcare-related purposes. Also, we have not designed sufficient security measures, and the current model still does not guarantee the full correctness of medical diagnoses.





### Example outputs

**Instruction**: 

**DOCTORWITHBLOOM**: 

**BLOOM-7b1**: 

**ChatGPT (free version dated March 25^th 2023)**:

---

**Instruction**: 

**DOCTORWITHBLOOM**: 

**BLOOM-7b1**: 

**ChatGPT (free version dated March 25^th 2023)**:

---

**Instruction**: 

**DOCTORWITHBLOOM**: 

**BLOOM-7b1**: 

**ChatGPT (free version dated March 25^th 2023)**:

---


**Instruction**: 

**DOCTORWITHBLOOM**: 

**BLOOM-7b1**: 

**ChatGPT (free version dated March 25^th 2023)**:

---


**Instruction**: 

**DOCTORWITHBLOOM**: 

**BLOOM-7b1**: 

**ChatGPT (free version dated March 25^th 2023)**:

---


**Instruction**: 

**DOCTORWITHBLOOM**: 

**BLOOM-7b1**: 

**ChatGPT (free version dated March 25^th 2023)**:

---


**Instruction**: 

**DOCTORWITHBLOOM**: 

**BLOOM-7b1**: 

**ChatGPT (free version dated March 25^th 2023)**:

---


**Instruction**: 

**DOCTORWITHBLOOM**: 

**BLOOM-7b1**: 

**ChatGPT (free version dated March 25^th 2023)**:

---


**Instruction**: 

**DOCTORWITHBLOOM**: 

**BLOOM-7b1**: 

**ChatGPT (free version dated March 25^th 2023)**:
