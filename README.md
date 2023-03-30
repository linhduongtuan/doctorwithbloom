![alt text](https://huggingface.co/blog/assets/86_bloom_megatron_deepspeed/bloom-banner.png)
## 🌲🤏 BLOOM-LoRA: Low-Rank Adaptation for InstructorDoctor-200Kd ataset.

<details><summary>REASONS WHY?</summary>

**We try to reimplement BLOOM-LoRA (much less restricted BLOOM license here https://huggingface.co/spaces/bigscience/license) using Chatdoctor-LoRA with this [**InstructorDoctor-200k dataset**](https://drive.google.com/file/d/1lyfqIwlLSClhgrCutWuEe_IACNq6XNUt/view?usp=sharing) "MedDialog: a large-scale medical dialogue dataset" or this [**InstructorDoctor-5k dataset**](https://drive.google.com/file/d/1nDTKZ3wZbZWTkFMBkxlamrzbNz0frugg/view?usp=sharing) following a paper, namely [**ChatDoctor: A Medical Chat Model Fine-tuned on LLaMA Model using Medical Domain Knowledge**](https://arxiv.org/pdf/2303.14070.pdf)**

**Why do we try to finetune these BLOOM models? Because the BLOOM licence seems to be more relax with [The BigScience RAIL License](https://bigscience.huggingface.co/blog/the-bigscience-rail-license)! Moreover, BLOOM models were trained on the dataset having [59 Languages (46 natural and 13 programing languages](https://huggingface.co/bigscience/bloom) including [2.7% Vietnamese (7^th in total 46 natural languages)](https://huggingface.co/bigscience/bloom)**

**For example, you can try our finetuned BLOOMZ-b71-mt-chatdoctor-200k (`LinhDuong/doctorwithbloom`) model out on A100 , Colab (coming soon)!**


We try to reimplement BLOOM-LoRA using a variety of sources such as [the original LLaMA](https://github.com/facebookresearch/llama), [Stanford-Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [Alpaca-LoRA](https://github.com/tloen/alpaca-lora), [BLOOMZ](https://github.com/NouamaneTazi/bloomz.cpp), and a name to few. These datasets for finetuning tasks can be found at [the original source](https://drive.google.com/file/d/1lyfqIwlLSClhgrCutWuEe_IACNq6XNUt/view?usp=sharing) or [my HuggingFace Hub](https://huggingface.co/LinhDuong).

In addition to the training code, which runs on a single RTX 4090 takes a week or runs on 8 GPUs A100( A100 is provided by takes around 8 hours using the InstructorDoctor-200k dataset. On the other hand, you can use the InstructorDoctor-5k dataset and run on a single RTX 3090/4090, it takes overnight.
We now publish a script for downloading and inference on the foundation model and LoRA,
as well as the resulting [LoRA weights themselves](https://huggingface.co/LinhDuong/doctorwithbloom/tree/main).
To fine-tune cheaply and efficiently, we use Hugging Face's [PEFT](https://github.com/huggingface/peft)
as well as Tim Dettmers' [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).

Without hyperparameter tuning, the LoRA model produces outputs comparable to the Stanford Alpaca model. (Please see the outputs included below.) Further tuning might be able to achieve better performance; I invite interested users to give it a try and report their results.

</details>

<details><summary>HOW TO SETUP?</summary>

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
</details>

<details><summary>HOW TO FINETUNE A MODEL?</summary>
## Training (`finetune.py`)

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
</details>


<details><summary>HOW TO INFERENCE?</summary>
## Inference (`generate.py`)

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
</details>

<details><summary>HOW TO EXPORT A FINETUNED WEIGHT?</summary>

## Checkpoint export (`export_*_checkpoint.py`)

These files contain scripts that merge the LoRA weights back into the base model
for export to Hugging Face format and to PyTorch `state_dicts`.
They should help users
who want to run inference in projects like [llama.cpp](https://github.com/ggerganov/llama.cpp)
or [alpaca.cpp](https://github.com/antimatter15/alpaca.cpp).

</details>

<details><summary>NOTES</summary>


- We can likely improve our model performance significantly if we had a better dataset. Consider supporting the [LAION Open Assistant](https://open-assistant.io/) effort to produce a high-quality dataset for supervised fine-tuning (or bugging them to release their data).
- We're continually fixing bugs and conducting training runs, and the weights on the Hugging Face Hub are being updated accordingly. In particular, those facing issues with response lengths should make sure that they have the latest version of the weights and code.
- Users with multiple GPUs should take a look [here](https://github.com/tloen/alpaca-lora/issues/8#issuecomment-1477490259).
</details>


 <details><summary>ABSTRACT</summary>
 
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

</details>

<details><summary>ACKNOWLEDGMENT</summary>

I would like to express my gratitude to [Phenikaa University](https://phenikaa-uni.edu.vn/en) and [Phenikaa-X](https://phenikaa-x.com/) company, without whose support, this fine-tuned model would not have been possible. Their generosity in providing me with access to their computation resources, namely A100 and RTX 4090, allowed me to train and test my work successfully. 

I would also like to extend my thanks to [Pham Ngoc Ninh](https://github.com/phamngocninh), who worked diligently to address any bugs and refine this model. Ninh's assistance was greatly appreciated.
</details>

<details><summary>EXAMPLE OUTPUTS</summary>


**QUESTION OF PATIENTS**: Doctor, I have been experiencing vomiting, nausea, and
abdominal pain. I recently found out that I have pyloric
stenosis. What medications should I take for it?



**RESPONSE OF  DOCTORWITHBLOOM**: Hi welcome to Chat Doctor. I understand your query and concern. Pyloric stenosis is the narrowing of the pyloric canal, which is the narrowest part of the stomach. It is a congenital abnormality of the gastrointestinal tract. It is usually asymptomatic and is detected incidentally on routine upper GI endoscopy. It is usually treated conservatively. I advise you to consult a gastroenterologist for diagnosis and management. You may need to take proton pump inhibitors like omeprazole or pantoprazole. You may also need to take a course of antibiotics like amoxicillin or metronidazole. Hope I have answered your query. I will be happy to help you further. Wishing you good health. Take care.

<details><summary>RESPONSE OF OTHER METHODS</summary>

**Response of ChatDoctor**: The most common treatment for pyloric stenosis is a surgical procedure called a pyloromyotomy. This involves
cutting and sealing the muscle around the opening of the
pylorus, which widens it and allows food to pass through
more easily. However, there are also some medications
that can help reduce the symptoms such as metoclopramide or cisapride. Which one would you prefer?

**Response of BLOOM-7b1-LoRA-Alpaca**: Pyloric stenosis is a condition in which the pylorus, the narrow passageway between the stomach and the small intestine, is too narrow to allow food to pass smoothly. This can lead to vomiting, nausea, and abdominal pain. Treatment for pyloric stenosis depends on the severity of the condition. If the condition is mild, medications such as antacids, H2 blockers, and proton pump inhibitors can be used to treat the symptoms. If the condition is more severe, surgery may be required to open up the passageway. In some cases, the passageway may need to be surgically enlarged to allow food to pass more easily. Pyloric stenosis is a rare condition, but it is important to seek medical attention if you are experiencing any of the symptoms associated with it. 

**Response of ChatGPT (free version dated March 25^th 2023)**: As an AI language model, I cannot diagnose or prescribe medications. However, pyloric stenosis is a medical condition that occurs when the muscle in the lower part of the stomach (the pylorus) becomes thickened, leading to a narrowing of the opening between the stomach and the small intestine. This narrowing can result in symptoms such as vomiting, nausea, and abdominal pain.

The treatment for pyloric stenosis typically involves surgical intervention, in which the thickened muscle is surgically corrected. If you have been diagnosed with pyloric stenosis, you should discuss the appropriate treatment options with your doctor. They will be able to provide you with information about the risks and benefits of surgical intervention, as well as any medication that may be prescribed to manage your symptoms before or after surgery.
</details>

---


**QUESTION OF PATIENTS**: Hi doctor, I have been losing hair rapidly for the past
few months. I think I might have Alopecia. What should
I do?

**RESPONSE OF DOCTORWITHBLOOM**: Hi, Welcome to Chat Doctor. I am Chat Doctor. I have read your question with care and understand your concerns. I would like to help you in the best possible way. Alopecia areata is a disease in which hair follicles are destroyed. It is a disease of the immune system. There is no cure for this disease. It is a disease of the immune system. 

<details><summary>RESPONSE OF OTHER METHODS</summary>

**Response of ChatDoctor**: There are several medications that can help with Alopecia. You may want to consider topical steroids or minoxidil, both of which have been proven effective. Additionally, immunosuppressants and biologics may be beneficial
for more severe cases.

**Response of BLOOM-7b1-LoRA-Alpaca**: Alopecia is a condition that causes hair loss. It can be caused by a variety of factors, such as genetics, stress, medications, and certain diseases. The most common types of alopecia are androgenic alopecia (male pattern baldness), telogen effluvium (female pattern baldness), and trichotillomania (hair pulling). Alopecia can be treated with a variety of medications, such as minoxidil, finasteride, and triamcinolone, as well as surgical procedures, such as follicular unit transplantation (FUT) and follicular unit extraction (FUE). It is important to consult with a doctor to determine the cause of your hair loss and to determine the best course of treatment for you. In addition, it is important to maintain a healthy lifestyle, such as eating a balanced diet and getting plenty of sleep, to help prevent further hair loss. Finally, it is important to remember that alopecia is not a sign of weakness, but rather a sign of strength. It is a sign that you are taking care of yourself and that you are taking action to improve your health and well-being. By doing so, you will be able to prevent further hair loss and

**Response of ChatGPT (free version dated March 25^th 2023)**: Hair loss can be caused by many different factors, including genetics, hormonal changes, medical conditions, medications, and lifestyle factors. Alopecia is a medical condition that causes hair loss and can occur in different forms. Therefore, I recommend that you see a dermatologist or a healthcare provider who specializes in hair loss to get an accurate diagnosis and the right treatment.

During the appointment, the healthcare provider will ask you questions about your medical history, medications you are taking, and any recent changes in your health. They may also perform a physical exam of your scalp and hair to evaluate the pattern of hair loss and the condition of your scalp.

Based on your diagnosis, your healthcare provider may recommend different treatments, such as topical medications, oral medications, or injections. In some cases, they may recommend hair transplantation surgery. They may also suggest lifestyle changes or counseling to help you manage stress or emotional factors that may be contributing to your hair loss.

It is essential to address hair loss promptly, as some types of hair loss can be irreversible if left untreated. The sooner you seek treatment, the better your chances of regaining your hair.
</details>
---


**QUESTION OF PATIENTS**: Doctor, I’ve been experiencing sudden jerks and twitches
in my muscles. It’s becoming very difficult to manage
them on a daily basis.

**RESPONSE OF DOCTORWITHBLOOM**: Hello, Welcome to Chat Doctor, The symptoms you have mentioned are suggestive of muscle spasm. This can be due to many reasons like stress, anxiety, lack of sleep etc. I would suggest you to consult a physiotherapist and get evaluated. Physiotherapy like ultrasound and TENS therapy will be helpful in relieving the symptoms. Hope this will help you. Take care!

<details><summary>RESPONSE OF OTHER METHODS</summary>
**Response of ChatDoctor**: There are several medications that can help with Myoclonus. Are you taking any other medications?

**Response of BLOOM-7b1-LoRA-Alpaca**: Sudden jerks and twitches in your muscles can be caused by a variety of conditions, including:

1. Muscular dystrophy: Muscular dystrophy is a genetic disease that causes progressive muscle weakness and wasting.

2. Fibromyalgia: Fibromyalgia is a chronic condition characterized by widespread pain and fatigue.

3. Muscle spasm: Muscle spasm is a sudden, involuntary contraction of a muscle.

4. Stress: Stress can cause sudden jerks and twitches in your muscles.

5. Muscle cramps: Muscle cramps are sudden, involuntary contractions of a muscle.

6. Muscle strain: Muscle strain is a type of muscle injury that occurs when a muscle is overworked.

7. Muscle twitch: Muscle twitch is a sudden, involuntary contraction of a muscle.

8. Muscle spasm: Muscle spasm is a sudden, involuntary contraction of a muscle.

9. Muscle cramps: Muscle cramps are sudden, involuntary contractions of a muscle.

10. Muscle strain: Muscle strain is a type of muscle injury that occurs when a muscle is overworked.

11. Muscle twitch

**Response of ChatGPT (free version dated March 25^th 2023)**: Muscle jerks and twitches can have many different causes, ranging from benign conditions like muscle fatigue or dehydration to more serious medical conditions like neurological disorders or autoimmune diseases. Therefore, I would recommend that you see a doctor for a proper evaluation and diagnosis.

During the appointment, your doctor will likely ask you questions about your medical history, medications you are taking, and any recent changes in your health. They may also perform a physical exam and order tests such as blood work or imaging studies to evaluate your condition.

Based on your diagnosis, your doctor may recommend different treatments. For benign causes, such as muscle fatigue or dehydration, your doctor may recommend lifestyle changes like getting enough rest, staying hydrated, and eating a balanced diet.

For more serious causes, such as neurological disorders, your doctor may refer you to a specialist or recommend medications or therapies to manage your symptoms. It's essential to follow your doctor's recommendations and attend any follow-up appointments as needed to monitor your condition.

In the meantime, you can try to manage your muscle jerks and twitches by practicing relaxation techniques, getting enough sleep, and avoiding triggers such as stress or caffeine. If your symptoms are severe or affecting your daily life, you should seek medical attention right away.

</details>
---



**QUESTION OF PATIENTS**: I think I have carbon monoxide poisoning. I feel dizzy
and nauseous

**RESPONSE OF DOCTORWITHBLOOM**: Hello, Welcome to Chat Doctor. I have read your question and understand your concern. Carbon monoxide is a colorless, odorless gas that is produced by incomplete combustion of organic materials such as wood, gasoline, and kerosene. The symptoms of carbon monoxide poisoning include nausea, vomiting, headache, dizziness, weakness, and confusion. If you have any of these symptoms, you should seek immediate medical attention. Hope I have answered your query. Let me know if I can assist you further!

<details><summary>RESPONSE OF OTHER METHODS</summary>

**Response of ChatDoctor**: I recommend that you get medical attention immediately
and start taking a breathing treatment with 100% oxygen. You should also receive an injection of the drug "Diamox" which helps to reduce the symptoms of Carbon
Monoxide Poisoning.

**Response of BLOOM-7b1-LoRA-Alpaca**: I think I have carbon monoxide poisoning. I feel dizzy and nauseous, and my breathing is labored. I have a burning sensation in my throat and my chest feels tight. I have a headache and my vision is blurry. My skin feels cold and clammy. 

**Response of ChatGPT (free version dated March 25^th 2023)**: If you suspect you have carbon monoxide poisoning, it is essential to seek medical attention immediately.

Carbon monoxide is a toxic gas that can be produced by faulty appliances, such as gas stoves, furnaces, or generators. It can cause symptoms like dizziness, nausea, headaches, and fatigue. In severe cases, it can lead to loss of consciousness and even death.

If you are experiencing symptoms of carbon monoxide poisoning, the first thing to do is to get to fresh air immediately. Open windows and doors and turn off any potential sources of carbon monoxide, such as gas stoves or generators.

Next, call your local emergency services or seek medical attention right away. They will be able to evaluate your condition, provide oxygen therapy to help you breathe, and monitor you for any signs of complications.

It is crucial to address carbon monoxide poisoning promptly, as the longer you are exposed to it, the more severe your symptoms can become. In some cases, it can cause permanent damage to the brain and other organs.

If you suspect you have been exposed to carbon monoxide, it is important to take steps to prevent it from happening again. Have your appliances checked regularly, install carbon monoxide detectors in your home, and avoid using gas-powered equipment in enclosed spaces.

</details>
---


**QUESTION OF PATIENTSt**: I have been experiencing memory problems and confusion
lately. I think I might have Wernicke Korsakoff syndrome.

**RESPONSE OF DOCTORWITHBLOOM**: Hello, Welcome to Chat Doctor. I have read your question and understand your concerns. Wernicke Korsakoff syn Chat Doctor!

<details><summary>RESPONSE OF OTHER METHODS</summary>
**Response of ChatDoctor**: Doctor: Yes, we can try some medications such as thiamine and memantine, but it is best to consult with a
neurologist for further evaluation and treatment.

**Response of BLOOM-7b1-LoRA-Alpaca**: Wernicke Korsakoff syndrome (WKS) is a type of neurological disorder that is characterized by memory loss, confusion, and disorientation. It is caused by damage to the brain caused by alcohol abuse or malnutrition. Symptoms of WKS include difficulty remembering new information, difficulty recalling old information, confusion, and disorientation. WKS can be caused by a variety of factors, such as alcohol abuse, malnutrition, or brain injury. It can also be a side effect of certain medications, such as antidepressants and antipsychotics. WKS can be treated with a combination of medications and behavioral therapy. If left untreated, WKS can lead to more serious problems, such as dementia and Alzheimer's disease. It is important to seek medical attention if you are experiencing memory problems and confusion, as it could be a sign of WKS.</s>

**Response of ChatGPT (free version dated March 25^th 2023)**: If you are experiencing memory problems and confusion, it is essential to see a doctor for a proper evaluation and diagnosis. While Wernicke-Korsakoff syndrome is a possible cause of these symptoms, other medical conditions can also cause similar symptoms.

Wernicke-Korsakoff syndrome is a neurological disorder that can occur due to a deficiency of thiamine, a vitamin essential for brain function. It is commonly associated with alcohol abuse, but it can also occur due to malnutrition, gastrointestinal disorders, or other underlying medical conditions.

If you have Wernicke-Korsakoff syndrome, early diagnosis and treatment are crucial to prevent further neurological damage. Treatment typically involves intravenous thiamine to correct the deficiency, followed by a high-dose oral thiamine regimen.

In addition to treatment, lifestyle changes may be recommended to prevent further damage, such as avoiding alcohol and maintaining a healthy diet.

It is important to seek medical attention promptly if you are experiencing memory problems and confusion, as prompt treatment can improve your chances of recovery. Your doctor may perform a physical exam, blood tests, imaging studies, or other tests to evaluate your condition and determine the appropriate treatment plan.
</details>
---
</details>


