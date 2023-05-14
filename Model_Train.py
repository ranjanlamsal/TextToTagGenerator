#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# #Working On sample Datset

# In[ ]:


from google.colab import drive
drive.mount('/content/drive/')


# In[ ]:


get_ipython().system('pip install wandb')


# In[ ]:





# In[ ]:


import pandas as pd
get_ipython().system('pip install datasets')
get_ipython().system('pip install transformers')
get_ipython().system('pip install torch')
get_ipython().system('pip install rouge_score')


# ## Reading Real Dataset To create sample dataset
# 

# ### Creating dataset using sample dataset as training data

# In[ ]:


import pandas as pd

data = pd.read_csv('sample_dataset.csv')
train = data.iloc[1:81]
test = data.iloc[81:91]
val = data.iloc[91:101]

print(len(train['text'].iloc[5]))
print(train['text'].iloc[5])
print(train['tags'][5])
# train.to_csv('sampletrain.csv')
# test.to_csv('sample_test.csv')
# val.to_csv('sample_val.csv')


# In[ ]:


import pandas as pd
from datasets import load_dataset, load_metric
raw_datasets = dataset = load_dataset('csv', data_files={
    'train': 'sampletrain.csv',
    'test': 'sample_test.csv',
    'eval_dataset': 'sample_val.csv'
})


# In[ ]:


metric = load_metric("rouge")


# In[ ]:


raw_datasets


# In[ ]:


from sklearn.model_selection import train_test_split as tts


# In[ ]:


model_checkpoint = "t5-small"


# In[ ]:


from transformers import AutoTokenizer
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# In[ ]:


max_input_length = 1024
max_target_length = 128

def preprocess_function(examples):
    inputs = [doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["tags"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# In[ ]:


preprocess_function(raw_datasets['train'][:2])


# In[ ]:


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)


# In[ ]:


from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


# ###Defining Model Arguments

# In[ ]:


batch_size = 8
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"TextToTagGeneratorSample",
    evaluation_strategy='steps',
    eval_steps=5,
    save_steps=5,
    save_strategy="steps",
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    logging_dir='./logs',
    logging_steps=5,
#     fp16=True,
    predict_with_generate=True,
    push_to_hub=True,
)


# ###Hugging Face Notebook login 
# Token ("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")

# In[ ]:


from huggingface_hub import notebook_login

token="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"
get_ipython().system('huggingface-cli login --token $token')


# In[ ]:


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# In[ ]:


import nltk
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


# ###Seq2SeqTrainer

# In[ ]:


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset =tokenized_datasets["eval_dataset"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# In[ ]:


import gc
import torch
gc.collect()
torch.cuda.empty_cache()


# ###Calling Training Function

# In[ ]:


import nltk
nltk.download('punkt')


# In[ ]:


trainer.train()


# # Splitting dataset on smaller chuncks

# In[55]:


#Train data indexing : 0-7694(first 7694 data done)*, 7694 - 15389 (second 7694 done) , upto 38472 and 4812 done take 7694 and 7695 alternately
#test val data indexing:  0 - 962 (first 962 data done)* , 962- 1924 (second 962 done), take 61 and 62 alternately
import pandas as pd
train_data = pd.read_csv('medium_articles_datasets/train_df.csv').iloc[38472: 46573 ]
test_data =  pd.read_csv('medium_articles_datasets/test_df.csv').iloc[4812: 5825]
val_data =  pd.read_csv('medium_articles_datasets/val_df.csv').iloc[4812: 5825]

max_text_length = 1024
for text in train_data["text"]:
    if len(text) > max_text_length:
        text =text[:max_text_length]


# In[56]:


train_data.to_csv('medium_articles_datasets/splitted_data/train6.csv')
test_data.to_csv('medium_articles_datasets/splitted_data/test6.csv')
val_data.to_csv('medium_articles_datasets/splitted_data/val6.csv')


# In[57]:


print(train_data.shape)
print(test_data.shape)
print(val_data.shape)


# In[ ]:





# # Working with Real Dataset

# In[58]:


import gc
import torch
gc.collect()
torch.cuda.empty_cache()


# ### wandb key:
# 7b5d8de237dfc61b1e63cc8a4fcdbad40d684ddc

# ###Definig the Modle To Train "T5"

# In[59]:


model_checkpoint = "Ranjan22/TextToTagGenerator"


# In[60]:


WANDB_INTEGRATION = True
if WANDB_INTEGRATION:
    import wandb

    wandb.login()


# ###Mounting google drive

# In[ ]:





# ## Requirements

# In[7]:


import pandas as pd
get_ipython().system('pip install datasets')
get_ipython().system('pip install transformers')
get_ipython().system('pip install torch')
get_ipython().system('pip install rouge_score')


# ##Reading Real Dataset from drive

# In[ ]:





# In[5]:


import pandas as pd


# In[61]:


import pandas as pd
from datasets import load_dataset, load_metric

raw_datasets = dataset = load_dataset('csv', data_files={
    'train': 'medium_articles_datasets/splitted_data/train6.csv',
    'test': 'medium_articles_datasets/splitted_data/test6.csv',
    'eval': 'medium_articles_datasets/splitted_data/val6.csv',
})


# ### Defining Metric

# In[62]:


metric = load_metric("rouge")


# ### Checking the Dataset

# In[63]:


raw_datasets


# ### Tokenizing

# In[64]:


from transformers import AutoTokenizer
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# ### Preprocess Function

# In[65]:


max_input_length = 512
max_target_length = 128

def preprocess_function(examples):
    inputs = [doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["tags"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# In[42]:


preprocess_function(raw_datasets['train'][:2])


# ### Maping

# In[66]:


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)


# In[67]:


from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


# ## Defining Model Args

# In[68]:


batch_size = 16
model_name = model_checkpoint
args = Seq2SeqTrainingArguments(
    f"TextToTagGenerator",
    evaluation_strategy='steps',
    eval_steps=100,
    save_steps=100,
    save_strategy="steps",
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    logging_dir='./logs',
    logging_steps=100,
#     fp16=True,
    predict_with_generate=True,
    push_to_hub=True,
)


# ### Hugging Face Notebook login 
# Token ("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")

# In[69]:


from huggingface_hub import notebook_login

token="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"
get_ipython().system('huggingface-cli login --token $token')


# ### Data Collator

# In[70]:


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# ### Function : Compute Matric

# In[71]:


import nltk
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


# ### Seq2Seq Trainer

# In[20]:


get_ipython().system('git pull')


# In[72]:


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset =tokenized_datasets["eval"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# ### Calling Trainer Function

# In[ ]:


trainer.train()


# In[ ]:


import os
os.system("shutdown /s /t 1")


# In[ ]:




