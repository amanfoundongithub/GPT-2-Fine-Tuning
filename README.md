# PEFT on GPT-2 for Summarization Task
---

GPT-2 is a language model designed for the English that was trained on Common Crawl. We applied PEFT to fine tune the model for summarization task

**Dataset Used** : CNN Daily Mail Dataset contains the `articles` and their corresponding `highlights`, which acts as summary.


Training the Models
---
To execute training, you need to download the CNN Daily Mail dataset from Kaggle and store it in the following format: 

![Dataset structure](./dataset_format.png?raw=true "Dataset Structure")


- Train a LoRA model by using following command:
```shell
python3 Train_LoRA.py
```

- Train a Traditional Fine tuned model by using the following command:
```shell 
python3 Train_TradFT.py
```

- Train a Soft Prompt model by using the following command:
```shell 
python3 Train_SoftPrompt.py
```

Restore the Pre-trained Models
---
You can restore the pre trained models by downloading the models from the [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/aman_r_students_iiit_ac_in/EitEuJaZ5WlBuqJ9AKcPG08BIXfpn_LDTwMKdr3ouHWWpA?e=MQtwzY).

Evaluating the Models
---
After running all three scripts (or taking the pre trained models from the link), you can evaluate the models on test dataset by using the following command:

```shell
python3 TestModels.py
```


