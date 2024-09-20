Layotus :
- [Introduction](#introduction)
- [Preprocessing](#preprocessing)
- [Model Selection](#model-selection)
- [Instruction](#instruction)





## Introduction: 

This problem aims to categorize resume automatically. The model is trained on the basis of a csv file comprises of features namely resume_str, resume_html, and the target column (category). After careful preprocessing, I leveraged the state-of-the-art text classification model(BERT model) in training. Further, the model achieved quite good performance after fine tuning. The mdoel held 80 percent accuracy, 80 percent recall on the validaion set. A python file **script.py** is created to run from any local machine, regardless of availablity of gpu.


## Preprocessing:

There are total four colulmns in the dataset namely **ID**, **Resume_str**, **Resume_html** and **Category**. There were no missing values in any of these columns. There were total 24 different categories in the target column. I encoded this **Category** column into integer values. The Resume_str column and Resume_html coulms hold string values. However, I dropped the latter one considering only the former one as the best feature for this classificaton problem. 

**Reason for omitting Resume_html**: The html tags are the representative of the design and structure. However, design and structure of resume don't give semantic meaning of the content. Therefore considering this as feature may introduce unncessary noise for BERT model. Therfore I omit it, considering that the text content in the Resume_str's pdf file is best for training. 

**Tokenization** : Tokenization is one of core challenges in dealing with text data. I applied BERT's tokenizer which uses a wordpiece tokenizer on the Resume_str column. I applied this tokenization process as BERT's tokenizer (Wordpiece) helps to handle out of vocabulary workds allowing the model to understand different words perfectly. Moreover, attention mask helps the model to identify which is actual content and which is padded. Further, the maximum capacity for BERT is 512 tokens. Sequence longer or shorter are truncated and padded respectively which helps prevent the model during model input.

**Split**: I split the dataset into thre parts namely train, val, and test. Before applying tokenization, I transformed these datasets innto hugging face dataset for better tokenization. 


## Model Selection: 

Once the dataset is prepared I applied BERT-base-uncased model on it. This is a pretrained model from Hugging Face Transformers libtrary. BERT is best known for its effectiveness in dealing with different NLP tasks as it's capable of capturing contextual relationship in text.

**Reason** : The reason for selecting this model is that this model has previous records of showing state-of-the-art perfromance on in text calssificaiton task. As the problem was about categorization I considered it as my model. Furthermore, the uncased version is helpful in dealing with text regardless of case sensitivity, which is particulary helpful for this dataset of resumes. Apart from these reasons- 

   - BERT is capable to understand the context of word based on both left and right context in a sentence.
   - As it has pretrained on big corpus like wikipedia it can uderstands grammar, common language patterns. Furthere fine-tuning allowed my task more efficient.
   - It's capable of handling long texts. As resumes ususally contain a lot of texts, making BERT as an ideal choice for this problem.

**Fine-Tuning** : As the BERT model is trained on big corpus like wikipedia, the pretrainedd weights are not best sutied for this downstream classification problem. Therefore, I trained the model with custom hyperparameters to enhance the model's
 performance. The notable model parameters that i considered are  - five epochs, learning rate 2e-5, batch size of 16, , weight deacay of .01. 

 **Evaluation metrics** Moreover, I used accuracy, F1-score, precision, recall metrics to evaluate model's performance in each evaluation step.



 ## Instruction 

 I have trained the model in kaggle using GPU T4X2 for faster training. However, I provided a script.py [script.py](#scriptpy-) file which will help you to get prediction. Because The model architecture, weights, and configurations remain the same regardless of whether the model is trained on a GPU or CPU. Hence, load the trained model on a CPU for inference or further processing without any issues. 

 However, you need to have the certain libraries like torch, transformers and so which are mentioned in the requriements.txt. To make inference on your local machine do the follwoing.

   - at first download the model.pt and script.py file
     
   - install the required dependencies by running the comman follwoing command.

         - pip install -r requirements.txt

   - run the script file using the following command.

         - python script.py /path to folder

     - Note: /path to folder : the folder path in your local machine which contains resumes in pdf format.






 ## script.py : 
   - This file takes a folder as input which contains resumnes in the pdf format, then makes prediction on each of the resumes and store the predictions in two ways. One is a csv file which contains two columns - id and  Category , the other is a folder where the resumes are placed in different folders regarding their categories. For example, if a resume is of type accountant, this script.py file will create a folder by this name and place this pdf to that folder.
