Layotus :
- [Introduction](#introduction)
- [Preprocessing](#preprocessing)
- [Model Selection](#model-selection)
- [Instruction](#instruction)





## Introduction: 

This problem aims to categorize resumes automatically. The model is trained based on a CSV file comprised of features namely resume_str, resume_html, and the target column (category). After careful preprocessing, I leveraged the state-of-the-art text classification model(BERT model) in training. Further, the model achieved quite good performance after fine-tuning. The model held 80 percent accuracy, 80 percent recall on the validation set. A python file **script.py** is created to run from any local machine, regardless of the availability of GPU.


## Preprocessing:

There are total of four columns in the dataset namely **ID**, **Resume_str**, **Resume_html**, and **Category**. There were no missing values in any of these columns. There were a total of 24 different categories in the target column. I encoded this **Category** column into integer values. The Resume_str column and Resume_html could hold string values. However, I dropped the latter, considering only the former as the best feature for this classification problem. 

**Reason for omitting Resume_html**: The HTML tags are representative of the design and structure. However, the design and structure of the resume don't give semantic meaning to the content. Therefore considering this as a feature may introduce unnecessary noise for the BERT model. Therefore I omit it, considering that the text content in the Resume_str's pdf file is best for training. 

**Tokenization**: Tokenization is one of the core challenges in dealing with text data. I applied BERT's tokenizer which uses a wordpiece tokenizer on the Resume_str column. I applied this tokenization process as BERT's tokenizer (Word piece) helps to handle out-of-vocabulary words allowing the model to understand different words perfectly. Moreover, the attention mask helps the model to identify which is the actual content and which is padded. Further, the maximum capacity for BERT is 512 tokens. Sequences longer or shorter are truncated and padded respectively which helps prevent the model during model input.

**Split**: I split the dataset into three parts namely train, val, and test. Before applying tokenization, I transformed these datasets into a hugging face dataset for better tokenization. 


## Model Selection: 

Once the dataset was prepared I applied a BERT-base-uncased model on it. This is a pre-trained model from the Hugging Face Transformers library. BERT is best known for its effectiveness in dealing with different NLP tasks as it's capable of capturing contextual relationships in text.

**Reason**: The reason for selecting this model is that this model has previous records showing state-of-the-art performance on in-text classification tasks. As the problem was about categorization I considered it as my model. Furthermore, the uncased version is helpful in dealing with text regardless of case sensitivity, which is particularly helpful for this dataset of resumes. Apart from these reasons- 

   - BERT is capable of understanding the context of a word based on both left and right context in a sentence.
   - As it has trained on big corpus like Wikipedia it can understand grammar and common language patterns. Further fine-tuning allowed my task more efficient.
   - It's capable of handling long texts. As resumes usually contain a lot of texts, making BERT as an ideal choice for this problem.

**Fine-Tuning**: As the BERT model is trained on big corpus like Wikipedia, the prestrained weights are not best suited for this downstream classification problem. Therefore, I trained the model with custom hyperparameters to enhance the model's
 performance. The notable model parameters that I considered are  - five epochs, learning rate 2e-5, batch size of 16, and weight decay of .01. 

 **Evaluation metrics** Moreover, I used accuracy, F1-score, precision, and recall metrics to evaluate the model's performance in each evaluation step.



 ## Instruction 

 I have trained the model in Kaggle using GPU T4X2 for faster training. However, I provided a script.py [script.py](#scriptpy-) file which will help you to get predictions. Because The model architecture, weights, and configurations remain the same regardless of whether the model is trained on a GPU or CPU. Hence, load the trained model on a CPU for inference or further processing without any issues. 

 However, you need to have certain libraries like torch, transformers and so which are mentioned in the requirements.txt. To make inferences on your local machine do the following.

   - first download the model.pt and script.py and Requirements.txt file
     
   - install the required dependencies by running the following command.

         - pip install -r Requirements.txt

   - run the script file using the following command.

         - python script.py /path to folder

     - Note: /path to folder : the folder path in your local machine which contains resumes in pdf format. Keep all three downloaded files (model.pt, script.py Requirements.txt, and the folder of resumes)
       in the same folder folder for convenience.






 ## script.py : 
   - This file takes a folder as input which contains resumes in the pdf format, then makes prediction on each of the resumes and store the predictions in two ways. One is a CSV file which contains two columns - id and  Category, the other is a folder where the resumes are placed in different folders regarding their categories. For example, if a resume is of type accountant, this script.py file will create a folder by this name and place this pdf in that folder.
