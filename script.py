
import torch
import os
import shutil
import pandas as pd
from transformers import BertTokenizer

# Load the saved full model and tokenizer
model_path = 'model.pt'  
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the entire model (architecture + weights)
model = torch.load(model_path)

# Set the model to evaluation mode
model.eval()

# Function to predict resume category
def predict_resume_category(resume_text):
    inputs = tokenizer(resume_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return predicted_class_id


#  category labels
category_labels=['HR', 'DESIGNER', 'INFORMATION-TECHNOLOGY', 'TEACHER', 'ADVOCATE',
       'BUSINESS-DEVELOPMENT', 'HEALTHCARE', 'FITNESS', 'AGRICULTURE',
       'BPO', 'SALES', 'CONSULTANT', 'DIGITAL-MEDIA', 'AUTOMOBILE',
       'CHEF', 'FINANCE', 'APPAREL', 'ENGINEERING', 'ACCOUNTANT',
       'CONSTRUCTION', 'PUBLIC-RELATIONS', 'BANKING', 'ARTS', 'AVIATION']

# Script to categorize resumes
def categorize_resumes(directory):
    resume_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]  
    categorized_data = []
    
    for resume in resume_files:
        resume_path = os.path.join(directory, resume)
        resume_text = extract_text_from_pdf(resume_path)  
        
        predicted_category = predict_resume_category(resume_text)
        category_name = category_labels[predicted_category]
        
        # writable directory for categorized resumes
        make_dir = os.path.join('categorized_resumes', category_name)
        os.makedirs(make_dir, exist_ok=True)
        
        # resume to the predicted category folder 
        shutil.copy(resume_path, os.path.join(make_dir, resume))
        
        # Append to categorized data for CSV
        categorized_data.append({"filename": resume, "category": category_name})
    
    # Save resulT
    categorized_resumes_df = pd.DataFrame(categorized_data)
    categorized_resumes_df.to_csv('categorized_resumes.csv', index=False)


# PDF text extraction 
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# MAIN 
if __name__ == "__main__":
    resume_directory = sys.argv[1]
    categorize_resumes(resume_directory)
