# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

# Load the skills dataset
skills_df = pd.read_csv('skills.csv')

# Load the job description dataset
job_description_df = pd.read_csv('job_title_des.csv')

def clean_text(text):
    cleaned_text = re.sub(r'http\S+\s', ' ', text)
    cleaned_text = re.sub(r'RT|cc', ' ', cleaned_text)
    cleaned_text = re.sub(r'#\S+\s', ' ', cleaned_text)
    cleaned_text = re.sub(r'@\S+', '  ', cleaned_text)  
    cleaned_text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleaned_text)
    cleaned_text = re.sub(r'[^\x00-\x7f]', ' ', cleaned_text) 
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text

def extract_skills(text, skills_list):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in word_tokens if word not in stop_words and word.isalnum()]
    
    skills_list_lower = set([skill.lower() for skill in skills_list])
    
    skills = set([word for word in filtered_tokens if word.lower() in skills_list_lower])
    
    return skills

clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = LabelEncoder()

def main():
    st.title("Resume Analyzer")
    st.sidebar.header("Upload Files")
    
    resume_file = st.sidebar.file_uploader("Upload Resume", type=['txt', 'pdf'])
    job_description_file = st.sidebar.file_uploader("Upload Job Description", type=['txt', 'pdf'])

    if resume_file is not None and job_description_file is not None:
        resume_text = resume_file.read().decode('utf-8')
        job_description_text = job_description_file.read().decode('utf-8')

        cleaned_resume = clean_text(resume_text)
        cleaned_job_description = clean_text(job_description_text)

        resume_skills = extract_skills(cleaned_resume, skills_df['Skill'])

        job_description_skills = extract_skills(cleaned_job_description, skills_df['Skill'])

        clf = pickle.load(open('clf.pkl', 'rb'))
        tfidf = pickle.load(open('tfidf.pkl', 'rb'))

        df = pd.read_csv('UpdatedResumeDataSet.csv')

        le = LabelEncoder()
        le.fit(df['Category'])

        input_features = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        category_names = le.inverse_transform(clf.classes_)
        category_mapping = {idx: category_name for idx, category_name in enumerate(category_names)}
        predicted_category = category_mapping.get(prediction_id, "Unknown")

        st.subheader("Resume Category Prediction")
        st.write("Predicted Category:", predicted_category)

        st.subheader("Extracted Skills from Resume")
        st.write(resume_skills)

        st.subheader("Extracted Skills from Job Description")
        st.write(job_description_skills)

        jaccard_similarity = len(resume_skills.intersection(job_description_skills)) / len(resume_skills.union(job_description_skills))
        similarity_percentage = jaccard_similarity * 100
        st.subheader("Similarity Percentage between Resume and Job Description")
        st.write(similarity_percentage)

if __name__ == '__main__':
    main()
