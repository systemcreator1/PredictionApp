import streamlit as st
import csv
from collections import defaultdict
import math
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set the style for seaborn
sns.set(style="whitegrid")

# Function to load CSV data
def load_csv(file):
    file.seek(0)  # Move to the beginning of the file
    reader = csv.reader(io.StringIO(file.read().decode("utf-8")))
    headers = next(reader)  # first row is headers
    data = [row for row in reader]
    return headers, data

# Function to calculate prior probabilities for each class in the selected category
def calculate_prior(data, category_index):
    prior_counts = defaultdict(int)
    total_count = len(data)
    
    for row in data:
        prior_counts[row[category_index]] += 1
    
    prior_probs = {category: count / total_count for category, count in prior_counts.items()}
    return prior_probs

# Function to calculate likelihoods of features given the class (simplified for categorical data)
def calculate_likelihoods(data, category_index):
    likelihoods = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    feature_counts = defaultdict(lambda: defaultdict(int))
    
    for row in data:
        category = row[category_index]
        for i, feature_value in enumerate(row):
            if i != category_index:  # Exclude category itself
                likelihoods[category][i][feature_value] += 1
                feature_counts[i][feature_value] += 1
    
    # Convert counts to probabilities
    for category in likelihoods:
        for i in likelihoods[category]:
            for feature_value in likelihoods[category][i]:
                likelihoods[category][i][feature_value] /= feature_counts[i][feature_value]
    
    return likelihoods

# Function to predict the category for each row based on Bayes' Theorem
def predict(data, headers, prior_probs, likelihoods, category_index):
    predictions = []
    probabilities = []
    
    for row in data:
        max_prob = -float('inf')
        best_category = None
        prob_details = {}
        
        for category in prior_probs:
            prob = math.log(prior_probs[category])
            prob_details[category] = prob
            
            for i, feature_value in enumerate(row):
                if i != category_index:
                    prob += math.log(likelihoods[category][i].get(feature_value, 1e-6))
                    prob_details[category] += math.log(likelihoods[category][i].get(feature_value, 1e-6))
            
            if prob > max_prob:
                max_prob = prob
                best_category = category
        
        exp_probs = {cat: math.exp(prob_details[cat]) for cat in prob_details}
        sum_exp = sum(exp_probs.values())
        normalized_probs = {cat: exp_probs[cat] / sum_exp for cat in exp_probs}
        
        best_category = max(normalized_probs, key=normalized_probs.get)
        predictions.append(best_category)
        probabilities.append(normalized_probs)
    
    return predictions, probabilities

# Streamlit interface
st.title("Naive Bayes Prediction App")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    headers, data = load_csv(uploaded_file)
    
    st.write("Available categories:")
    for i, header in enumerate(headers):
        st.write(f"{i}: {header}")
    
    category_index = st.number_input("Select a category index for prediction:", min_value=0, max_value=len(headers)-1, step=1)
    
    prior_probs = calculate_prior(data, category_index)
    likelihoods = calculate_likelihoods(data, category_index)
    
    # Plotting prior probabilities
    st.subheader("Prior Probabilities")
    fig, ax = plt.subplots()
    categories = list(prior_probs.keys())
    probabilities = list(prior_probs.values())
    sns.barplot(x=categories, y=probabilities, ax=ax)
    ax.set_title("Prior Probabilities for Categories")
    ax.set_xlabel("Categories")
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    if st.button("Make Predictions"):
        predictions, probabilities = predict(data, headers, prior_probs, likelihoods, category_index)
        
        st.write(f"Predictions for '{headers[category_index]}' category:")
        for i, (prediction, prob) in enumerate(zip(predictions, probabilities)):
            st.write(f"Row {i+1}: {prediction} with probabilities {prob}")
        
        # Plotting predictions
        st.subheader("Predicted Categories Pie Chart")
        pred_counts = defaultdict(int)
        for pred in predictions:
            pred_counts[pred] += 1
        
        # Create pie chart
        fig2, ax2 = plt.subplots()
        pred_categories = list(pred_counts.keys())
        pred_values = list(pred_counts.values())
        
        ax2.pie(pred_values, labels=pred_categories, autopct='%1.1f%%', startangle=90)
        ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax2.set_title("Distribution of Predicted Categories")
        
        st.pyplot(fig2)
