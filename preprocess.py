import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from transformers import pipeline, GPT2TokenizerFast
import os


from datasets import load_dataset


# Load the dataset
dataset = load_dataset("Pablinho/movies-dataset")


# Explore the dataset structure
print(dataset["train"].column_names)
print(dataset["train"][0])  # Print the first entry to understand its structure
print("Sample Record:", dataset["train"].select(range(5)))


import re


def clean_text(text):
   if text is None:
       return ""
   text = text.lower()
   text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
   return text


# Example to check for missing values
print("Missing Overviews:", sum([1 for entry in dataset["train"] if entry["Overview"] is None]))




# Load the sentence transformer model
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
# Generate embeddings for each movie overview
overviews = [clean_text(entry['Overview']) for entry in dataset['train']]
embeddings = model.encode(overviews)
# Save embeddings to a .npz file
np.savez_compressed("data/movie_embeddings.npz", embeddings=embeddings)



