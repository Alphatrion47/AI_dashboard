import streamlit as st
import pandas as pd
from groq import Groq
import spacy
from nltk.stem.snowball import SnowballStemmer
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load('en_core_web_sm')
stemmer = SnowballStemmer("english")

# client = Groq(api_key= st.secrets["groq_passkey"])

# st.title("AI dashboard")

# if "df" not in st.session_state:
#     st.session_state.df = None

# if "mydf" not in st.session_state:
#     st.session_state.mydf = None

text = pd.read_excel("TCS e-Commerce Scorecard 202501-TCS Comments.xlsx",sheet_name = "Scorecard")


nlp.add_pipe("spacytextblob")
doc = nlp(text)

spacy.sentiment