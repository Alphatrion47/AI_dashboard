import os
import certifi
import ssl

# Set certificate bundle environment variables
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["CURL_CA_BUNDLE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Ensure online mode
os.environ["TRANSFORMERS_OFFLINE"] = "0"

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import streamlit as st
import pandas as pd
import re
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from transformers import pipeline

# Load spaCy model and add the TextBlob sentiment component
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("spacytextblob")

# Allowed measurement categories (as expected after cleaning)
allowed_categories = [
    "Cost & Control",
    "Quality",
    "Delivery",
    "Communication & Governance",
    "Management"
]

# Mapping for standardizing customer names
name_mapping = {
    "Russ Woodward": "Russ",
    "Robert Churchill": "Rob Churchill",
    "Robert Bell": "Robert Bell",
    "Robert": "Robert Bell",
    "Jonathan King": "Jonathan King",
    "Molly Clarke": "Molly"
}


def standardize_name(name):
    return name_mapping.get(name, name)


def unify_categories(category):
    """
    Force 'Measurement Category' strings to match the known allowed_categories.
    Replace non-breaking spaces, remove extra spaces, lowercase, then normalize.
    """
    c = category.replace("\u00A0", " ")
    c = re.sub(r"\s+", " ", c).strip().lower()
    if c == "cost & control":
        return "Cost & Control"
    elif c == "communication & governance":
        return "Communication & Governance"
    elif c == "quality":
        return "Quality"
    elif c == "delivery":
        return "Delivery"
    elif c == "management":
        return "Management"
    return category.strip()


def sentiment_to_simple(sentiment):
    """
    Converts a sentiment string to a simple emoji.
    Works for both detailed outputs ("😊 Happy", etc.) and breakdown results ("Positive", "Neutral", "Negative").
    """
    mapping = {
        "😊 Happy": "😊",
        "😐 Neutral": "😐",
        "😔 Sad": "😔",
        "Positive": "😊",
        "Neutral": "😐",
        "Negative": "😔"
    }
    return mapping.get(sentiment, sentiment)


def read_uploaded_excel(uploaded_file):
    """
    Reads the Excel file directly from the uploaded file object.
    Assumes that the header row is at index 10 and fills forward missing values.
    """
    df = pd.read_excel(uploaded_file, header=None)
    df.ffill(axis=0, inplace=True)
    df.columns = df.iloc[10].astype(str).fillna("Unnamed").str.strip()
    df = df.iloc[11:].reset_index(drop=True)
    df = df.loc[:, ~df.columns.str.contains("Unnamed|nan", case=False, na=False)]
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def detect_sentiment(comment):
    """
    Computes sentiment using spaCy's TextBlob extension.
    Returns a sentiment label with a smiley: "😊 Happy", "😐 Neutral", or "😔 Sad".
    """
    doc = nlp(comment)
    polarity = doc._.blob.sentiment.polarity
    if polarity >= 0.05:
        return "😊 Happy"
    elif polarity <= -0.05:
        return "😔 Sad"
    else:
        return "😐 Neutral"


def extract_matched_comments(df):
    """
    Extracts matched comments from the DataFrame.
    - Cleans the "Measurement Category" column by unifying names.
    - Filters rows to keep only allowed_categories.
    - Uses a regex that allows ampersands to capture customer: comment lines.
    """
    # Identify required columns
    category_col = next((col for col in df.columns if "measurement category" in col.lower()), None)
    comments_col = next((col for col in df.columns if "comments" in col.lower()), None)
    if not category_col or not comments_col:
        raise ValueError("❌ Required columns not found in the uploaded file.")

    df_matched = df[[category_col, comments_col]].dropna()
    df_matched.columns = ["Measurement Category", "Comments"]

    # Clean and unify the Measurement Category values
    df_matched["Measurement Category"] = df_matched["Measurement Category"].apply(unify_categories)
    df_matched = df_matched[df_matched["Measurement Category"].isin(allowed_categories)]

    # Regex to capture lines like "Cost & Control: some comment"
    pattern = r"([A-Za-z& ]+):\s*(.*)"
    extracted_data = []
    for _, row in df_matched.iterrows():
        category = row["Measurement Category"]
        comments = str(row["Comments"])
        matches = re.findall(pattern, comments)
        if matches:
            for name, comment in matches:
                extracted_data.append({
                    "Measurement Category": category,
                    "Customer": standardize_name(name.strip()),
                    "Comment": comment.strip(),
                    "Sentiment": detect_sentiment(comment)
                })
        else:
            extracted_data.append({
                "Measurement Category": category,
                "Customer": "",
                "Comment": comments.strip(),
                "Sentiment": detect_sentiment(comments)
            })
    # Debug print (check console)
    print("Unique categories in final data:", df_matched["Measurement Category"].unique())
    return pd.DataFrame(extracted_data)


@st.cache_resource
def get_summarizer():
    """Initializes and caches the T5-small summarization pipeline."""
    return pipeline("summarization", model="t5-small")


def summarize_text_minimal(text, summarizer, max_length=120, min_length=30):
    """
    Summarizes the given text with T5-small by prepending 'summarize: ' to the text.
    """
    input_text = "summarize: " + text
    try:
        output = summarizer(input_text, max_length=max_length, min_length=min_length, do_sample=False)
        return output[0]["summary_text"]
    except Exception as e:
        return text


### Consolidated Customer Review Analysis ###

def analyze_customer_review_by_area(review_text):
    """
    Splits the consolidated review text into assessment areas by splitting on newline.
    For each line that contains a colon, splits into area and review, then computes sentiment.
    Returns a list of breakdown results and an overall sentiment (average polarity).
    """
    lines = review_text.split("\n")
    area_results = []
    polarities = []
    for line in lines:
        if ":" in line:
            area, text = line.split(":", 1)
            area = area.strip()
            text = text.strip()
            doc = nlp(text)
            polarity = doc._.blob.sentiment.polarity
            polarities.append(polarity)
            if polarity >= 0.05:
                sentiment = "Positive"
            elif polarity <= -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            area_results.append({
                "Assessment Area": area,
                "Review": text,
                "Sentiment": sentiment
            })
    overall_polarity = sum(polarities) / len(polarities) if polarities else 0
    if overall_polarity >= 0.05:
        overall_sentiment = "Positive"
    elif overall_polarity <= -0.05:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"
    return area_results, overall_sentiment


def consolidate_customer_reviews_textblob(df):
    """
    Groups all unique reviews by customer (ignoring assessment areas) and concatenates them using newline.
    Then applies a breakdown analysis that splits the consolidated review into assessment areas,
    computes sentiment for each, and derives an overall sentiment.
    Returns a list of dictionaries (one per customer).
    """
    all_reviews = []
    for _, row in df.iterrows():
        text = row["Comments"]
        # Split into lines and remove empties; use a set for uniqueness
        reviews = {line.strip() for line in text.split("\n") if line.strip()}
        for review in reviews:
            # Expect format: "Customer: review text"
            parts = review.split(":", 1)
            if len(parts) < 2:
                continue
            customer = standardize_name(parts[0].strip())
            review_text = parts[1].strip()
            all_reviews.append({"Customer": customer, "Review": review_text})
    df_reviews = pd.DataFrame(all_reviews)

    # Consolidate unique reviews per customer using newline separator
    grouped = df_reviews.groupby("Customer")["Review"].apply(lambda reviews: "\n".join(set(reviews))).reset_index()

    consolidated_data = []
    for idx, row in grouped.iterrows():
        customer = row["Customer"]
        review_text = row["Review"]
        breakdown, overall = analyze_customer_review_by_area(review_text)
        consolidated_data.append({
            "Customer": customer,
            "Overall Sentiment": overall,
            "Breakdown": breakdown,
            "Consolidated Review": review_text
        })
    return consolidated_data


### Main App UI ###

def main():
    st.set_page_config(page_title="Customer Comments Consolidator", layout="wide")

    if st.session_state.get("uploaded_file"):
        st.markdown(
            f"<h3 style='margin: 0; padding: 0;'>{st.session_state.uploaded_file.name}</h3>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.title("Customer Feedback Dashboard")

    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
        st.session_state.processed_df = pd.DataFrame()

    # File upload section
    if st.session_state.uploaded_file is None:
        st.header("📤 Upload Excel File")
        uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])
        if uploaded_file is not None:
            try:
                df = read_uploaded_excel(uploaded_file)
                # Process file using TextBlob-based extraction for detailed view
                df_matched = extract_matched_comments(df)
                st.session_state.uploaded_file = uploaded_file
                st.session_state.processed_df = df_matched
                st.success("✅ File uploaded and processed successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
    else:
        view_mode = st.sidebar.radio("View Mode",
                                     options=["Detailed (TextBlob)", "Consolidated Customer (TextBlob)"],
                                     index=0)
        df = st.session_state.processed_df
        if df.empty:
            st.warning("No comments available for the selected filters.")
        else:
            if view_mode == "Detailed (TextBlob)":
                # Optional filtering based on Measurement Category or Customer
                selected_category = st.sidebar.selectbox("Select Measurement Category",
                                                         ["All"] + sorted(df["Measurement Category"].unique()), index=0)
                selected_customer = st.sidebar.selectbox("Filter by Customer",
                                                         ["All"] + sorted(df["Customer"].unique()), index=0)

                filtered_df = df.copy()
                if selected_category != "All":
                    filtered_df = filtered_df[filtered_df["Measurement Category"] == selected_category]
                if selected_customer != "All":
                    filtered_df = filtered_df[filtered_df["Customer"] == selected_customer]
                filtered_df = filtered_df.drop_duplicates(subset=["Customer", "Comment"])

                # Remove the first row if it contains spurious header info
                if not filtered_df.empty:
                    filtered_df = filtered_df.iloc[1:]

                if filtered_df.empty:
                    st.warning("No comments available for the selected filters.")
                else:
                    # Build table with Customer, Sentiment (emoji), and Comment
                    table_data = []
                    for _, row in filtered_df.iterrows():
                        simple_sentiment = sentiment_to_simple(row['Sentiment'])
                        table_data.append([row['Customer'], simple_sentiment, row['Comment']])
                    df_display = pd.DataFrame(table_data, columns=["Customer", "Sentiment", "Comment"])
                    st.markdown(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.write("### Consolidated Customer Reviews with Overall Sentiment (TextBlob)")
                try:
                    uploaded_file = st.session_state.uploaded_file
                    df_full = read_uploaded_excel(uploaded_file)
                    consolidated = consolidate_customer_reviews_textblob(df_full)
                    # Display each customer's results in an expander (partitioned view)
                    for entry in consolidated:
                        overall_emoji = sentiment_to_simple(entry["Overall Sentiment"])
                        with st.expander(f"Customer: {entry['Customer']} | Overall Sentiment: {overall_emoji}"):
                            st.markdown("**Breakdown:**")
                            for item in entry["Breakdown"]:
                                area_emoji = sentiment_to_simple(item["Sentiment"])
                                st.markdown(f"- **{item['Assessment Area']}**: {area_emoji} | {item['Review']}")
                            st.markdown("**Consolidated Review:**")
                            st.write(entry["Consolidated Review"])
                except Exception as e:
                    st.error(f"❌ Error processing TextBlob sentiment analysis: {e}")

        if st.button("Upload Another File"):
            st.session_state.uploaded_file = None
            st.session_state.processed_df = pd.DataFrame()
            st.rerun()


if __name__ == "__main__":
    main()
