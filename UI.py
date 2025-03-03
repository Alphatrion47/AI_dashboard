import streamlit as st
import pandas as pd
import re

# Allowed categories
allowed_categories = [
    "Cost & Control",
    "Quality",
    "Delivery",
    "Communication & Governance",
    "Management"
]


def read_uploaded_excel(uploaded_file):
    df = pd.read_excel(uploaded_file, header=None)
    header_row_index = 8
    df.columns = df.iloc[header_row_index]
    df = df.iloc[header_row_index + 1:].reset_index(drop=True)
    return df


def detect_comments_column(df):
    for col in df.columns:
        if "Comments" in str(col):
            return col
    raise ValueError("❌ Could not find 'Comments' column in the file.")


def standardize_name(name):
    name = name.strip()
    name_corrections = {
        "Rob Churchill": "Robert Churchill",
        "Robert Bell": "Robert Bell",
        "Rob": "Robert"
    }
    return name_corrections.get(name, name)


def extract_customer_names_from_comments(comments_column):
    pattern = r"([A-Za-z ]+):"
    all_names = set()

    for comment in comments_column.dropna():
        matches = re.findall(pattern, str(comment))
        for name in matches:
            all_names.add(standardize_name(name))

    return sorted(list(all_names))


def split_and_expand_comments(comments, customer_names):
    pattern = r"(" + "|".join(re.escape(name) for name in customer_names) + r"):"
    split_comments = re.split(pattern, str(comments))
    split_comments = [chunk.strip() for chunk in split_comments if chunk.strip()]

    customer_comment_dict = {}
    for i in range(0, len(split_comments), 2):
        if i + 1 < len(split_comments):
            customer = standardize_name(split_comments[i])
            comment = split_comments[i + 1]
            if customer not in customer_comment_dict:
                customer_comment_dict[customer] = []
            customer_comment_dict[customer].append(comment)

    return customer_comment_dict


def detect_sentiment(comment):
    comment = comment.lower()
    if any(word in comment for word in ["good", "great", "excellent", "happy", "satisfied"]):
        return "😊 Happy"
    elif any(word in comment for word in ["bad", "poor", "unsatisfied", "sad", "issue"]):
        return "😔 Sad"
    else:
        return "😐 Neutral"


def show_comments(df):
    for _, row in df.iterrows():
        st.markdown(f"""
        **Customer:** {row['Customer']}  
        **Measurement Category:** {row['Measurement Category']}  
        **Comment:** {row['Comment']}  
        """)

        sentiment_icon = {
            "😊 Happy": "😊",
            "😐 Neutral": "😐",
            "😔 Sad": "😔"
        }.get(row['Sentiment'], "❓")

        st.markdown(f"**Sentiment:** {sentiment_icon}")
        st.markdown("---")


def main():
    st.title("Customer Comments Consolidator")

    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

    if 'result_df' not in st.session_state:
        st.session_state.result_df = None

    if 'selected_sentiment' not in st.session_state:
        st.session_state.selected_sentiment = None

    if 'selected_customers' not in st.session_state:
        st.session_state.selected_customers = []

    if st.session_state.uploaded_file is None:
        upload_page()
    else:
        process_page()


def upload_page():
    st.header("Step 1: Upload Excel File")
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.rerun()


def process_page():
    st.header("RHS pane")

    if st.session_state.result_df is None:
        df = read_uploaded_excel(st.session_state.uploaded_file)

        try:
            comments_column = detect_comments_column(df)
        except ValueError as e:
            st.error(str(e))
            if st.button("Upload New File"):
                st.session_state.uploaded_file = None
                st.rerun()
            return

        measurement_category_col = "Measurement Category"
        if measurement_category_col not in df.columns:
            st.error(f"❌ Column '{measurement_category_col}' not found!")
            if st.button("Upload New File"):
                st.session_state.uploaded_file = None
                st.rerun()
            return

        df = df[df[measurement_category_col].isin(allowed_categories)].reset_index(drop=True)

        customer_names = extract_customer_names_from_comments(df[comments_column])
        st.session_state.customer_names = customer_names

        result_data = []
        for _, row in df.iterrows():
            comments = row[comments_column]
            category = row[measurement_category_col]

            customer_comments = split_and_expand_comments(comments, customer_names)

            for customer, comment_list in customer_comments.items():
                for comment in comment_list:
                    sentiment = detect_sentiment(comment)
                    result_data.append({
                        "Customer": customer,
                        "Measurement Category": category,
                        "Comment": comment,
                        "Sentiment": sentiment
                    })

        st.session_state.result_df = pd.DataFrame(result_data)

    result_df = st.session_state.result_df

    st.write("Consolidated Comments with Sentiment Analysis")

    left, right = st.columns([0.8, 0.2])

    with right:
        st.markdown("### Sentiment Legend")
        st.markdown("""
        😊 Happy  
        😐 Neutral  
        😔 Sad  
        """, unsafe_allow_html=True)

    with left:
        filtered_df = result_df

        if st.session_state.selected_sentiment:
            filtered_df = filtered_df[filtered_df["Sentiment"] == st.session_state.selected_sentiment]

        if st.session_state.selected_customers:
            filtered_df = filtered_df[filtered_df["Customer"].isin(st.session_state.selected_customers)]

        show_comments(filtered_df)

    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Processed Data", csv, "Consolidated_Comments.csv", "text/csv")

    st.write("### Filter by Sentiment and Customer")

    col1, col2 = st.columns([0.4, 0.6])

    with col1:
        selected_sentiment = st.selectbox(
            "Choose Sentiment to Filter",
            ["😊 Happy", "😐 Neutral", "😔 Sad"],
            index=None,
            placeholder="Select Sentiment"
        )

    with col2:
        st.write("Choose Customers")
        selected_customers = st.multiselect(
            "Select Customers",
            options=st.session_state.customer_names,
            default=st.session_state.selected_customers,
            label_visibility="collapsed"
        )

    if selected_sentiment != st.session_state.selected_sentiment or selected_customers != st.session_state.selected_customers:
        st.session_state.selected_sentiment = selected_sentiment
        st.session_state.selected_customers = selected_customers
        st.rerun()

    if st.button("Upload New File"):
        st.session_state.uploaded_file = None
        st.session_state.result_df = None
        st.session_state.selected_sentiment = None
        st.session_state.selected_customers = []
        st.rerun()


if __name__ == "__main__":
    main()
