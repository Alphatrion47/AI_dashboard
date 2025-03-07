import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
 
def main():
    # Set page config for full screen width
    st.set_page_config(layout="wide")
 
    # Custom styling
    st.markdown(
        """
        <style>
            body {
                background-color: #f60;
                margin: 0;
                padding: 0;
            }
            .stApp {
                padding: 10px;
                border: 6px solid black;
            }
            .stHeader {
                background-color: #f60;
                padding: 20px;
                text-align: center;
                font-size: 34px;
                font-weight: bold;
                color: white;
                box-shadow: 5px 5px 16px rgba(0,0,0,0.4);
                position: sticky;
                top: 0;
                width: 100%;
                z-index: 1000;
            }
            .stSubHeader {
                background-color: #ff7f00;
                padding: 8px;
                text-align: center;
                font-size: 18px;
                font-weight: bold;
                color: white;
                border-radius: 8px;
                margin-bottom: 8px;
            }
            .stTable table {
                width: 100%;
                border-collapse: collapse;
            }
            .stTable th, .stTable td {
                border: 2px solid white;
                padding: 12px;
                text-align: center;
                font-size: 18px;
                font-weight: bold;
            }
            .stCustomerSummary {
                margin-top: 20px;
                padding: 15px;
                background-color: #ff7f00;
                border-radius: 16px;
                box-shadow: 10px 10px 20px rgba(0,0,0,0.5);
                color: white;
                font-size: 18px;
                font-weight: bold;
            }
            .stLegendBox {
                background-color: #ff7f00;
                padding: 8px;
                border-radius: 8px;
                color: white;
                font-size: 14px;
                font-weight: bold;
                box-shadow: 5px 5px 10px rgba(0,0,0,0.3);
                width: 180px;
                margin-left: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
 
    st.markdown("<div class='stHeader'>✈️ eBot</div>", unsafe_allow_html=True)
 
    st.file_uploader("Upload File (Max 200MB)", type=["csv", "xlsx", "json"])
 
    uploaded_File = st.multiselect("Select Category",
                                   ["TCS Ecommerce", "TCS Technology", "TCS Reliability", "TCS Innovation",
                                    "TCS Productivity"])
 
    st.subheader("Customer Engagement Data")
    columns = ["Client Review Files"] + [f"{month} 2025" for month in
                                         ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]]
 
    def highlight_cells(value):
        try:
            numeric_value = int(value.strip('%'))
            if 60 <= numeric_value < 70:
                return "background-color: lightgreen; color: black; font-weight: bold; font-size: 18px;"
            elif numeric_value >= 70:
                return "background-color: darkgreen; color: white; font-weight: bold; font-size: 18px;"
            elif 50 <= numeric_value < 60:
                return "background-color: yellow; color: black; font-weight: bold; font-size: 18px;"
        except ValueError:
            pass
        return "font-weight: bold; font-size: 18px;"
 
    df = pd.DataFrame([
        ["Ecommerce"] + [f"{65 + (i % 5)}%" for i in range(12)],
        ["Technology"] + [f"{66 + (i % 4)}%" for i in range(12)],
        ["Reliability"] + [f"{67 + (i % 3)}%" for i in range(12)],
        ["Innovation"] + [f"{68 + (i % 2)}%" for i in range(12)],
        ["Productivity"] + [f"{69 + (i % 4)}%" for i in range(12)],
    ], columns=columns, index=range(1, 6))
 
    col1, col2, col3 = st.columns([5, 1, 2], gap="small")
    with col1:
        st.markdown("<div class='stSubHeader'>Customer Engagement Data</div>", unsafe_allow_html=True)
st.write(df.style.applymap(highlight_cells).set_table_styles([
            {"selector": "th", "props": "font-weight: bold; background-color: #ff7f00; color: white; font-size: 18px;"},
            {"selector": "td", "props": "font-weight: bold; font-size: 18px;"}
        ]), unsafe_allow_html=True)
 
    with col2:
        st.markdown(
            """
            <div class='stLegendBox'>
                <strong>Color Coding:</strong><br>
                🟩 <strong>Dark Green</strong>: Very Good (>= 70%)<br>
                🟢 <strong>Light Green</strong>: Good (60% - 70%)<br>
                🟨 <strong>Yellow</strong>: Average (50% - 60%)
            </div>
            """,
            unsafe_allow_html=True
        )
 
    with col3:
        st.markdown("<div class='stSubHeader'>Overall Monthly Performance</div>", unsafe_allow_html=True)
        avg_values = [sum([int(cell.strip('%')) for cell in row[1:]]) / len(row[1:]) for row in df.values]
        labels = df["Client Review Files"].tolist()
 
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        wedges, texts, autotexts = ax.pie(avg_values, labels=labels, autopct="%1.1f%%", startangle=90,
                                          colors=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0"])
 
        for text, label in zip(texts, labels):
            text.set_text(f"{label}")
 
        ax.axis("equal")
        st.pyplot(fig)
 
    st.markdown("<div class='stCustomerSummary'>", unsafe_allow_html=True)
    st.markdown("<div class='stSubHeader'>Customer Comments and Sentiment</div>", unsafe_allow_html=True)
 
    customer_data = pd.DataFrame([
        ["John Doe", "Great service! Very satisfied with the support.", "😊", "No"],
        ["Jane Smith", "Average experience. Could improve response time.", "😐", "Yes"],
        ["Alice Johnson", "Very disappointed. Issues were not resolved.", "😢", "No"],
        ["Bob Brown", "Excellent support! Quick and efficient service.", "😊", "Yes"],
        ["Charlie Davis", "Could be better. Some delays in delivery.", "😐", "No"]
    ], columns=["Customer", "Comment", "Sentiment", "Repetition Happened"], index=range(1, 6))
 
    st.table(customer_data)
    st.markdown("</div>", unsafe_allow_html=True)
 
    st.button("Refresh Dashboard")
 
if __name__ == "__main__":
    main()
