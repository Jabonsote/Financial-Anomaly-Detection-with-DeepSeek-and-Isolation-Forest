import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import ollama
import model
from model import AnomalyDetector
import os
import threading
import webbrowser
from fpdf import FPDF
import matplotlib.pyplot as plt
import markdown2
import re



# SQLite setup
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user TEXT,
        bot TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()

def save_to_db(user_message, bot_response):
    cursor.execute("INSERT INTO chat (user, bot) VALUES (?, ?)", (user_message, bot_response))
    conn.commit()

def fetch_chat_history():
    cursor.execute("SELECT user, bot, timestamp FROM chat ORDER BY timestamp DESC")
    return cursor.fetchall()

def clear_chat_history():
    cursor.execute("DELETE FROM chat")
    conn.commit()

def detect_time_series_columns(df):
    """Detect potential timestamp columns and numerical value columns."""
    timestamp_cols = []
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
            if df[col].notna().sum() > len(df) * 0.9 and df[col].dt.hour.nunique() > 1:
                timestamp_cols.append(col)
        except:
            continue
    
    valid_value_cols = [col for col in numerical_cols if col.lower() not in ['id', 'zip']]
    return timestamp_cols, valid_value_cols


class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(200, 10, "Professional Audit Report: Anomaly Detection", ln=True, align="C")
        self.ln(10)

    def chapter_title(self, title):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, title.encode('latin-1', 'replace').decode('latin-1'), ln=True, align="L")
        self.ln(5)

    def chapter_body(self, body):
        self.set_font("Arial", "", 12)

        # Convertir Markdown a texto enriquecido sin HTML
        body_plain_text = markdown2.markdown(body, extras=["nl2br"]).strip()
        
        # Convertir listas <ol> y <ul> a texto plano
        body_plain_text = re.sub(r"<li><strong>(.*?)</strong>:", r"• \1:", body_plain_text)  # Negrita en listas
        body_plain_text = re.sub(r"<li>(.*?)</li>", r"• \1", body_plain_text)  # Listas simples
        body_plain_text = re.sub(r"<.*?>", "", body_plain_text)  # Eliminar etiquetas HTML restantes
        
        self.multi_cell(0, 10, body_plain_text.encode('latin-1', 'replace').decode('latin-1'))
        self.ln()



# Streamlit UI Setup
st.set_page_config(page_title="📊 Audit LLM 🧠", layout="wide")
st.title("📊 Audit LLM 🧠")


# File Upload
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xls", "xlsx"], accept_multiple_files=False)

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Display dataset preview and statistics
    st.subheader(f"📊 Dataset Preview: {uploaded_file.name}")
    st.dataframe(df.head())  # Show first few rows
    st.subheader("📊 Dataset Statistics")
    st.write(df.describe(include="all"))

    # Detect timestamp and numerical columns
    timestamp_columns, detected_numerical_cols = detect_time_series_columns(df)

    # User selects timestamp column
    selected_timestamp = st.selectbox(
        "🕒 Select Timestamp Column:", 
        df.columns, 
        index=df.columns.get_loc(timestamp_columns[0]) if timestamp_columns else 0
    )

    # Convert selected timestamp column to datetime
    df[selected_timestamp] = pd.to_datetime(df[selected_timestamp], errors='coerce')
    if df[selected_timestamp].isna().all():
        st.error("❌ Invalid timestamp format detected. Please select another column.")

    # **Fix: Ensure user can always select numerical variables**
    # Find numeric columns (including any detected)
    all_numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    valid_numeric_cols = list(set(all_numeric_cols) | set(detected_numerical_cols))

    if not valid_numeric_cols:
        st.warning("⚠️ No valid numerical columns found. Please select at least one manually.")
        valid_numeric_cols = df.columns.tolist()  # Allow selection of any column

    # User selects numerical variables to graph
    selected_values = st.multiselect(
        "📈 Select Value Columns (Numerical Only):", 
        valid_numeric_cols, 
        default=valid_numeric_cols[:1] if valid_numeric_cols else []
    )

    if not selected_values:
        st.warning("⚠️ No valid numerical columns selected for visualization. Please select at least one.")

    else:
        # Sort and filter by date range
        df = df.sort_values(by=selected_timestamp)
        date_range = st.date_input("📆 Select Date Range:", [])
        if len(date_range) == 2:
            df = df[(df[selected_timestamp] >= str(date_range[0])) & (df[selected_timestamp] <= str(date_range[1]))]

        # Aggregation Options
        aggregation_level = st.radio("🔍 Aggregation Level:", ["Daily", "Monthly", "Yearly"], index=0)

        # Resample Data
        if aggregation_level == "Daily":
            df = df.set_index(selected_timestamp).resample('D').mean().reset_index()
        elif aggregation_level == "Monthly":
            df = df.set_index(selected_timestamp).resample('M').mean().reset_index()
        elif aggregation_level == "Yearly":
            df = df.set_index(selected_timestamp).resample('Y').mean().reset_index()

        # Normalize values for better visualization
        df[selected_values] = (df[selected_values] - df[selected_values].min()) / (df[selected_values].max() - df[selected_values].min())

        # Create dynamic color scheme
        colors = px.colors.qualitative.Set1  

        fig = go.Figure()
        for i, value in enumerate(selected_values):
            fig.add_trace(go.Scatter(
                x=df[selected_timestamp], 
                y=df[value], 
                mode='lines', 
                name=value,
                line=dict(color=colors[i % len(colors)])
            ))

        # Detect and highlight outliers dynamically
        for value in selected_values:
            q1, q3 = df[value].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = df[(df[value] < lower_bound) | (df[value] > upper_bound)]

            if not outliers.empty:
                fig.add_trace(go.Scatter(
                    x=outliers[selected_timestamp], 
                    y=outliers[value], 
                    mode='markers', 
                    name=f"Outliers: {value}", 
                    marker=dict(color='red', size=8)
                ))

        # Graph Customization
        fig.update_layout(
            title="📈 Time Series Visualization with Outliers",
            xaxis_title="Time",
            yaxis_title="Normalized Value",
            template="plotly_white",
            legend_title="Metrics"
        )
        st.plotly_chart(fig, use_container_width=True)






if not os.path.exists("data/anomalies.csv"):
    print("Running model.py to generate anomalies...")
    os.system("python model.py")
    
# Initialize the anomaly detector
detector = AnomalyDetector()

# Streamlit Button to Generate Report
if st.button("Generate Audit Report"):
    anomalies_file = "data/anomalies.csv"
    st.write("📝 **Generating full audit report...**")
    if os.path.exists(anomalies_file):
        try:
            anomalies = pd.read_csv(anomalies_file)
            
            # Load the anomalies dataset
            df_anomalies = pd.read_csv(anomalies_file)

            # Display dataset preview and statistics
            st.subheader("📊 Anomaly Detection Report")
            st.write("🚨 **Anomalies detected using the trained model!** 🚨")
            st.dataframe(df_anomalies.head())  # Show first few rows

            # Graph anomalies distribution
            st.subheader("📉 Anomalies Overview")
            if "score" in df_anomalies.columns:  # Check if 'score' column exists for anomaly severity
                fig, ax = plt.subplots()
                df_anomalies["score"].hist(bins=30, edgecolor="black", alpha=0.7, ax=ax)
                ax.set_title("Anomaly Score Distribution")
                ax.set_xlabel("Anomaly Score")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
            else:
                st.write("📌 No anomaly score column found, showing row count instead.")
                st.bar_chart(df_anomalies.count())
            
            if anomalies.empty:
                st.write("No anomalies detected.")
            else:
                # Generate summary using deepseek
                anomalies_dict = anomalies.to_dict(orient="records")
                response = ollama.generate(
                    model="deepseek-r1:1.5b",
                    prompt = f"""
                        You are a professional data scientist and anomaly detection expert analyzing a company dataset for irregularities. 
                        Your goal is to detect, analyze, and explain anomalies in a structured, professional manner.  

                        **Instructions:**  
                        1. **Identify Anomalies:**  
                        - Use a robust anomaly detection technique such as Isolation Forest, One-Class SVM, DBSCAN, Z-score, or IQR.  
                        - Ensure the approach is suitable for the data type (e.g., time-series, categorical, numerical).  

                        2. **Provide a Comprehensive Summary:**  
                        - Clearly explain the detected anomalies, their characteristics, and patterns.  
                        - Include metrics such as deviation from the norm, anomaly scores, and statistical significance.  

                        3. **Analyze Possible Causes:**  
                        - Discuss potential reasons for the anomalies (e.g., data entry errors, fraudulent activities, business process failures).  
                        - If relevant, correlate anomalies with external factors like seasonal trends, operational changes, or market shifts.  

                        4. **Explain Business Implications:**  
                        - Assess how these anomalies impact business performance, risk management, or decision-making.  
                        - Highlight whether anomalies indicate fraud, inefficiencies, compliance issues, or operational risks.  

                        5. **Present the Findings Professionally:**  
                        - Summarize the results concisely with structured insights.  
                        - Provide actionable recommendations for further investigation or corrective measures.  

                        **Data Input:** {anomalies_dict}  

                        Deliver a clear, professional, and insightful analysis based on the detected anomalies.
                        """

                )
                # Después de generar `summary` en Markdown:
                summary_markdown = response["response"]  # El texto generado contiene Markdown

                pdf = PDF()
                pdf.add_page()

                pdf.chapter_title("Company: ABC Corp - Location: Ensenada")
                pdf.chapter_title("Audit Report Date: 2025-02-22")
                pdf.chapter_title("Executive Summary")
                pdf.chapter_body("This report provides an in-depth analysis of detected anomalies in the dataset. The anomalies are analyzed for their root causes, business impact, and potential corrective actions.")

                pdf.chapter_title("Dataset Overview")
                pdf.chapter_body(f"Total records analyzed: {len(df)}\nColumns analyzed: {', '.join(df.columns)}")

                pdf.chapter_title("Anomaly Summary")
                pdf.chapter_body(summary_markdown)  # Se mantiene en Markdown pero convertido a texto formateado

                for index, row in anomalies.iterrows():
                    pdf.chapter_title(f"Anomaly {index + 1}")
                    details = "\n".join([f"{col}: {str(row[col]).encode('latin-1', 'replace').decode('latin-1')}" for col in anomalies.columns])
                    pdf.chapter_body(details)

                report_path = "anomaly_report.pdf"
                pdf.output(report_path, "F")

                # Botón para descargar el informe en Streamlit
                st.subheader("Audit Report Preview")
                with open(report_path, "rb") as file:
                    st.download_button("Download Audit Report", file, file_name="Anomaly_Report.pdf")
                    st.markdown(summary_markdown)  # Muestra el resumen con formato en Streamlit
        except pd.errors.EmptyDataError:
            st.error("Error: Anomalies file is empty. Ensure model.py generated data correctly.")
        except Exception as e:
            st.error(f"Error processing anomalies file: {str(e)}")
    else:
        st.write("No anomalies file found. Please check if model.py executed correctly.")


    documents = df.astype(str).apply(lambda row: ' '.join(row), axis=1).tolist()
    embedder = HuggingFaceEmbeddings()
    vector = FAISS.from_texts(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
user_input = st.text_input("Ask a question about your dataset:")
if user_input:
        relevant_docs = retriever.get_relevant_documents(user_input)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"Context: {context}\nQuestion: {user_input}\nAnswer:"
        response = ollama.generate(model="deepseek-r1:14b", prompt=prompt)
        bot_response = response["response"]
        save_to_db(user_input, bot_response)
        
        st.write(f"**You:** {user_input}")
        st.write(f"**Bot:** {bot_response}")

with st.expander("Chat History", expanded=False):
    history = fetch_chat_history()
    for user_msg, bot_msg, timestamp in history:
        st.write(f"{timestamp} - **You:** {user_msg}")
        st.write(f"{timestamp} - **Bot:** {bot_msg}")

if st.sidebar.button("Download Chat History"):
    history_str = "\n".join([f"{ts} - You: {u} | Bot: {b}" for u, b, ts in history])
    st.sidebar.download_button("Download", history_str, file_name="chat_history.txt")
if st.sidebar.button("Clear Chat History"):
    clear_chat_history()
    st.sidebar.success("Chat history cleared!")
