from flask import Flask, render_template, request, redirect, url_for, session
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import google.generativeai as genai
import os


from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')

# Set your Gemini API key here or via environment variable

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

@app.route('/', methods=['GET', 'POST'])
def index():
    table1 = None
    table2 = None
    table3 = None
    insights = None
    bot_answer = None
    df = None
    # Load DataFrame from session if available
    if 'tracker_data' in session:
        try:
            df = pd.read_json(StringIO(session['tracker_data']))
        except Exception:
            df = None
    if request.method == 'POST':
        # If file is uploaded, process it and store in session
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            if file:
                df = pd.read_excel(file)
                # Save DataFrame to session for later use
                session['tracker_data'] = df.to_json()
        # If bot question is asked, process it using session DataFrame
        if 'bot_question' in request.form:
            question = request.form.get('bot_question')
            if df is not None and question:
                table_md = df.head(30).to_markdown(index=False)
                prompt = (
                    "You are a helpful assistant for resource status tracking. Answer the user's question using the following tracker data.\n"
                    f"Tracker Table:\n{table_md}\n\nQuestion: {question}\n\nAnswer concisely and only using the table data."
                )
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(prompt)
                    bot_answer = response.text if hasattr(response, 'text') else "No answer returned."
                except Exception as e:
                    bot_answer = f"Error generating answer: {e}"
    # Generate tables and insights if df is available
    if df is not None:
        # 1. No of positions Open by towers, Skill, and Priority
        if all(col in df.columns for col in ['Tower', 'Status', 'No of positions', 'Priority', 'Skill']):
            open_positions = df[df['Status'].str.lower() == 'open']
            table1 = open_positions.groupby(['Tower', 'Skill', 'Priority'])['No of positions'].sum().reset_index()
        # 2. No of revenue loss positions vs Delivery positions open with Priority
        if all(col in df.columns for col in ['Type', 'Status', 'No of positions', 'Priority']):
            open_positions = df[df['Status'].str.lower() == 'open']
            rev_loss = open_positions[open_positions['Type'].str.lower() == 'revenue loss'].groupby('Priority')['No of positions'].sum().to_dict()
            delivery = open_positions[open_positions['Type'].str.lower() == 'delivery'].groupby('Priority')['No of positions'].sum().to_dict()
            table2 = {'Revenue Loss': rev_loss, 'Delivery': delivery}
        # 3. No of positions which are closed, include Tower, Skill, and Priority
        if all(col in df.columns for col in ['Status', 'No of positions', 'Tower', 'Skill', 'Priority']):
            closed_positions = df[df['Status'].str.lower() == 'closed']
            table3 = closed_positions.groupby(['Tower', 'Skill', 'Priority'])['No of positions'].sum().reset_index()
        # Gen AI Insights using Gemini (based only on Comments for open requirements)
        comments_text = ""
        if 'Comments' in df.columns and 'Status' in df.columns:
            open_comments = df[df['Status'].str.lower() == 'open']['Comments'].dropna().tolist()
            comments_text = "\n".join(open_comments)
        prompt = (
            "Based only on the following comments from open resource requirements, provide concise, space-efficient actionable insights (max 3 bullet points, each under 20 words).\n"
            f"Comments:\n{comments_text}"
        )
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            insights = response.text if hasattr(response, 'text') else "No insights returned."
        except Exception as e:
            insights = f"Error generating insights: {e}"
    # Pass flags to template to avoid ambiguous DataFrame truth value
    table1_valid = table1 is not None and not table1.empty if table1 is not None else False
    table3_valid = table3 is not None and not table3.empty if table3 is not None else False
    return render_template('index.html', table1=table1, table1_valid=table1_valid, table2=table2, table3=table3, table3_valid=table3_valid, insights=insights, bot_answer=bot_answer)

if __name__ == '__main__':
    app.run(debug=True, port=5050)
