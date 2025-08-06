from flask import Flask, render_template, request, redirect, url_for, session
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import google.generativeai as genai
import os
import difflib


from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')

# Set your Gemini API key here or via environment variable

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

# Mapping of logical column names to possible keywords in Excel
COLUMN_KEYWORDS = {
    'sno': ['sno', 'serial', 'number'],
    'skill': ['skill'],
    'quantity': ['quantity', 'positions', 'count', 'no'],
    'aia_non_aia': ['aia', 'non aia'],
    'location': ['location', 'site', 'place'],
    'priority': ['priority', 'priortty'],
    'fulfilment_date_cutoff': ['fulfilment date cut off', 'cutoff', 'deadline'],
    'region': ['region'],
    'rev_loss': ['rev loss', 'revenue loss'],
    'delivery_risk': ['delivery risk', 'risk'],
    'category': ['category'],
    'tower': ['tower'],
    'project_mapping': ['project mapping', 'project'],
    'requirement_received_date': ['requirement received date', 'received date'],
    'status': ['status', 'sourcing status'],
    'profile_shared_on': ['profile shared on', 'profile date'],
    'comments': ['comments', 'remark', 'note'],
    'req_type': ['req type', 'requirement type'],
    'backfill_type': ['backfill type'],
    'revenue_contribution': ['revenue contribution'],
    'closing_date': ['closing date', 'closed date'],
    'cts_poc': ['cts poc', 'poc'],
}

def find_column(df, keywords):
    """Find the best matching column for given keywords."""
    for kw in keywords:
        for col in df.columns:
            if kw.lower() in col.lower():
                return col
    # Fallback: fuzzy match
    matches = difflib.get_close_matches(keywords[0], df.columns, n=1, cutoff=0.6)
    return matches[0] if matches else None

def detect_columns(df):
    """Detect all relevant columns from the DataFrame using COLUMN_KEYWORDS."""
    detected = {}
    for logical_name, keywords in COLUMN_KEYWORDS.items():
        col = find_column(df, keywords)
        if col:
            detected[logical_name] = col
    return detected

@app.route('/', methods=['GET', 'POST'])
def index():
    table_open = None
    table_closed = None
    table_revloss = None
    table_deliveryrisk = None
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
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            if file:
                df = pd.read_excel(file)
                session['tracker_data'] = df.to_json()
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

    if df is not None:
        cols = detect_columns(df)
        # Tab 1: Summary of open positions by skill, location, tower
        if all(x in cols for x in ['status', 'quantity']):
            open_positions = df[df[cols['status']].astype(str).str.lower() == 'open']
            group_cols = [cols.get(c) for c in ['skill', 'location', 'tower'] if cols.get(c)]
            if group_cols:
                table_open = open_positions.groupby(group_cols)[cols['quantity']].sum().reset_index()

        # Tab 2: Summary of closed positions by skill, location, tower
        if all(x in cols for x in ['status', 'quantity']):
            closed_positions = df[df[cols['status']].astype(str).str.lower() == 'closed']
            group_cols = [cols.get(c) for c in ['skill', 'location', 'tower'] if cols.get(c)]
            if group_cols:
                table_closed = closed_positions.groupby(group_cols)[cols['quantity']].sum().reset_index()

        # Tab 3: No of revenue loss positions by skill, tower, location
        if all(x in cols for x in ['rev_loss', 'skill', 'tower', 'location', 'quantity']):
            rev_loss_positions = df[df[cols['rev_loss']].astype(str).str.lower().str.startswith('y')]
            group_cols = [cols.get(c) for c in ['skill', 'tower', 'location'] if cols.get(c)]
            if group_cols:
                table_revloss = rev_loss_positions.groupby(group_cols)[cols['quantity']].sum().reset_index()

        # Tab 4: No of delivery risk positions by skill, tower, location
        if all(x in cols for x in ['delivery_risk', 'skill', 'tower', 'location', 'quantity']):
            delivery_risk_positions = df[df[cols['delivery_risk']].astype(str).str.lower().str.startswith('y')]
            group_cols = [cols.get(c) for c in ['skill', 'tower', 'location'] if cols.get(c)]
            if group_cols:
                table_deliveryrisk = delivery_risk_positions.groupby(group_cols)[cols['quantity']].sum().reset_index()

        # Validation flags
        table_open_valid = table_open is not None and isinstance(table_open, pd.DataFrame) and not table_open.empty
        table_closed_valid = table_closed is not None and isinstance(table_closed, pd.DataFrame) and not table_closed.empty
        table_revloss_valid = table_revloss is not None and isinstance(table_revloss, pd.DataFrame) and not table_revloss.empty
        table_deliveryrisk_valid = table_deliveryrisk is not None and isinstance(table_deliveryrisk, pd.DataFrame) and not table_deliveryrisk.empty

        # Gen AI Insights using Gemini (narrative for pending client feedback + actionable updates for revenue loss and delivery risk)
        insights = None
        comments_text = ""
        feedback_comments = ""
        if 'comments' in cols and 'status' in cols:
            open_comments = df[df[cols['status']].astype(str).str.lower() == 'open'][cols['comments']].dropna().tolist()
            comments_text = "\n".join(map(str, open_comments))
            # Filter comments mentioning feedback
            feedback_comments = "\n".join([c for c in open_comments if 'feedback' in str(c).lower() or 'client' in str(c).lower()])

        # Prepare actionable updates for revenue loss and delivery risk positions
        actionable_text = ""
        if table_revloss_valid and isinstance(table_revloss, pd.DataFrame):
            actionable_rows = table_revloss.to_dict(orient='records')
            actionable_text = "\n".join([
                f"Skill: {row.get('skill', '')}, Tower: {row.get('tower', '')}, Revenue Loss: {row.get('rev_loss', '')}, Delivery Risk: {row.get('delivery_risk', '')}, Quantity: {row.get('quantity', '')}"
                for row in actionable_rows
            ])

        prompt = (
            "Provide a concise narrative summary based on the following comments, focusing on pending client feedback. "
            "Additionally, provide actionable updates for positions with revenue loss and delivery risk:\n\n"
            f"Comments (Pending Client Feedback):\n{feedback_comments}\n\n"
            f"Revenue Loss & Delivery Risk Positions:\n{actionable_text}\n\n"
            "Format your response as:\n"
            "1. Narrative on pending client feedback (2-3 sentences)\n"
            "2. Actionable updates (bullet points, each under 20 words)"
        )
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            insights = response.text if hasattr(response, 'text') else "No insights returned."
        except Exception as e:
            insights = f"Error generating insights: {e}"

    return render_template(
        'index.html',
        table_open=table_open, table_open_valid=table_open_valid,
        table_closed=table_closed, table_closed_valid=table_closed_valid,
        table_revloss=table_revloss, table_revloss_valid=table_revloss_valid,
        table_deliveryrisk=table_deliveryrisk, table_deliveryrisk_valid=table_deliveryrisk_valid,
        insights=insights, bot_answer=bot_answer
    )

if __name__ == '__main__':
    app.run(debug=True, port=5050)
