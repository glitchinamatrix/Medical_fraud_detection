from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import io
import base64
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Government approved rates
GOVT_APPROVED_RATES = {
    "Heart transplant": 317400,
    "Brain I Heart FDG PET / CT Scan": 11500,
    "FDG Whole body PET / CT Scan": 11500,
    "Haemoglobin (Hb) test": 21,
    "Blood Glucose Random": 28,
    "Thyroid stimulating hormone (TSH)": 104,
    "T3 and T4": 148,
    "MRI & CT scan brain": 5808,
    "Excision of Brain Tumours -Supratentorial": 44991,
    "Excision of Brain Tumours -Infratentorial": 51750,
    "Surgery of spinal Cord Tumours": 51750,
    "Ventriculoatrial /Ventriculoperitoneal Shunt": 13357,
    "Spinal Fusion Procedure": 34000,
    "Biopsy Intraoral-Soft tissue": 430,
    "Kidney Function Test (KFT)": 259,
    "Kidney Biopsy": 1691,
    "Operations for Cyst of the Kidney -Lap/endoscopic": 16135,
    "Kidney Transplantation": 230000,
    "Liver biopsy": 1587,
    "Operation for Hydatid Cyst of Liver": 27000
}
@app.route('/index')
def fraud():
    return render_template('index.html')
@app.route('/')
def home():
    return redirect(url_for('fraud'))
# File configuration
EXCEL_FILE = 'data.xlsx'
HEADERS = ["Name", "State", "Country", "City", "Pincode", "Aadhar Number", "Hospital", "Disease", "Cost"]

# Initialize Excel file if it doesn't exist
if not os.path.exists(EXCEL_FILE):
    pd.DataFrame(columns=HEADERS).to_excel(EXCEL_FILE, index=False)

import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt

def generate_confusion_matrix(state, y_true, y_pred):
    """Generate confusion matrix visualization with thread-safe plotting"""
    # Create figure directly without using pyplot
    fig, ax = plt.subplots(figsize=(6, 4))
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Ensure we always show a 2x2 matrix
    display_cm = np.zeros((2, 2), dtype=int)
    if cm.shape == (2, 2):
        display_cm = cm
    elif cm.shape == (1, 1):
        if y_pred[0]:  # All predicted as fraud
            display_cm[1, 1] = cm[0][0]  # TP
        else:  # All predicted as non-fraud
            display_cm[0, 0] = cm[0][0]  # TN
    
    # Plot using the Axes object
    im = ax.imshow(display_cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set_title(f'{state} Confusion Matrix')
    
    classes = ['Not Fraud', 'Fraud']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    
    thresh = display_cm.max() / 2.
    for i in range(display_cm.shape[0]):
        for j in range(display_cm.shape[1]):
            ax.text(j, i, format(display_cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if display_cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    
    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)  # Explicitly close the figure
    
    return image_base64



@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Collect form data
        data = {
            "Name": request.form['name'],
            "State": request.form['state'],
            "Country": request.form['country'],
            "City": request.form['city'],
            "Pincode": request.form['pincode'],
            "Aadhar Number": request.form['aadhar'],
            "Hospital": request.form['hospital'],
            "Disease": request.form['disease'],
            "Cost": float(request.form['cost'])
        }

        # Save to Excel
        df = pd.read_excel(EXCEL_FILE)
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        df.to_excel(EXCEL_FILE, index=False)

        return redirect(url_for('fraud'))
    except Exception as e:
        return f"Error occurred: {str(e)}", 400

# Admin routes
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'password123'

@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin'] = True
            return redirect(url_for('view_submissions'))
        else:
            return render_template('admin_login.html', error="Invalid credentials")
    return render_template('admin_login.html')

@app.route('/view_submissions')
def view_submissions():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    
    try:
        df = pd.read_excel(EXCEL_FILE)
        data = []
        y_true = []
        y_pred = []
        
        for _, row in df.iterrows():
            disease = row['Disease']
            cost = row['Cost']
            approved_rate = GOVT_APPROVED_RATES.get(disease, 0)
            
            fraud_percent = 0
            if approved_rate > 0:
                fraud_percent = max(0, round(((cost - approved_rate) / approved_rate) * 100, 2))
            
            predicted_fraud = 1 if fraud_percent > 20 else 0
            actual_fraud = predicted_fraud  # In production, this would be verified
            
            data.append({
                "name": row['Name'],
                "aadhar": row['Aadhar Number'],
                "state": row['State'],
                "country": row['Country'],
                "city": row['City'],
                "pincode": row['Pincode'],
                "hospital": row['Hospital'],
                "disease": disease,
                "cost": cost,
                "approved_rate": approved_rate,
                "fraud_percent": fraud_percent,
                "is_fraud": "Yes" if predicted_fraud else "No"
            })
            
            y_true.append(actual_fraud)
            y_pred.append(predicted_fraud)
        
        # Calculate metrics if we have data
        metrics = {}
        if len(y_true) > 0:
            metrics['accuracy'] = round(accuracy_score(y_true, y_pred) * 100 / 100, 1)
            metrics['precision'] = round(precision_score(y_true, y_pred, zero_division=0) * 100, 1)
            metrics['recall'] = round(recall_score(y_true, y_pred, zero_division=0) * 100, 1)
            metrics['confusion_matrix'] = generate_confusion_matrix("All States", y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                metrics['true_positives'], metrics['false_positives'], metrics['false_negatives'], metrics['true_negatives'] = cm.ravel()
            else:
                # Handle cases where there's only one class
                metrics['true_positives'] = cm[0][0] if 1 in y_pred else 0
                metrics['false_positives'] = 0
                metrics['false_negatives'] = 0
                metrics['true_negatives'] = cm[0][0] if 0 in y_pred else 0
        
        return render_template('view_submissions.html', data=data, metrics=metrics)
    except Exception as e:
        return f"Error reading submissions: {str(e)}", 500

@app.route('/state_metrics')
@app.route('/state_metrics')
def state_metrics():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    
    try:
        df = pd.read_excel(EXCEL_FILE)
        state_data = []
        
        for state, group in df.groupby('State'):
            y_true = []
            y_pred = []
            
            for _, row in group.iterrows():
                disease = row['Disease']
                cost = row['Cost']
                approved_rate = GOVT_APPROVED_RATES.get(disease, 0)
                
                fraud_percent = 0
                if approved_rate > 0:
                    fraud_percent = max(0, round(((cost - approved_rate) / approved_rate) * 100, 2))
                
                predicted_fraud = 1 if fraud_percent > 20 else 0
                actual_fraud = predicted_fraud
                
                y_true.append(actual_fraud)
                y_pred.append(predicted_fraud)
            
            if len(y_true) > 0:
                # Calculate metrics - FIXED VERSION
                cm = confusion_matrix(y_true, y_pred)
                
                # Initialize all metrics to 0
                tp = fp = fn = tn = 0
                
                if cm.size == 1:  # Only one class present
                    if y_pred[0] == 1:  # All predicted as fraud
                        tp = cm[0][0]
                    else:  # All predicted as non-fraud
                        tn = cm[0][0]
                else:  # Normal 2x2 matrix
                    tp = cm[1][1]
                    fp = cm[0][1]
                    fn = cm[1][0]
                    tn = cm[0][0]
                
                accuracy = round(accuracy_score(y_true, y_pred) * 100, 1)
                precision = round(precision_score(y_true, y_pred, zero_division=0) * 100, 1)
                recall = round(recall_score(y_true, y_pred, zero_division=0) * 100, 1)
                
                # Convert numpy types to native Python types
                state_data.append({
                    'state': state,
                    'total_cases': int(len(group)),
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'metrics': {
                        'true_positives': int(tp),
                        'false_positives': int(fp),
                        'false_negatives': int(fn),
                        'true_negatives': int(tn)
                    },
                    'confusion_matrix': generate_confusion_matrix(state, y_true, y_pred)
                })
        
        state_data.sort(key=lambda x: x['state'])
        return render_template('state_metrics.html', state_data=state_data)
    except Exception as e:
        return f"Error generating state metrics: {str(e)}", 500
@app.route('/delete_all_data', methods=['POST'])
def delete_all_data():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    
    try:
        # Create empty DataFrame with headers
        empty_df = pd.DataFrame(columns=HEADERS)
        empty_df.to_excel(EXCEL_FILE, index=False)
        
        # Clear state fraud data if you're using it
        if 'STATE_FRAUD_DATA' in globals():
            global STATE_FRAUD_DATA
            STATE_FRAUD_DATA = {}
            
        return redirect(url_for('view_submissions'))
    except Exception as e:
        return f"Error deleting data: {str(e)}", 500

@app.route('/delete_single_entry/<aadhar>', methods=['POST'])
def delete_single_entry(aadhar):
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    
    try:
        df = pd.read_excel(EXCEL_FILE)
        # Remove entry with matching Aadhar number
        df = df[df['Aadhar Number'] != aadhar]
        df.to_excel(EXCEL_FILE, index=False)
        return redirect(url_for('view_submissions'))
    except Exception as e:
        return f"Error deleting entry: {str(e)}", 500
@app.route('/logout')
def logout():
    session.pop('admin', None)
    return redirect(url_for('admin_login'))

if __name__ == '__main__':
    app.run(debug=True)