from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import telegram
from collections import Counter

app = Flask(__name__)

# Load model and preprocessors
model = joblib.load("rf2_model.joblib")
scaler = joblib.load("scaler2.joblib")
le = joblib.load("label_encoder2.joblib")
column_order = joblib.load("column_order2.joblib")

# Telegram bot config
TELEGRAM_TOKEN = "7581609196:AAGmJ9xrewaaIGPipI9CJfJl22nyJOYzVeY"
CHAT_ID = "1271011122"
bot = telegram.Bot(token=TELEGRAM_TOKEN)

def preprocess_and_predict(df):
    # Encode categorical features
    df_encoded = pd.get_dummies(df, columns=["protocol_type", "service", "flag"])
    # Align columns
    df_encoded = df_encoded.reindex(columns=column_order, fill_value=0)
    # Scale
    X_scaled = scaler.transform(df_encoded)
    # Predict
    y_pred = model.predict(X_scaled)
    labels = le.inverse_transform(y_pred)
    return labels

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    try:
        # Load uploaded CSV without header
        df = pd.read_csv(file, header=None)
        # Assign columns (adjust if file columns differ)
        columns = [
            "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
            "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
            "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
            "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
            "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
            "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
            "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty_level"
        ]
        df.columns = columns
        
        # Drop label & difficulty_level (they won't be in test)
        df_features = df.drop(['label', 'difficulty_level'], axis=1)
        
        # Predict
        preds = preprocess_and_predict(df_features)
        
        # Count summary
        summary = Counter(preds)
        
        # Telegram message
        message = "Attack Prediction Summary:\n" + "\n".join(f"{k}: {v}" for k, v in summary.items())
        bot.send_message(chat_id=CHAT_ID, text=message)
        
        return jsonify({"summary": summary, "predictions": list(preds)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
