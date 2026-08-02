import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title='Continuous Authentication', page_icon='🛡️', layout='wide')


def find_model_path():
    for path in [
        Path('tuned_behavior_focused_pipeline.pkl'),
        Path('model') / 'tuned_behavior_focused_pipeline.pkl',
        Path('model') / 'fraud_model.pkl',
    ]:
        if path.exists():
            return path
    return None


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


def create_features(values):
    timestamp = values['timestamp']
    hour = timestamp.hour
    speed = max(float(values['Keyboard_Speed_WPM']), 1e-6)
    return pd.DataFrame([{
        'Keyboard_Speed_WPM': values['Keyboard_Speed_WPM'],
        'Keystroke_Duration_ms': values['Keystroke_Duration_ms'],
        'Typing_Accuracy_pct': values['Typing_Accuracy_pct'],
        'Mouse_Speed_pxs': values['Mouse_Speed_pxs'],
        'Network_Status': values['Network_Status'],
        'CPU_Usage_pct': values['CPU_Usage_pct'],
        'Memory_Usage_pct': values['Memory_Usage_pct'],
        'Previous_Authentication_Score': values['Previous_Authentication_Score'],
        'Hour': hour,
        'Is_Weekend': int(timestamp.weekday() >= 5),
        'Hour_Sin': np.sin(2 * np.pi * hour / 24),
        'Hour_Cos': np.cos(2 * np.pi * hour / 24),
        'Keystroke_Efficiency': values['Keystroke_Duration_ms'] / speed,
        'Mouse_to_Keyboard_Ratio': values['Mouse_Speed_pxs'] / speed,
        'Typing_Accuracy_Adjusted_Speed': (
            values['Keyboard_Speed_WPM'] * values['Typing_Accuracy_pct'] / 100
        ),
    }])


def get_decision(risk_score):
    if risk_score < 0.30:
        return 'ALLOW'
    if risk_score <= 0.70:
        return 'REQUEST_OTP'
    return 'BLOCK_AND_ALERT'


st.title('🛡️ Continuous Authentication')
st.caption('Behavior-focused Random Forest risk assessment')

model_path = find_model_path()
if model_path is None:
    st.error('Model not found. Run the notebook save-model cell first.')
    st.stop()

model = load_model(str(model_path))
st.success(f'Model loaded: {model_path}')

with st.form('authentication_form'):
    st.subheader('Session Information')
    session_id = st.text_input('Session ID', value=f'SESSION-{datetime.now():%Y%m%d%H%M%S}')
    session_date = st.date_input('Session Date', value=datetime.now().date())
    session_time = st.time_input('Session Time', value=datetime.now().time().replace(microsecond=0))

    st.subheader('Behavioral and System Features')
    left, right = st.columns(2)
    with left:
        keyboard_speed = st.number_input('Keyboard Speed (WPM)', 0.0, 200.0, 50.0)
        keystroke_duration = st.number_input('Keystroke Duration (ms)', 0.0, 500.0, 130.0)
        typing_accuracy = st.number_input('Typing Accuracy (%)', 0.0, 100.0, 90.0)
        mouse_speed = st.number_input('Mouse Speed (px/s)', 0.0, 1000.0, 320.0)
    with right:
        cpu_usage = st.number_input('CPU Usage (%)', 0.0, 100.0, 45.0)
        memory_usage = st.number_input('Memory Usage (%)', 0.0, 100.0, 50.0)
        previous_score = st.number_input('Previous Authentication Score', 0.0, 1.0, 0.75)
        network_status = st.selectbox('Network Status', ['Stable', 'Unstable'])

    submitted = st.form_submit_button('Check Session', type='primary')

if submitted:
    timestamp = datetime.combine(session_date, session_time)
    values = {
        'timestamp': timestamp,
        'Keyboard_Speed_WPM': keyboard_speed,
        'Keystroke_Duration_ms': keystroke_duration,
        'Typing_Accuracy_pct': typing_accuracy,
        'Mouse_Speed_pxs': mouse_speed,
        'Network_Status': network_status,
        'CPU_Usage_pct': cpu_usage,
        'Memory_Usage_pct': memory_usage,
        'Previous_Authentication_Score': previous_score,
    }

    try:
        features = create_features(values)
        risk_score = float(model.predict_proba(features)[0, 1])
        decision = get_decision(risk_score)
        payload = {
            'session_id': session_id,
            'timestamp': timestamp.isoformat(),
            'risk_score': round(risk_score, 6),
            'decision': decision,
            'features': features.iloc[0].to_dict(),
        }

        st.subheader('Authentication Result')
        result_left, result_right = st.columns(2)
        result_left.metric('Risk Score', f'{risk_score:.2%}')
        result_right.metric('Decision', decision)

        if decision == 'ALLOW':
            st.success('Session allowed.')
        elif decision == 'REQUEST_OTP':
            st.warning('OTP verification is required.')
        else:
            st.error('Session blocked. Security alert is required.')

        with st.expander('Webhook Payload'):
            st.json(payload)

        webhook_url = os.getenv('N8N_WEBHOOK_URL')
        if webhook_url:
            if st.button('Send Result to n8n'):
                response = requests.post(webhook_url, json=payload, timeout=10)
                response.raise_for_status()
                st.success('Result sent to n8n successfully.')
        else:
            st.info('Set N8N_WEBHOOK_URL to enable the n8n webhook button.')
    except Exception as error:
        st.error(f'Prediction failed: {error}')
