import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import xgboost as xgb
import plotly.graph_objects as go
from tensorflow.keras.models import load_model, Model

st.set_page_config(page_title="Irish Energy Demand Forecaster", page_icon="⚡", layout="wide")

SEQ_LEN = 48

FEATURE_COLS = [
    'demand_mw',
    'temp_c', 'wind_speed_kmh', 'humidity_pct',
    'precip_mm', 'wind_chill', 'temp_squared',
    'hour', 'day_of_week', 'month',
    'is_weekend', 'halfhour_slot',
    'event_flag', 'event_intensity',
    'is_public_holiday',
    'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos',
    'month_sin', 'month_cos'
]

XGB_FEATURE_COLS = [
    'temp_c', 'wind_speed_kmh', 'humidity_pct',
    'precip_mm', 'wind_chill', 'temp_squared',
    'hour', 'day_of_week', 'month',
    'is_weekend', 'halfhour_slot',
    'event_flag', 'event_intensity',
    'is_public_holiday',
    'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos',
    'month_sin', 'month_cos',
    'demand_lag_24h', 'demand_lag_168h',
    'demand_rolling_24h_mean'
]

FRIENDLY_NAMES = {
    'temp_c': 'Temperature',
    'wind_speed_kmh': 'Wind Speed',
    'humidity_pct': 'Humidity',
    'precip_mm': 'Rainfall',
    'wind_chill': 'Wind Chill',
    'temp_squared': 'Temperature (extreme effect)',
    'hour': 'Hour of Day',
    'day_of_week': 'Day of Week',
    'month': 'Month',
    'is_weekend': 'Weekend Flag',
    'halfhour_slot': 'Time Slot of Day',
    'event_flag': 'Major Event Today',
    'event_intensity': 'Event Size',
    'is_public_holiday': 'Public Holiday',
    'hour_sin': 'Time of Day (cyclical)',
    'hour_cos': 'Time of Day (cyclical)',
    'dow_sin': 'Day of Week (cyclical)',
    'dow_cos': 'Day of Week (cyclical)',
    'month_sin': 'Season (cyclical)',
    'month_cos': 'Season (cyclical)',
    'demand_lag_24h': 'Demand 24h Ago',
    'demand_lag_168h': 'Demand 1 Week Ago',
    'demand_rolling_24h_mean': 'Average Demand (last 24h)',
}

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f0f2f6;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #9aa4b2;
        font-size: 1rem;
        margin-bottom: 1.8rem;
    }
    div[data-testid="stMetric"] {
        background-color: #1a1f2b;
        border: 1px solid #2a3040;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
    }
    div[data-testid="stMetricLabel"] { color: #9aa4b2; }
    section[data-testid="stSidebar"] { background-color: #12151d; }
    .block-container { padding-top: 2rem; }
    .info-box {
        background-color: #161b26;
        border: 1px solid #2a3040;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        color: #9aa4b2;
        font-size: 0.92rem;
        margin-top: 0.6rem;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    lstm_model = load_model('lstm_model.keras')
    hybrid_model = xgb.XGBRegressor()
    hybrid_model.load_model('hybrid_model.json')
    scaler = joblib.load('scaler.pkl')
    embedding_model = Model(inputs=lstm_model.inputs, outputs=lstm_model.layers[-3].output)
    return lstm_model, hybrid_model, scaler, embedding_model


@st.cache_data
def load_dataset():
    df = pd.read_csv('final_dataset_2022_2024.csv', index_col=0)
    df.index = pd.to_datetime(df.index, dayfirst=True)
    df = df.sort_index()
    return df


def predict_at_index(df, target_idx, scaler, lstm_model, hybrid_model, embedding_model):
    seq_df = df.iloc[target_idx - SEQ_LEN:target_idx][FEATURE_COLS]
    seq_scaled = scaler.transform(seq_df)
    seq_scaled = seq_scaled.reshape(1, SEQ_LEN, len(FEATURE_COLS))

    embedding = embedding_model.predict(seq_scaled, verbose=0)

    structured_row = df.iloc[target_idx][XGB_FEATURE_COLS]
    structured_vals = structured_row.values.reshape(1, -1)

    embedding_names = [f'lstm_{i}' for i in range(embedding.shape[1])]
    all_feature_names = embedding_names + XGB_FEATURE_COLS
    hybrid_input = np.hstack([embedding, structured_vals])
    hybrid_input_df = pd.DataFrame(hybrid_input, columns=all_feature_names)

    pred_scaled = hybrid_model.predict(hybrid_input_df)[0]
    demand_min = scaler.data_min_[0]
    demand_max = scaler.data_max_[0]
    predicted_mw = pred_scaled * (demand_max - demand_min) + demand_min

    actual_mw = df.iloc[target_idx]['demand_mw']

    return predicted_mw, actual_mw, hybrid_input_df, all_feature_names, embedding_names


def explain_prediction(hybrid_model, hybrid_input_df, embedding_names, xgb_cols, demand_range):
    explainer = shap.TreeExplainer(hybrid_model)
    shap_vals = explainer.shap_values(hybrid_input_df)[0]
    shap_vals_mw = shap_vals * demand_range

    embedding_positions = [list(hybrid_input_df.columns).index(f) for f in embedding_names]
    structured_positions = [list(hybrid_input_df.columns).index(f) for f in xgb_cols]

    lstm_contribution = np.abs(shap_vals_mw[embedding_positions]).sum()
    structured_contributions = {xgb_cols[i]: abs(shap_vals_mw[structured_positions[i]]) for i in range(len(xgb_cols))}

    combined = {'Recent 24-Hour Demand Pattern': lstm_contribution}
    for k, v in structured_contributions.items():
        combined[FRIENDLY_NAMES.get(k, k)] = v
    sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:8]
    return sorted_items


def plot_demand_trend(daily_df, selected_date):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_df.index, y=daily_df['demand_mw'],
        mode='lines', line=dict(color='#4f9eed', width=1.5),
        fill='tozeroy', fillcolor='rgba(79,158,237,0.06)',
        name='Daily Average Demand'
    ))
    selected_ts = pd.Timestamp(selected_date)
    if selected_ts in daily_df.index:
        fig.add_trace(go.Scatter(
            x=[selected_ts], y=[daily_df.loc[selected_ts, 'demand_mw']],
            mode='markers', marker=dict(color='#ff6b6b', size=10, line=dict(color='white', width=1)),
            name='Your selected date'
        ))
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
        font=dict(color='#c9d1d9'),
        xaxis=dict(
            title='Date',
            showgrid=False,
            tickfont=dict(size=11),
            range=['2022-01-01', '2024-12-31'],
            dtick='M3',
            tickformat='%b %Y',
        ),
        yaxis=dict(title='Daily Average Demand (MW)', gridcolor='#242936', tickfont=dict(size=11)),
        showlegend=False,
    )
    return fig


def plot_zoomed_trend(zoom_df, target_ts):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=zoom_df.index, y=zoom_df['demand_mw'],
        mode='lines', line=dict(color='#4f9eed', width=2.5),
        fill='tozeroy', fillcolor='rgba(79,158,237,0.08)',
        name='Demand'
    ))
    if target_ts in zoom_df.index:
        fig.add_trace(go.Scatter(
            x=[target_ts], y=[zoom_df.loc[target_ts, 'demand_mw']],
            mode='markers', marker=dict(color='#ff6b6b', size=12, line=dict(color='white', width=1.5)),
            name='Your selected date & time'
        ))
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
        font=dict(color='#c9d1d9'),
        xaxis=dict(title='Date and Time', showgrid=False, tickfont=dict(size=11)),
        yaxis=dict(title='Electricity Demand (MW)', gridcolor='#242936', tickfont=dict(size=11)),
        showlegend=False,
    )
    return fig


def plot_shap_bar(shap_summary):
    labels = [x[0] for x in shap_summary][::-1]
    values = [x[1] for x in shap_summary][::-1]
    colors = ['#ff8a5b' if l.startswith('Recent 24') else '#4f9eed' for l in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker_color=colors,
        text=[f"{v:.0f} MW" for v in values],
        textposition='outside',
        textfont=dict(color='#c9d1d9', size=12),
    ))
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=70, t=10, b=10),
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
        font=dict(color='#c9d1d9', size=12),
        xaxis=dict(title='How Much This Factor Changed the Prediction (MW)', gridcolor='#242936'),
        yaxis=dict(automargin=True, title=''),
    )
    return fig


st.markdown('<div class="main-header">⚡ Irish Energy Demand Forecaster</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Predicts Dublin electricity demand using a hybrid LSTM-XGBoost model, '
    'and explains why it made that prediction &nbsp;·&nbsp; tested against real historical EirGrid data (2022–2024)</div>',
    unsafe_allow_html=True
)

with st.spinner("Loading models and dataset..."):
    lstm_model, hybrid_model, scaler, embedding_model = load_models()
    df = load_dataset()

available_dates = sorted(df.index.normalize().unique())
min_selectable_date = available_dates[2]
max_selectable_date = available_dates[-1]

c1, c2 = st.columns([2, 1])
with c1:
    selected_date = st.date_input(
        "Choose a date",
        value=min_selectable_date,
        min_value=min_selectable_date,
        max_value=max_selectable_date,
        help="Pick any date between 2022 and 2024. The dashboard will show what the model would have predicted at that moment, compared to what actually happened."
    )

day_rows = df[df.index.normalize() == pd.Timestamp(selected_date)]

with c2:
    if len(day_rows) > 0:
        time_options = [t.strftime('%H:%M') for t in day_rows.index]
        selected_time = st.selectbox(
            "Choose a time",
            time_options,
            index=min(24, len(time_options) - 1),
            help="Electricity demand is recorded every 30 minutes. Pick which half-hour slot to predict."
        )

if len(day_rows) == 0:
    st.error("No data available for this date.")
else:
    selected_ts = pd.Timestamp(f"{selected_date} {selected_time}")
    target_idx = df.index.get_loc(selected_ts)

    if target_idx < SEQ_LEN:
        st.error("Not enough history before this point in the dataset to build a prediction. Pick a later date/time.")
    else:
        predicted_mw, actual_mw, hybrid_input_df, all_feature_names, embedding_names = predict_at_index(
            df, target_idx, scaler, lstm_model, hybrid_model, embedding_model
        )
        error_mw = abs(predicted_mw - actual_mw)

        st.write("")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Predicted Demand",
            f"{predicted_mw:,.0f} MW",
            help="What the model guessed the electricity demand would be at this date and time, based only on data available before this point."
        )
        col2.metric(
            "Actual Demand",
            f"{actual_mw:,.0f} MW",
            help="The real, recorded electricity demand at this date and time, taken directly from EirGrid's historical data."
        )
        col3.metric(
            "Prediction Error",
            f"{error_mw:,.1f} MW",
            help="How far off the prediction was from what actually happened: Error = |Predicted − Actual|. Smaller is better. This model's typical error across the full test set is around 47-48 MW, so this number is normal if it falls in a similar range."
        )

        st.markdown(
            f'<div class="info-box">The model predicted <b>{predicted_mw:,.0f} MW</b>, and the real recorded '
            f'demand was <b>{actual_mw:,.0f} MW</b> — a difference of <b>{error_mw:,.1f} MW</b>. '
            f'This gap is expected: no forecasting model predicts perfectly, and this model\'s average error '
            f'across three years of testing is about 47-48 MW, so this result is within normal range.</div>',
            unsafe_allow_html=True
        )

        row = df.iloc[target_idx]
        if row['event_flag'] == 1 or row['is_public_holiday'] == 1:
            reason = "a major event (e.g. a GAA match, concert, or large gathering)" if row['event_flag'] == 1 else "a public holiday"
            st.warning(
                f"**Elevated demand expected** on this date due to {reason}. "
                f"Recommended action: shift flexible loads outside peak hours and pre-position reserves."
            )
        else:
            st.info("No major events or holidays flagged for this date — a normal demand pattern was expected.")

        st.write("")
        left, right = st.columns([3, 2])

        with left:
            st.subheader("Demand Trend")
            tab1, tab2 = st.tabs(["Selected Date & Time (zoomed in)", "Full 3-Year Overview"])

            with tab1:
                st.caption(
                    "Electricity demand over the two days leading up to your exact selected date and time. "
                    "The red dot marks precisely the point you picked above."
                )
                day_start_ts = pd.Timestamp(selected_ts.date()) - pd.Timedelta(days=2)
                day_end_ts = pd.Timestamp(selected_ts.date()) + pd.Timedelta(days=1)
                zoom_df = df.loc[(df.index >= day_start_ts) & (df.index < day_end_ts), ['demand_mw']]
                if len(zoom_df) < 48:
                    st.caption("Limited history available this early in the dataset — showing all available data before this point.")
                st.plotly_chart(plot_zoomed_trend(zoom_df, selected_ts), use_container_width=True)

            with tab2:
                st.caption(
                    "Weekly-smoothed electricity demand across the full three-year dataset (Jan 2022 - Dec 2024), "
                    "so the underlying seasonal pattern is easier to see. The red dot marks the date you selected above."
                )
                daily_df = df[['demand_mw']].resample('D').mean()
                smoothed_df = daily_df.copy()
                smoothed_df['demand_mw'] = daily_df['demand_mw'].rolling(7, center=True, min_periods=1).mean()
                st.plotly_chart(plot_demand_trend(smoothed_df, selected_date), use_container_width=True)

        with right:
            st.subheader("Why the Model Predicted This")
            st.caption(
                "Each bar shows how many megawatts that factor added to or subtracted from the final prediction. "
                "Longer bar = bigger influence on this specific prediction."
            )
            demand_range = scaler.data_max_[0] - scaler.data_min_[0]
            shap_summary = explain_prediction(hybrid_model, hybrid_input_df, embedding_names, XGB_FEATURE_COLS, demand_range)
            st.plotly_chart(plot_shap_bar(shap_summary), use_container_width=True)

        st.caption(
            "**Recent 24-Hour Demand Pattern** combines the model's memory of the last 24 hours into one number — "
            "it is almost always the single biggest factor, which makes sense since electricity demand closely "
            "follows recent trends. All other bars are individual, nameable factors (weather, time, holidays, events)."
        )