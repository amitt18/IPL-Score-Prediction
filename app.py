from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "next_ball_model.joblib"
st.set_page_config(page_title="IPL Next-Ball Predictor", page_icon="🏏", layout="wide")
st.title("🏏 IPL Next-Ball Predictor")
st.write("Estimates the run outcome for the selected upcoming delivery from historical IPL data. It is a probability estimate, not a live feed or a guaranteed result.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("Model file not found. From this folder run: python train.py")
    st.stop()

model = load_model()

TEAMS = [
    "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans",
    "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians",
    "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bangalore",
    "Sunrisers Hyderabad"
]
VENUES = [
    "Wankhede Stadium", "Eden Gardens", "M Chinnaswamy Stadium",
    "MA Chidambaram Stadium", "Arun Jaitley Stadium",
    "Narendra Modi Stadium, Ahmedabad", "Rajiv Gandhi International Stadium",
    "Punjab Cricket Association IS Bindra Stadium", "Sawai Mansingh Stadium",
    "Dr DY Patil Sports Academy", "Unknown"
]

with st.form("prediction_form"):
    left, right = st.columns(2)
    with left:
        innings = st.selectbox("Innings", [1, 2])
        over = st.slider("Over number (0–19)", 0, 19, 9)
        ball = st.slider("Ball number in over", 1, 6, 3)
        venue = st.selectbox("Venue", VENUES)
        batting_team = st.selectbox("Batting team", TEAMS)
    with right:
        batter = st.text_input("Batter on strike", "V Kohli").strip()
        bowler = st.text_input("Bowler", "JJ Bumrah").strip()
        score = st.number_input("Current innings score (before this ball)", min_value=0, max_value=400, value=50)
        wickets = st.number_input("Wickets fallen (before this ball)", min_value=0, max_value=10, value=2)
        recent_runs = st.number_input("Runs in previous 6 balls", min_value=0, max_value=36, value=6)
        recent_wickets = st.number_input("Wickets in previous 6 balls", min_value=0, max_value=10, value=0)
    submitted = st.form_submit_button("Predict next delivery", use_container_width=True)

if submitted:
    balls_elapsed = over * 6 + ball - 1
    run_rate = (score * 6 / balls_elapsed) if balls_elapsed > 0 else 0.0
    row = pd.DataFrame([{
        "innings": innings, "overs": over, "ballnumber": ball,
        "balls_elapsed": balls_elapsed, "score_before_ball": score,
        "wickets_before_ball": wickets, "runs_last_6_balls": recent_runs,
        "wickets_last_6_balls": recent_wickets, "run_rate": run_rate,
        "batter": batter or "Unknown", "bowler": bowler or "Unknown",
        "BattingTeam": batting_team, "Venue": venue
    }])
    probabilities = model.predict_proba(row)[0]
    classes = model.named_steps["classifier"].classes_
    order = np.argsort(probabilities)[::-1]
    best_class = int(classes[order[0]])
    labels = {0: "Dot ball", 1: "1 run", 2: "2 runs", 3: "3 runs",
              4: "4 runs", 5: "5 runs", 6: "6 runs"}
    st.subheader(f"Most likely runs: {labels.get(best_class, str(best_class) + ' runs')}")
    st.caption(f"Model probability for this outcome: {probabilities[order[0]] * 100:.1f}%")
    chart = pd.DataFrame({
        "Outcome": [labels.get(int(classes[i]), f"{classes[i]} runs") for i in range(len(classes))],
        "Probability (%)": [round(float(p) * 100, 2) for p in probabilities]
    }).sort_values("Probability (%)", ascending=False)
    st.markdown("### Probability for each outcome")
    st.dataframe(chart, hide_index=True, use_container_width=True)
    st.bar_chart(chart.set_index("Outcome"))
    st.info("The model predicts run totals, not whether a run was a bye, leg bye, wide, or no-ball. Wicket probability needs a separately trained target. Probabilities may be poorly calibrated; evaluate the training report before relying on them.")

plot = ROOT / "eda_plots" / "run_distribution.png"
if plot.exists():
    st.markdown("---")
    st.subheader("Historical run distribution")
    st.image(str(plot), use_container_width=True)
