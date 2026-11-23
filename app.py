import os
from datetime import timedelta
import psycopg2
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Smart Parking Demand & Event Impact",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_database_url():
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url
    try:
        return st.secrets["DATABASE_URL"]
    except Exception:
        return None


def query_df(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [desc[0] for desc in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def format_ampm(hour_int):
    if hour_int == 0:
        return "12 AM"
    elif hour_int < 12:
        return f"{hour_int} AM"
    elif hour_int == 12:
        return "12 PM"
    else:
        return f"{hour_int - 12} PM"


@st.cache_data(ttl=300)
def run_analytics_pipeline():
    db_url = get_database_url()
    if not db_url:
        st.error("DATABASE_URL is not configured. Add it to your .env file or deployment secrets.")
        st.stop()

    conn = psycopg2.connect(db_url)
    lot_df = query_df(conn, "SELECT * FROM lot_config LIMIT 1")
    raw_events = query_df(conn, "SELECT * FROM raw_gate_events ORDER BY gate_timestamp")
    payments = query_df(conn, "SELECT * FROM payment_transactions ORDER BY paid_at")
    local_events = query_df(conn, "SELECT * FROM local_events")
    weather = query_df(conn, "SELECT * FROM weather_logs")
    conn.close()

    capacity = int(lot_df.iloc[0]["capacity"])
    entry_offset = int(lot_df.iloc[0]["entry_gate_offset_sec"])
    exit_offset = int(lot_df.iloc[0]["exit_gate_offset_sec"])

    # ==========================================
    # 1. CLOCK SYNCHRONIZATION
    # ==========================================
    events = raw_events.copy()
    events["gate_timestamp"] = pd.to_datetime(events["gate_timestamp"])
    events["corrected_ts"] = np.where(
        events["event_type"] == "ENTRY",
        events["gate_timestamp"] - pd.to_timedelta(entry_offset, unit="s"),
        events["gate_timestamp"] - pd.to_timedelta(exit_offset, unit="s"),
    )
    events["corrected_ts"] = pd.to_datetime(events["corrected_ts"])
    events = events.sort_values(["vehicle_plate", "corrected_ts"]).reset_index(drop=True)

    # ==========================================
    # 2. SEQUENCE CLEANING & SESSION RECONCILIATION
    # ==========================================
    sessions = []
    stats = {
        "raw_events": len(events),
        "matched_pairs": 0,
        "unmatched_exits_imputed": 0,
        "orphan_entries_closed_via_payment": 0,
        "orphan_entries_closed_via_median": 0,
        "negative_durations_repaired": 0,
    }

    clean_durations = []
    for _, group in events.groupby("vehicle_plate"):
        prev_entry = None
        for _, row in group.iterrows():
            if row["event_type"] == "ENTRY":
                prev_entry = row["corrected_ts"]
            elif row["event_type"] == "EXIT" and prev_entry is not None:
                diff = (row["corrected_ts"] - prev_entry).total_seconds() / 60.0
                if 1 <= diff <= 1440:
                    clean_durations.append(diff)
                prev_entry = None

    median_stay_mins = float(np.median(clean_durations)) if clean_durations else 120.0

    payments["paid_at"] = pd.to_datetime(payments["paid_at"])
    for plate, group in events.groupby("vehicle_plate"):
        open_entry = None
        plate_payments = payments[payments["vehicle_plate"] == plate]["paid_at"].tolist()

        for _, row in group.iterrows():
            ts = row["corrected_ts"]
            if row["event_type"] == "ENTRY":
                if open_entry is not None:
                    matching_pay = [p for p in plate_payments if open_entry < p < ts]
                    if matching_pay:
                        est_exit = matching_pay[0] + timedelta(minutes=3)
                        stats["orphan_entries_closed_via_payment"] += 1
                    else:
                        est_exit = min(open_entry + timedelta(minutes=median_stay_mins), ts - timedelta(minutes=1))
                        stats["orphan_entries_closed_via_median"] += 1
                    sessions.append((plate, open_entry, est_exit, "IMPUTED_EXIT"))
                open_entry = ts

            elif row["event_type"] == "EXIT":
                if open_entry is None:
                    matching_pay = [p for p in plate_payments if (ts - timedelta(hours=12)) <= p <= ts]
                    if matching_pay:
                        est_entry = matching_pay[-1] - timedelta(minutes=median_stay_mins)
                    else:
                        est_entry = ts - timedelta(minutes=median_stay_mins)
                    sessions.append((plate, est_entry, ts, "IMPUTED_ENTRY"))
                    stats["unmatched_exits_imputed"] += 1
                else:
                    if ts < open_entry:
                        ts = open_entry + timedelta(minutes=5)
                        stats["negative_durations_repaired"] += 1
                    sessions.append((plate, open_entry, ts, "CLEAN"))
                    stats["matched_pairs"] += 1
                    open_entry = None

        if open_entry is not None:
            matching_pay = [p for p in plate_payments if open_entry < p <= (open_entry + timedelta(hours=16))]
            if matching_pay:
                est_exit = matching_pay[0] + timedelta(minutes=3)
                stats["orphan_entries_closed_via_payment"] += 1
            else:
                est_exit = open_entry + timedelta(minutes=median_stay_mins)
                stats["orphan_entries_closed_via_median"] += 1
            sessions.append((plate, open_entry, est_exit, "IMPUTED_EXIT"))

    sessions_df = pd.DataFrame(sessions, columns=["plate", "entry_ts", "exit_ts", "status"])
    sessions_df["dwell_mins"] = ((sessions_df["exit_ts"] - sessions_df["entry_ts"]).dt.total_seconds() / 60).round(1)

    # Build cleaned event sequence timeline (Entries & Exits enriched with day of week, AM/PM, weather, event flag)
    cleaned_entries = sessions_df[["plate", "entry_ts", "status"]].rename(columns={"entry_ts": "ts"})
    cleaned_entries["event_type"] = "ENTRY"
    cleaned_exits = sessions_df[["plate", "exit_ts", "status"]].rename(columns={"exit_ts": "ts"})
    cleaned_exits["event_type"] = "EXIT"

    seq_df = pd.concat([cleaned_entries, cleaned_exits], ignore_index=True)
    seq_df["date"] = seq_df["ts"].dt.date
    seq_df["hour"] = seq_df["ts"].dt.hour
    seq_df["ampm_label"] = seq_df["hour"].apply(format_ampm)
    seq_df["day_of_week"] = seq_df["ts"].dt.day_name()
    seq_df["is_weekend"] = seq_df["ts"].dt.dayofweek >= 5
    seq_df["log_hour"] = seq_df["ts"].dt.floor("h")

    local_events["event_date"] = pd.to_datetime(local_events["event_date"]).dt.date
    weather["log_hour"] = pd.to_datetime(weather["log_hour"])

    seq_df = seq_df.merge(local_events, left_on="date", right_on="event_date", how="left")
    seq_df["is_event_day"] = seq_df["event_name"].notna()
    seq_df = seq_df.merge(weather, on="log_hour", how="left")
    seq_df["condition"] = seq_df["condition"].fillna("Clear")

    # ==========================================
    # 3. OCCUPANCY ESTIMATION (15-Minute Intervals)
    # ==========================================
    min_ts = sessions_df["entry_ts"].min().floor("D")
    max_ts = sessions_df["exit_ts"].max().ceil("D")
    grid = pd.date_range(min_ts, max_ts, freq="15min")

    deltas = []
    for _, s in sessions_df.iterrows():
        deltas.append((s["entry_ts"].floor("15min"), 1))
        deltas.append((s["exit_ts"].floor("15min"), -1))

    delta_df = pd.DataFrame(deltas, columns=["ts", "delta"]).groupby("ts")["delta"].sum()
    occ_series = delta_df.reindex(grid, fill_value=0).cumsum().clip(lower=0, upper=capacity)

    occ_df = pd.DataFrame({"timestamp": occ_series.index, "occupancy": occ_series.values})
    occ_df["utilization_pct"] = (occ_df["occupancy"] / capacity * 100).round(1)
    occ_df["date"] = occ_df["timestamp"].dt.date
    occ_df["hour"] = occ_df["timestamp"].dt.hour
    occ_df["ampm_label"] = occ_df["hour"].apply(format_ampm)
    occ_df["day_of_week"] = occ_df["timestamp"].dt.day_name()

    occ_df = occ_df.merge(local_events, left_on="date", right_on="event_date", how="left")
    occ_df["is_event_day"] = occ_df["event_name"].notna()
    occ_df["day_type"] = np.where(occ_df["is_event_day"], "Event Day", "Non-Event Day")

    occ_df["log_hour"] = occ_df["timestamp"].dt.floor("h")
    occ_df = occ_df.merge(weather, on="log_hour", how="left")
    occ_df["condition"] = occ_df["condition"].fillna("Clear")

    return {
        "capacity": capacity,
        "exit_offset": exit_offset,
        "median_stay_mins": round(median_stay_mins, 1),
        "stats": stats,
        "sessions_df": sessions_df,
        "seq_df": seq_df,
        "occ_df": occ_df,
    }


data = run_analytics_pipeline()
occ_df = data["occ_df"]
seq_df = data["seq_df"]
sessions_df = data["sessions_df"]
capacity = data["capacity"]
stats = data["stats"]

ORDERED_HOURS = list(range(24))
ORDERED_AMPM = [format_ampm(h) for h in ORDERED_HOURS]
DAYS_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ==========================================
# SIDEBAR FILTERS (Days of Week, Weather, Peak Threshold)
# ==========================================
st.sidebar.header("Sequence & Demand Filters")

day_group = st.sidebar.radio(
    "Day Type Preset",
    options=["All Days", "Weekdays (Mon–Fri)", "Weekends (Sat–Sun)", "Custom Days"],
    index=1,
)

if day_group == "Weekdays (Mon–Fri)":
    default_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
elif day_group == "Weekends (Sat–Sun)":
    default_days = ["Saturday", "Sunday"]
else:
    default_days = DAYS_ORDER

selected_days = st.sidebar.multiselect(
    "Days of Week",
    options=DAYS_ORDER,
    default=default_days,
)

weather_filter = st.sidebar.multiselect(
    "Weather Condition Filter (vs Normal Clear Baseline)",
    options=["Clear", "Cloudy", "Rain"],
    default=["Rain"],
    format_func=lambda w: {"Clear": "Normal / Clear Day", "Cloudy": "Cloudy Day", "Rain": "Rainy Day"}[w],
)

event_flow_type = st.sidebar.selectbox(
    "Gate Sequence Flow",
    options=["ALL", "ENTRY", "EXIT"],
    format_func=lambda x: {"ALL": "All Cleaned Events (Entry + Exit)", "ENTRY": "Entries Only", "EXIT": "Exits Only"}[x],
)

peak_threshold_pct = st.sidebar.slider("Peak Occupancy Threshold (%)", min_value=50, max_value=95, value=75, step=5)

if st.sidebar.button("Refresh Data from Neon DB"):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# HEADER & TOP KPI METRICS
# ==========================================
st.title("Smart Parking Demand & Event Impact 🅿️📊")
st.markdown(
    "Clock-synchronized sequence reconstruction, continuous occupancy estimation, and **Event vs. Non-Event** demand analysis."
)

event_avg = occ_df[occ_df["is_event_day"]]["occupancy"].mean()
non_event_avg = occ_df[~occ_df["is_event_day"]]["occupancy"].mean()
lift_pct = ((event_avg - non_event_avg) / max(1, non_event_avg)) * 100
peak_intervals = (occ_df["utilization_pct"] >= peak_threshold_pct).sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Cleaned Sequences", f"{len(sessions_df):,}", f"{stats['unmatched_exits_imputed']} exits imputed")
k2.metric("Avg Event-Day Occupancy", f"{event_avg:.1f} / {capacity}", f"+{lift_pct:.1f}% vs Normal")
k3.metric("Avg Non-Event Occupancy", f"{non_event_avg:.1f} / {capacity}")
k4.metric(f"Peak Intervals (≥{peak_threshold_pct}%)", f"{peak_intervals} slots", "15-min intervals")

st.markdown("---")

# ==========================================
# SECTION 1: CLEAN EVENT SEQUENCES (NORMAL DAY VS FILTERED SEQUENCE)
# ==========================================
st.header("1. Clean Event Sequences: Normal Day vs. Filtered Sequence")
st.caption(
    "Compares the baseline normal day event rate (Non-Event + Clear weather) against your selected Day-of-Week & Weather filter across AM/PM hours."
)

# Filter sequence flow if Entry-only or Exit-only selected
base_seq = seq_df if event_flow_type == "ALL" else seq_df[seq_df["event_type"] == event_flow_type]

# Baseline: Normal Day (Non-Event Day & Clear Weather across selected days or all days)
normal_seq = base_seq[(~base_seq["is_event_day"]) & (base_seq["condition"] == "Clear")]
normal_days_count = max(1, normal_seq["date"].nunique())
normal_hourly = (
    normal_seq.groupby("hour")
    .size()
    .reindex(ORDERED_HOURS, fill_value=0)
    / normal_days_count
).reset_index(name="avg_events_per_day")
normal_hourly["Sequence"] = "Baseline: Normal Day (Clear, Non-Event)"

# Filtered Sequence: Matches selected Days of Week AND selected Weather Conditions
filtered_seq = base_seq[
    (base_seq["day_of_week"].isin(selected_days)) & (base_seq["condition"].isin(weather_filter))
]
filtered_days_count = max(1, filtered_seq["date"].nunique())
filtered_hourly = (
    filtered_seq.groupby("hour")
    .size()
    .reindex(ORDERED_HOURS, fill_value=0)
    / filtered_days_count
).reset_index(name="avg_events_per_day")
weather_label = "+".join(weather_filter) if weather_filter else "None"
filtered_hourly["Sequence"] = f"Filtered ({day_group} | {weather_label})"

seq_compare_df = pd.concat([normal_hourly, filtered_hourly], ignore_index=True)
seq_compare_df["Time (AM/PM)"] = seq_compare_df["hour"].apply(format_ampm)

c_seq1, c_seq2 = st.columns([1.4, 1])

# ==========================================
# LEFT CHART: ORIGINAL NORMAL VS FILTERED LINE GRAPH
# ==========================================
with c_seq1:
    fig_seq = px.line(
        seq_compare_df,
        x="Time (AM/PM)",
        y="avg_events_per_day",
        color="Sequence",
        markers=True,
        color_discrete_map={
            "Baseline: Normal Day (Clear, Non-Event)": "#1f78b4",
            f"Filtered ({day_group} | {weather_label})": "#e31a1c",
        },
        labels={
            "avg_events_per_day": "Avg Cleaned Gate Events / Day",
            "Time (AM/PM)": "Time of Day (AM / PM)",
        },
        title="Hourly Gate Sequence Rate (AM / PM): Normal Day vs. Filtered Condition",
        height=400,
    )
    fig_seq.update_xaxes(categoryorder="array", categoryarray=ORDERED_AMPM)
    fig_seq.update_layout(
        margin=dict(l=10, r=10, t=80, b=10),
        title=dict(y=0.95, x=0),
        legend=dict(
            title_text="",
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
    )
    st.plotly_chart(fig_seq, use_container_width=True)

# ==========================================
# RIGHT CHART: IMPROVED HEATMAP (DAY OF WEEK x AM/PM)
# ==========================================
with c_seq2:
    active_days = selected_days if selected_days else DAYS_ORDER
    subset_seq = base_seq[base_seq["day_of_week"].isin(active_days)]

    # Normalize by weekday occurrences so comparison across days is fair
    day_counts = subset_seq.groupby("day_of_week")["date"].nunique().to_dict()
    dow_hour_df = (
        subset_seq.groupby(["day_of_week", "hour"])
        .size()
        .reset_index(name="total_events")
    )
    dow_hour_df["avg_events"] = dow_hour_df.apply(
        lambda r: round(r["total_events"] / max(1, day_counts.get(r["day_of_week"], 1)), 1),
        axis=1,
    )

    pivot_df = (
        dow_hour_df.pivot(index="day_of_week", columns="hour", values="avg_events")
        .reindex(index=[d for d in DAYS_ORDER if d in active_days], columns=range(6, 23))
        .fillna(0)
    )
    pivot_df.columns = [format_ampm(h) for h in pivot_df.columns]

    fig_heatmap = px.imshow(
        pivot_df,
        text_auto=".0f",
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="Time (AM/PM)", y="Day of Week", color="Avg Events"),
        title="Sequence Intensity Heatmap by Day & Time",
        height=400,
    )
    fig_heatmap.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(tickangle=-45),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
st.markdown("---")

# ==========================================
# SECTION 2: ESTIMATED OCCUPANCY
# ==========================================
st.header("2. Estimated Lot Occupancy (15-Minute Resolution)")
st.caption(
    f"Reconstructed from synchronized entry/exit timestamps using sweep-line delta aggregation and clamped at physical lot capacity ({capacity} spaces)."
)

occ_plot_df = occ_df[occ_df["day_of_week"].isin(selected_days if selected_days else DAYS_ORDER)].copy()

fig_occ = go.Figure()
fig_occ.add_trace(
    go.Scatter(
        x=occ_plot_df["timestamp"],
        y=occ_plot_df["occupancy"],
        mode="lines",
        name="Estimated Occupancy",
        line=dict(color="#2b8cbe", width=2),
        fill="tozeroy",
        fillcolor="rgba(43, 140, 190, 0.15)",
    )
)

# Highlight Event Day intervals
event_occ_points = occ_plot_df[occ_plot_df["is_event_day"]]
fig_occ.add_trace(
    go.Scatter(
        x=event_occ_points["timestamp"],
        y=event_occ_points["occupancy"],
        mode="markers",
        name="Event Day Intervals",
        marker=dict(color="#d95f02", size=5),
    )
)

fig_occ.add_hline(
    y=capacity,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Max Capacity ({capacity})",
)
fig_occ.add_hline(
    y=capacity * (peak_threshold_pct / 100),
    line_dash="dot",
    line_color="green",
    annotation_text=f"Peak Threshold ({peak_threshold_pct}%)",
)
fig_occ.update_layout(
    xaxis_title="Timeline",
    yaxis_title="Estimated Vehicles in Lot",
    height=380,
    margin=dict(l=10, r=10, t=20, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(fig_occ, use_container_width=True)

st.markdown("---")

# ==========================================
# SECTION 3: PEAK PERIODS, EVENT VS NON-EVENT & LIMITATIONS
# ==========================================
st.header("3. Peak Periods, Event vs. Non-Event Comparison & Documented Limitations")

p_left, p_right = st.columns(2)

with p_left:
    st.subheader("Event vs. Non-Event Hourly Occupancy (AM / PM)")
    hourly_comp = (
        occ_df.groupby(["hour", "day_type"])["occupancy"]
        .mean()
        .reset_index()
    )
    hourly_comp["Time (AM/PM)"] = hourly_comp["hour"].apply(format_ampm)

    fig_comp = px.bar(
        hourly_comp,
        x="Time (AM/PM)",
        y="occupancy",
        color="day_type",
        barmode="group",
        color_discrete_map={"Event Day": "#d95f02", "Non-Event Day": "#1f78b4"},
        labels={"occupancy": "Mean Vehicles", "day_type": "Day Category"},
        height=400,
    )
    fig_comp.add_hline(
        y=capacity * (peak_threshold_pct / 100),
        line_dash="dot",
        line_color="green",
        annotation_text=f"Peak Threshold ({peak_threshold_pct}%)",
    )
    fig_comp.update_xaxes(categoryorder="array", categoryarray=ORDERED_AMPM)
    fig_comp.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig_comp, use_container_width=True)

with p_right:
    st.subheader(f"Identified Peak Windows (≥ {peak_threshold_pct}% Capacity)")
    peak_df = (
        occ_df[occ_df["utilization_pct"] >= peak_threshold_pct]
        .groupby(["hour", "day_type"])
        .size()
        .reset_index(name="peak_intervals")
    )
    peak_df["Time (AM/PM)"] = peak_df["hour"].apply(format_ampm)

    if peak_df.empty:
        st.info(f"No 15-minute intervals exceeded {peak_threshold_pct}% capacity. Lower the sidebar threshold slider to inspect near-peak periods.")
    else:
        fig_peak = px.bar(
            peak_df,
            x="Time (AM/PM)",
            y="peak_intervals",
            color="day_type",
            color_discrete_map={"Event Day": "#d95f02", "Non-Event Day": "#1f78b4"},
            labels={"peak_intervals": "15-Min Peak Intervals Count", "day_type": "Day Category"},
            height=400,
        )
        fig_peak.update_xaxes(categoryorder="array", categoryarray=ORDERED_AMPM)
        fig_peak.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        st.plotly_chart(fig_peak, use_container_width=True)

# Documented Limitations Visual Summary
st.subheader("Documented Methodology & Data Limitations")
lim1, lim2, lim3, lim4 = st.columns(4)

with lim1:
    st.warning(
        f"**1. Imputed Dwell Time Bias**\n\n"
        f"{stats['unmatched_exits_imputed']} unmatched exits without payment links rely on the global median stay "
        f"({data['median_stay_mins']} mins), which can smooth out rapid turnover during event surges."
    )
with lim2:
    st.info(
        f"**2. Static Clock Drift Offset**\n\n"
        f"Gate synchronization applies a fixed `{data['exit_offset']}s` exit offset. Progressive hardware clock drift "
        f"over multi-week operations requires periodic NTP recalibration."
    )
with lim3:
    st.info(
        f"**3. Capacity Clamping (Unobserved Spillover)**\n\n"
        f"Occupancy is capped at physical lot capacity ({capacity} spaces). Vehicles turned away when the lot is full "
        f"represent censored demand not captured by gate sensors."
    )
with lim4:
    st.info(
        "**4. Event vs. Commuter Confounding**\n\n"
        "Local event schedules are joined at daily granularity; overlapping weekday office traffic and localized "
        "rain cells can confound pure event attribution."
    )