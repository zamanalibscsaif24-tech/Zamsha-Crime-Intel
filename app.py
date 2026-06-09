import io
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import mysql.connector
except Exception:
    mysql = None

try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None


APP_TITLE = "Crime Intelligence Platform"
APP_SUBTITLE = "Decision support for trends, hotspots, outcomes, and operational risk"
DATA_DIR = Path("data")
REPORT_DIR = Path("reports")

CITY_COORDS = {
    "Karachi": [24.8607, 67.0011],
    "Lahore": [31.5204, 74.3587],
    "Islamabad": [33.6844, 73.0479],
    "Rawalpindi": [33.5909, 73.0436],
    "Faisalabad": [31.4187, 73.0791],
    "Peshawar": [34.0151, 71.5249],
    "Multan": [30.1575, 71.5249],
    "Quetta": [30.1798, 66.9750],
    "Gujranwala": [32.1617, 74.1883],
    "Hyderabad": [25.3960, 68.3578],
    "Sialkot": [32.4925, 74.5310],
}

REQUIRED_COLUMNS = {
    "Case_ID",
    "Date",
    "City",
    "Crime_Type",
    "Outcome",
    "Crime_Severity",
}


st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="shield")

st.markdown(
    """
<style>
    :root {
        --ink: #111827;
        --muted: #64748b;
        --line: #e5e7eb;
        --panel: #ffffff;
        --accent: #2563eb;
        --danger: #dc2626;
        --ok: #059669;
        --warn: #d97706;
    }
    .main .block-container { padding-top: 1.25rem; max-width: 1500px; }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
        min-height: 110px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: .35rem;
        padding: .25rem .6rem;
        border: 1px solid var(--line);
        border-radius: 999px;
        font-size: .84rem;
        color: var(--ink);
        background: #f8fafc;
    }
    .section-note {
        color: var(--muted);
        font-size: .95rem;
        margin-top: -.35rem;
        margin-bottom: .75rem;
    }
    .stButton > button {
        border-radius: 7px;
        font-weight: 650;
    }
</style>
""",
    unsafe_allow_html=True,
)


@dataclass
class AppConfig:
    mysql_host: str = os.getenv("MYSQL_HOST", "localhost")
    mysql_user: str = os.getenv("MYSQL_USER", "root")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "")
    mysql_database: str = os.getenv("MYSQL_DATABASE", "crime_db")
    mongo_uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    mongo_database: str = os.getenv("MONGO_DATABASE", "crime_project")


class CrimeDataEngine:
    def __init__(self):
        self.config = AppConfig()
        self.df = pd.DataFrame()
        self.model = None
        self.model_features = []
        self.model_accuracy = None
        self.model_report = ""
        self.model_classes = []
        self.audit_events = []

    def audit(self, action, detail=""):
        self.audit_events.insert(
            0,
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "action": action,
                "detail": detail,
            },
        )
        self.audit_events = self.audit_events[:250]

    def load_demo_data(self, rows=2500, seed=42):
        rng = np.random.default_rng(seed)
        cities = list(CITY_COORDS.keys())
        crime_types = ["Theft", "Robbery", "Assault", "Fraud", "Burglary", "Cybercrime", "Kidnapping", "Homicide"]
        weapons = ["None", "Knife", "Firearm", "Blunt Object", "Digital Device", "Unknown"]
        locations = ["Residential", "Commercial", "Street", "Transit", "Online", "Industrial", "Public Space"]
        times = ["Morning", "Afternoon", "Evening", "Night"]
        outcomes = ["Under Investigation", "Arrest Made", "Closed", "Evidence Pending", "Court Trial"]
        genders = ["Male", "Female", "Unknown"]

        start = datetime(2019, 1, 1)
        dates = [start + timedelta(days=int(x)) for x in rng.integers(0, 2500, size=rows)]
        severity_base = rng.normal(5.2, 2.0, rows)
        night_boost = rng.choice([0, 0.5, 1.1], rows, p=[0.62, 0.26, 0.12])
        severity = np.clip(severity_base + night_boost, 1, 10).round(1)

        df = pd.DataFrame(
            {
                "Case_ID": [f"CASE-{i + 1:06d}" for i in range(rows)],
                "Date": dates,
                "City": rng.choice(cities, rows, p=[.19, .16, .1, .08, .09, .08, .08, .06, .08, .05, .03]),
                "Crime_Type": rng.choice(crime_types, rows, p=[.22, .14, .16, .12, .12, .12, .05, .07]),
                "Weapon_Used": rng.choice(weapons, rows, p=[.38, .16, .12, .11, .08, .15]),
                "Victim_Age": np.clip(rng.normal(34, 14, rows), 5, 90).round().astype(int),
                "Victim_Gender": rng.choice(genders, rows, p=[.53, .43, .04]),
                "Suspect_Age": np.clip(rng.normal(31, 11, rows), 12, 85).round().astype(int),
                "Suspect_Gender": rng.choice(genders, rows, p=[.69, .24, .07]),
                "Location_Type": rng.choice(locations, rows),
                "Time_of_Day": rng.choice(times, rows, p=[.21, .27, .26, .26]),
                "Reported_to_Police": rng.choice([1, 0], rows, p=[.82, .18]),
                "Outcome": rng.choice(outcomes, rows, p=[.34, .22, .2, .13, .11]),
                "Crime_Severity": severity,
                "Crime_Count": 1,
                "Days_Since_Previous_Crime": np.clip(rng.exponential(18, rows), 0, 180).round().astype(int),
            }
        )
        self.df = self.clean_dataframe(df)
        self.audit("Loaded demo dataset", f"{len(self.df):,} rows")
        return self.df

    def load_uploaded_file(self, uploaded_file):
        if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        self.df = self.clean_dataframe(df)
        self.audit("Loaded uploaded dataset", f"{uploaded_file.name}: {len(self.df):,} rows")
        return self.df

    def clean_dataframe(self, df):
        df = df.copy()
        df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

        rename_map = {
            "Crime_Typ": "Crime_Type",
            "Weapon_U": "Weapon_Used",
            "Victim_Ge": "Victim_Gender",
            "Suspect_A": "Suspect_Age",
            "Suspect_G": "Suspect_Gender",
            "Location_T": "Location_Type",
            "Time_of_D": "Time_of_Day",
            "Reported_t": "Reported_to_Police",
            "Crime_Sev": "Crime_Severity",
            "High_Seve": "High_Severity",
            "Crime_Cou": "Crime_Count",
            "Month_Na": "Month_Name",
            "Day_of_We": "Day_of_Week",
        }
        df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

        if "Case_ID" not in df.columns:
            df["Case_ID"] = [f"CASE-{i + 1:06d}" for i in range(len(df))]
        if "Date" not in df.columns:
            df["Date"] = pd.date_range(end=datetime.today(), periods=len(df), freq="D")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).reset_index(drop=True)

        defaults = {
            "City": "Unknown",
            "Crime_Type": "Unknown",
            "Weapon_Used": "Unknown",
            "Victim_Gender": "Unknown",
            "Suspect_Gender": "Unknown",
            "Location_Type": "Unknown",
            "Time_of_Day": "Unknown",
            "Outcome": "Unknown",
        }
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default
            df[col] = df[col].astype(str).str.strip().replace({"": default, "nan": default, "None": default})

        numeric_defaults = {
            "Victim_Age": np.nan,
            "Suspect_Age": np.nan,
            "Reported_to_Police": 0,
            "Crime_Severity": 5,
            "Crime_Count": 1,
            "Days_Since_Previous_Crime": np.nan,
        }
        for col, default in numeric_defaults.items():
            if col not in df.columns:
                df[col] = default
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Crime_Severity"] = df["Crime_Severity"].fillna(5).clip(1, 10)
        df["Crime_Count"] = df["Crime_Count"].fillna(1).clip(lower=1)
        df["Reported_to_Police"] = df["Reported_to_Police"].fillna(0).astype(int).clip(0, 1)
        df["High_Severity"] = (df["Crime_Severity"] >= 7).astype(int)
        df["Risk_Level"] = pd.cut(
            df["Crime_Severity"],
            bins=[0, 4.99, 7.49, 10],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        ).astype(str)
        df["Night_Flag"] = df["Time_of_Day"].str.lower().isin(["night", "evening"]).astype(int)
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Month_Name"] = df["Date"].dt.month_name()
        df["Day"] = df["Date"].dt.day
        df["Day_of_Week"] = df["Date"].dt.day_name()
        df["Quarter"] = "Q" + df["Date"].dt.quarter.astype(str)
        df["Season"] = df["Month"].apply(self.season_from_month)
        df["Resolved_Flag"] = df["Outcome"].str.lower().isin(["closed", "arrest made", "court trial"]).astype(int)
        df["Risk_Score"] = self.compute_risk_score(df)
        return df

    @staticmethod
    def season_from_month(month):
        if month in [12, 1, 2]:
            return "Winter"
        if month in [3, 4, 5]:
            return "Spring"
        if month in [6, 7, 8]:
            return "Summer"
        return "Autumn"

    @staticmethod
    def compute_risk_score(df):
        severity = df["Crime_Severity"].fillna(5) * 8
        high = df["High_Severity"].fillna(0) * 10
        night = df["Night_Flag"].fillna(0) * 7
        unreported = (1 - df["Reported_to_Police"].fillna(0)) * 8
        repeat = (180 - df["Days_Since_Previous_Crime"].fillna(90).clip(0, 180)) / 180 * 7
        return np.clip(severity + high + night + unreported + repeat, 0, 100).round(1)

    def filtered_data(self, cities, crime_types, years, risk_levels):
        df = self.df.copy()
        if cities:
            df = df[df["City"].isin(cities)]
        if crime_types:
            df = df[df["Crime_Type"].isin(crime_types)]
        if years:
            df = df[df["Year"].isin(years)]
        if risk_levels:
            df = df[df["Risk_Level"].isin(risk_levels)]
        return df

    def data_quality_report(self):
        if self.df.empty:
            return pd.DataFrame()
        rows = []
        for col in self.df.columns:
            rows.append(
                {
                    "column": col,
                    "missing": int(self.df[col].isna().sum()),
                    "missing_percent": round(float(self.df[col].isna().mean() * 100), 2),
                    "unique_values": int(self.df[col].nunique(dropna=True)),
                    "dtype": str(self.df[col].dtype),
                }
            )
        return pd.DataFrame(rows).sort_values(["missing_percent", "column"], ascending=[False, True])

    def detect_anomalies(self, df):
        if df.empty or len(df) < 20:
            return pd.DataFrame()
        features = df[["Crime_Severity", "Crime_Count", "Reported_to_Police", "High_Severity", "Night_Flag", "Risk_Score"]].fillna(0)
        model = IsolationForest(contamination=0.04, random_state=42)
        labels = model.fit_predict(features)
        scores = model.decision_function(features)
        out = df.copy()
        out["Anomaly"] = labels == -1
        out["Anomaly_Score"] = (-scores).round(4)
        return out[out["Anomaly"]].sort_values("Anomaly_Score", ascending=False)

    def train_model(self):
        if self.df.empty:
            raise ValueError("Load data before training the model.")
        df = self.df[self.df["Outcome"].notna()].copy()
        min_class = df["Outcome"].value_counts().min()
        stratify = df["Outcome"] if min_class >= 2 and df["Outcome"].nunique() > 1 else None
        features = [
            "City",
            "Crime_Type",
            "Weapon_Used",
            "Location_Type",
            "Time_of_Day",
            "Victim_Age",
            "Suspect_Age",
            "Reported_to_Police",
            "Crime_Severity",
            "High_Severity",
            "Risk_Score",
        ]
        X = df[features]
        y = df["Outcome"].astype(str)
        if y.nunique() < 2:
            raise ValueError("The Outcome column needs at least two classes for training.")
        preprocessor = ColumnTransformer(
            [
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    ["City", "Crime_Type", "Weapon_Used", "Location_Type", "Time_of_Day"],
                ),
                (
                    "num",
                    Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                    ["Victim_Age", "Suspect_Age", "Reported_to_Police", "Crime_Severity", "High_Severity", "Risk_Score"],
                ),
            ]
        )
        model = Pipeline(
            [
                ("prep", preprocessor),
                ("clf", RandomForestClassifier(n_estimators=180, min_samples_leaf=3, random_state=42, class_weight="balanced")),
            ]
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42, stratify=stratify)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        self.model = model
        self.model_features = features
        self.model_accuracy = round(float(accuracy_score(y_test, pred)), 4)
        self.model_report = classification_report(y_test, pred, zero_division=0)
        self.model_classes = sorted(y.unique().tolist())
        self.audit("Trained outcome model", f"Accuracy {self.model_accuracy}")
        return self.model_accuracy, self.model_report, confusion_matrix(y_test, pred, labels=self.model_classes)

    def predict(self, payload):
        if self.model is None:
            raise ValueError("Train the outcome model first.")
        return self.model.predict(pd.DataFrame([payload]))[0]

    def forecast_year(self, target_year):
        yearly = self.df.groupby("Year", as_index=False)["Crime_Count"].sum().rename(columns={"Crime_Count": "total_crimes"})
        if len(yearly) < 3:
            raise ValueError("Need at least three years of data for forecasting.")
        model = LinearRegression().fit(yearly[["Year"]], yearly["total_crimes"])
        pred = int(max(0, round(float(model.predict(pd.DataFrame({"Year": [target_year]}))[0]))))
        return yearly, pred

    def save_report(self, name, df):
        REPORT_DIR.mkdir(exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower()
        path = REPORT_DIR / f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(path, index=False)
        self.audit("Saved CSV report", str(path))
        return path

    def export_excel_bytes(self, sheets):
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=name[:31], index=False)
        bio.seek(0)
        return bio.getvalue()

    def mongo_status(self):
        if MongoClient is None:
            return False, "pymongo is not installed"
        try:
            client = MongoClient(self.config.mongo_uri, serverSelectionTimeoutMS=900)
            client.admin.command("ping")
            return True, "MongoDB connected"
        except Exception as exc:
            return False, str(exc)

    def mysql_status(self):
        if mysql is None or not hasattr(mysql, "connector"):
            return False, "mysql-connector-python is not installed"
        try:
            conn = mysql.connector.connect(
                host=self.config.mysql_host,
                user=self.config.mysql_user,
                password=self.config.mysql_password,
                database=self.config.mysql_database,
                connection_timeout=1,
            )
            conn.close()
            return True, "MySQL connected"
        except Exception as exc:
            return False, str(exc)


def init_state():
    if "engine" not in st.session_state:
        st.session_state.engine = CrimeDataEngine()
        st.session_state.engine.load_demo_data()


def sidebar_filters(engine):
    st.sidebar.title("Crime Intelligence")
    st.sidebar.caption("Advanced analytics workspace")
    role = st.sidebar.selectbox("Role", ["Admin", "Analyst", "Public Viewer"])
    page = st.sidebar.radio(
        "Workspace",
        [
            "Command Center",
            "Data Intake",
            "Hotspots & Trends",
            "Case Explorer",
            "Prediction Lab",
            "Forecasting",
            "Data Quality",
            "Reports & Exports",
            "System Health",
        ],
    )
    st.sidebar.divider()
    df = engine.df
    cities = st.sidebar.multiselect("Cities", clean_options(df, "City"))
    crime_types = st.sidebar.multiselect("Crime types", clean_options(df, "Crime_Type"))
    years = st.sidebar.multiselect("Years", sorted(df["Year"].dropna().astype(int).unique()) if not df.empty else [])
    risk_levels = st.sidebar.multiselect("Risk levels", ["High", "Medium", "Low"])
    return role, page, cities, crime_types, years, risk_levels

def clean_options(df, column):
    if df.empty or column not in df.columns:
        return []
    values = df[column].dropna().astype(str).str.strip()
    values = values[(values != "") & (values.str.lower() != "nan")]
    return sorted(values.unique().tolist())
def page_header(title, note=None):
    st.title(title)
    if note:
        st.markdown(f"<div class='section-note'>{note}</div>", unsafe_allow_html=True)


def render_kpis(df):
    c1, c2, c3, c4, c5 = st.columns(5)
    total = int(df["Crime_Count"].sum()) if not df.empty else 0
    high = int(df["High_Severity"].sum()) if not df.empty else 0
    resolved_rate = round(float(df["Resolved_Flag"].mean() * 100), 1) if not df.empty else 0
    avg_risk = round(float(df["Risk_Score"].mean()), 1) if not df.empty else 0
    top_city = df.groupby("City")["Crime_Count"].sum().sort_values(ascending=False).index[0] if not df.empty else "N/A"
    c1.metric("Total incidents", f"{total:,}")
    c2.metric("High severity", f"{high:,}")
    c3.metric("Resolved rate", f"{resolved_rate}%")
    c4.metric("Average risk", avg_risk)
    c5.metric("Top city", top_city)


def chart_bar(df, x, y, title, color="#2563eb"):
    return (
        alt.Chart(df)
        .mark_bar(color=color)
        .encode(
            x=alt.X(x, sort="-y", title=None),
            y=alt.Y(y, title=None),
            tooltip=list(df.columns),
        )
        .properties(height=340, title=title)
    )


def render_command_center(engine, df):
    page_header("Command Center", "Live operational summary with filters, risk scoring, and anomaly detection.")
    render_kpis(df)
    st.divider()
    left, right = st.columns([1.25, 1])
    with left:
        monthly = df.groupby(["Year", "Month"], as_index=False)["Crime_Count"].sum()
        if not monthly.empty:
            monthly["Period"] = pd.to_datetime(monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str) + "-01")
            chart = (
                alt.Chart(monthly)
                .mark_line(point=True, color="#2563eb")
                .encode(x=alt.X("Period:T", title="Month"), y=alt.Y("Crime_Count:Q", title="Incidents"), tooltip=["Period:T", "Crime_Count"])
                .properties(height=350, title="Incident Trend")
            )
            st.altair_chart(chart, width="stretch")
    with right:
        risk = df.groupby("Risk_Level", as_index=False)["Crime_Count"].sum()
        if not risk.empty:
            st.altair_chart(chart_bar(risk, "Risk_Level", "Crime_Count", "Risk Distribution", "#dc2626"), width="stretch")

    anomalies = engine.detect_anomalies(df)
    st.subheader("Priority Alerts")
    if anomalies.empty:
        st.info("No statistical anomalies detected in the current filter.")
    else:
        show_cols = ["Case_ID", "Date", "City", "Crime_Type", "Outcome", "Crime_Severity", "Risk_Score", "Anomaly_Score"]
        st.dataframe(anomalies[show_cols].head(25), width="stretch", hide_index=True)


def render_data_intake(engine):
    page_header("Data Intake", "Upload a file, preview cleaned data, or reset to a built-in demo dataset.")
    left, right = st.columns([1, 1])
    with left:
        uploaded = st.file_uploader("Upload crime dataset", type=["csv", "xlsx", "xls"])
        if uploaded and st.button("Load Uploaded Dataset", type="primary", width="stretch"):
            try:
                engine.load_uploaded_file(uploaded)
                st.success(f"Loaded and cleaned {len(engine.df):,} rows.")
            except Exception as exc:
                st.error(f"Upload failed: {exc}")
    with right:
        rows = st.slider("Demo rows", 500, 10000, 2500, 500)
        if st.button("Generate Demo Dataset", width="stretch"):
            engine.load_demo_data(rows=rows, seed=int(datetime.now().timestamp()) % 100000)
            st.success(f"Generated {len(engine.df):,} demo rows.")

    st.subheader("Cleaned Preview")
    st.dataframe(engine.df.head(200), width="stretch", hide_index=True)
    missing = REQUIRED_COLUMNS.difference(engine.df.columns)
    if missing:
        st.warning(f"Missing recommended columns: {', '.join(sorted(missing))}")
    else:
        st.success("Required analytical columns are present.")


def render_hotspots(df):
    page_header("Hotspots & Trends", "Compare geography, crime types, time patterns, and severity concentration.")
    c1, c2 = st.columns([1, 1])
    city = df.groupby("City", as_index=False)["Crime_Count"].sum().sort_values("Crime_Count", ascending=False).head(15)
    crime = df.groupby("Crime_Type", as_index=False)["Crime_Count"].sum().sort_values("Crime_Count", ascending=False)
    with c1:
        st.altair_chart(chart_bar(city, "City", "Crime_Count", "Top Cities"), width="stretch")
    with c2:
        st.altair_chart(chart_bar(crime, "Crime_Type", "Crime_Count", "Crime Types", "#059669"), width="stretch")

    map_df = city.copy()
    if not map_df.empty:
        map_df["lat"] = map_df["City"].map(lambda x: CITY_COORDS.get(x, [30.3753, 69.3451])[0])
        map_df["lon"] = map_df["City"].map(lambda x: CITY_COORDS.get(x, [30.3753, 69.3451])[1])
        map_df["size"] = np.clip(map_df["Crime_Count"] / max(map_df["Crime_Count"].max(), 1) * 1800, 120, 1800)
        st.subheader("Geospatial Distribution")
        st.map(map_df.rename(columns={"lat": "latitude", "lon": "longitude"}), latitude="latitude", longitude="longitude", size="size", zoom=4)

    heat = df.groupby(["Day_of_Week", "Time_of_Day"], as_index=False)["Crime_Count"].sum()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat["Day_of_Week"] = pd.Categorical(heat["Day_of_Week"], categories=order, ordered=True)
    chart = (
        alt.Chart(heat)
        .mark_rect()
        .encode(
            x=alt.X("Time_of_Day:N", title="Time"),
            y=alt.Y("Day_of_Week:N", title="Day"),
            color=alt.Color("Crime_Count:Q", scale=alt.Scale(scheme="orangered"), title="Incidents"),
            tooltip=["Day_of_Week", "Time_of_Day", "Crime_Count"],
        )
        .properties(height=320, title="Day and Time Heatmap")
    )
    st.altair_chart(chart, width="stretch")


def render_case_explorer(engine, df):
    page_header("Case Explorer", "Search, sort, and inspect individual cases with local notes.")
    query = st.text_input("Search case, city, crime type, or outcome")
    cases = df.copy()
    if query:
        q = query.lower()
        mask = cases[["Case_ID", "City", "Crime_Type", "Outcome"]].astype(str).apply(lambda s: s.str.lower().str.contains(q, na=False)).any(axis=1)
        cases = cases[mask]
    sort_col = st.selectbox("Sort by", ["Risk_Score", "Date", "Crime_Severity", "City"])
    cases = cases.sort_values(sort_col, ascending=False)
    st.dataframe(
        cases[["Case_ID", "Date", "City", "Crime_Type", "Outcome", "Risk_Level", "Risk_Score", "Crime_Severity"]].head(300),
        width="stretch",
        hide_index=True,
    )
    selected = st.selectbox("Open case", cases["Case_ID"].head(300).tolist() if not cases.empty else [])
    if selected:
        row = cases[cases["Case_ID"] == selected].iloc[0]
        left, right = st.columns([1, 1])
        with left:
            st.json(row[["Case_ID", "Date", "City", "Crime_Type", "Weapon_Used", "Location_Type", "Outcome", "Risk_Score"]].astype(str).to_dict())
        with right:
            note_key = f"note_{selected}"
            st.text_area("Case note", key=note_key, height=180)
            if st.button("Add Local Note", type="primary"):
                engine.audit("Added case note", selected)
                st.success("Note added to this session audit trail.")


def render_prediction_lab(engine, df):
    page_header("Prediction Lab", "Train a model and estimate likely case outcomes from structured inputs.")
    left, right = st.columns([.9, 1.1])
    with left:
        if st.button("Train Outcome Model", type="primary", width="stretch"):
            try:
                acc, report, matrix = engine.train_model()
                st.success(f"Model trained. Accuracy: {acc}")
                st.text(report)
                if engine.model_classes:
                    st.dataframe(pd.DataFrame(matrix, index=engine.model_classes, columns=engine.model_classes), width="stretch")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
        if engine.model_accuracy is not None:
            st.info(f"Current model accuracy: {engine.model_accuracy}")
            st.text(engine.model_report)
    with right:
        payload = {}
        payload["City"] = st.selectbox("City", clean_options(df, "City"))
        payload["Crime_Type"] = st.selectbox("Crime type", clean_options(df, "Crime_Type"))
        payload["Weapon_Used"] = st.selectbox("Weapon used", clean_options(df, "Weapon_Used"))
        payload["Location_Type"] = st.selectbox("Location type", clean_options(df, "Location_Type"))
        payload["Time_of_Day"] = st.selectbox("Time of day", clean_options(df, "Time_of_Day"))
        c1, c2 = st.columns(2)
        payload["Victim_Age"] = c1.number_input("Victim age", 1, 100, 30)
        payload["Suspect_Age"] = c2.number_input("Suspect age", 1, 100, 28)
        payload["Reported_to_Police"] = st.selectbox("Reported to police", [1, 0])
        payload["Crime_Severity"] = st.slider("Crime severity", 1.0, 10.0, 6.0, 0.1)
        payload["High_Severity"] = int(payload["Crime_Severity"] >= 7)
        temp = pd.DataFrame([payload])
        temp["Night_Flag"] = temp["Time_of_Day"].str.lower().isin(["night", "evening"]).astype(int)
        temp["Days_Since_Previous_Crime"] = 30
        payload["Risk_Score"] = float(engine.compute_risk_score(temp)[0])
        st.metric("Computed risk score", payload["Risk_Score"])
        if st.button("Predict Outcome", width="stretch"):
            try:
                st.success(f"Predicted outcome: {engine.predict(payload)}")
            except Exception as exc:
                st.error(str(exc))


def render_forecasting(engine):
    page_header("Forecasting", "Simple baseline forecast from historical yearly totals.")
    target_year = st.number_input("Target year", 2026, 2055, 2027)
    if st.button("Run Forecast", type="primary"):
        try:
            yearly, pred = engine.forecast_year(target_year)
            st.metric(f"Forecast for {target_year}", f"{pred:,} incidents")
            chart = (
                alt.Chart(yearly)
                .mark_line(point=True, color="#2563eb")
                .encode(x=alt.X("Year:O", title="Year"), y=alt.Y("total_crimes:Q", title="Incidents"), tooltip=["Year", "total_crimes"])
                .properties(height=360)
            )
            st.altair_chart(chart, width="stretch")
        except Exception as exc:
            st.error(str(exc))


def render_quality(engine):
    page_header("Data Quality", "Column completeness, uniqueness, and anomaly review.")
    st.dataframe(engine.data_quality_report(), width="stretch", hide_index=True)
    st.subheader("Duplicate Case IDs")
    dupes = engine.df[engine.df["Case_ID"].duplicated(keep=False)].sort_values("Case_ID")
    if dupes.empty:
        st.success("No duplicate Case_ID values found.")
    else:
        st.dataframe(dupes, width="stretch", hide_index=True)


def render_reports(engine, df):
    page_header("Reports & Exports", "Generate analyst-friendly tables and downloadable workbooks.")
    reports = {
        "City summary": df.groupby("City", as_index=False)["Crime_Count"].sum().sort_values("Crime_Count", ascending=False),
        "Crime type summary": df.groupby("Crime_Type", as_index=False)["Crime_Count"].sum().sort_values("Crime_Count", ascending=False),
        "Outcome summary": df.groupby("Outcome", as_index=False)["Crime_Count"].sum().sort_values("Crime_Count", ascending=False),
        "High risk cases": df.sort_values("Risk_Score", ascending=False).head(200),
    }
    choice = st.selectbox("Report", list(reports.keys()))
    report_df = reports[choice]
    st.dataframe(report_df, width="stretch", hide_index=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.download_button(
            "Download selected CSV",
            report_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{choice.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            width="stretch",
        )
    with c2:
        st.download_button(
            "Download full Excel workbook",
            engine.export_excel_bytes(reports),
            file_name=f"crime_intelligence_workbook_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
    if st.button("Save selected report to reports folder"):
        path = engine.save_report(choice, report_df)
        st.success(f"Saved: {path}")


def render_system_health(engine):
    page_header("System Health", "Configuration status, optional database checks, and audit trail.")
    my_ok, my_msg = engine.mysql_status()
    mo_ok, mo_msg = engine.mongo_status()
    c1, c2, c3 = st.columns(3)
    c1.metric("MySQL", "Connected" if my_ok else "Offline")
    c2.metric("MongoDB", "Connected" if mo_ok else "Offline")
    c3.metric("Rows in memory", f"{len(engine.df):,}")
    st.caption(f"MySQL: {my_msg}")
    st.caption(f"MongoDB: {mo_msg}")
    st.subheader("Environment-driven configuration")
    st.code(
        "MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE, MONGO_URI, MONGO_DATABASE",
        language="text",
    )
    st.subheader("Session Audit Trail")
    st.dataframe(pd.DataFrame(engine.audit_events), width="stretch", hide_index=True)


def main():
    init_state()
    engine = st.session_state.engine
    role, page, cities, crime_types, years, risk_levels = sidebar_filters(engine)
    df = engine.filtered_data(cities, crime_types, years, risk_levels)

    st.markdown(f"<span class='status-pill'>{role}</span>", unsafe_allow_html=True)

    if page == "Command Center":
        render_command_center(engine, df)
    elif page == "Data Intake":
        render_data_intake(engine)
    elif page == "Hotspots & Trends":
        render_hotspots(df)
    elif page == "Case Explorer":
        render_case_explorer(engine, df)
    elif page == "Prediction Lab":
        render_prediction_lab(engine, df)
    elif page == "Forecasting":
        render_forecasting(engine)
    elif page == "Data Quality":
        render_quality(engine)
    elif page == "Reports & Exports":
        render_reports(engine, df)
    elif page == "System Health":
        render_system_health(engine)


if __name__ == "__main__":
    main()

