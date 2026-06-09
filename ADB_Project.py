import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
from pymongo import MongoClient
from pathlib import Path
from datetime import datetime
from decimal import Decimal
import os
import hashlib
import io
import random
import pydeck as pdk

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import altair as alt

# ==========================================================
# CONFIG & ZAMSHA BRANDING
# ==========================================================
APP_TITLE = "Crime Intelligence - Zamsha Enterprise"

MYSQL_CONFIG = {"host": "localhost", "user": "root", "password": "4510145424465", "database": "crime_db"}
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "crime_project"
MONGO_COLLECTION_REPORTS = "crime_reports"
MONGO_COLLECTION_EVIDENCE = "case_evidence"

CITY_COORDS = {
    "Karachi": [24.8607, 67.0011], "Lahore": [31.5204, 74.3587], "Islamabad": [33.6844, 73.0479],
    "Rawalpindi": [33.5909, 73.0436], "Faisalabad": [31.4187, 73.0791], "Peshawar": [34.0151, 71.5249],
    "Multan": [30.1575, 71.5249], "Quetta": [30.1798, 66.9750], "Gujranwala": [32.1617, 74.1883],
    "Hyderabad": [25.3960, 68.3578], "Sialkot": [32.4925, 74.5310]
}

st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🛡️")

# Zamsha Premium Dark Mode CSS
st.markdown("""
<style>
    body { background-color: #0f172a; color: #f8fafc; }
    .stApp { background-color: #0f172a; color: #f8fafc; }
    .metric-card { 
        background: rgba(30, 41, 59, 0.7); 
        backdrop-filter: blur(10px); 
        padding: 1.5rem; 
        border-radius: 12px; 
        border: 1px solid rgba(255,255,255,0.05); 
        border-top: 4px solid #38bdf8; 
        margin-bottom: 1rem; 
    }
    .metric-val { font-size: 2.5rem; font-weight: bold; color: #f8fafc; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
    
    /* Buttons */
    div.stButton > button { 
        background: linear-gradient(90deg, #0ea5e9, #2563eb); 
        color: white; border: none; border-radius: 8px; font-weight: bold; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); transition: 0.3s;
    }
    div.stButton > button:hover { background: linear-gradient(90deg, #38bdf8, #3b82f6); box-shadow: 0 6px 12px rgba(0,0,0,0.5); }
    
    /* Headers */
    h1 { color: #f8fafc; font-family: 'Inter', sans-serif; border-bottom: 1px solid #334155; padding-bottom: 0.5rem; margin-bottom: 2rem; }
    h2, h3 { color: #e2e8f0; font-family: 'Inter', sans-serif; }
    
    /* Zamsha Branding Classes */
    .zamsha-footer {
        position: fixed; left: 0; bottom: 0; width: 100%; text-align: center;
        background: rgba(15, 23, 42, 0.95); padding: 10px; border-top: 1px solid #334155;
        color: #94a3b8; font-family: monospace; z-index: 1000;
        backdrop-filter: blur(5px);
    }
    .zamsha-sidebar {
        margin-top: 30px; margin-bottom: 30px; padding: 15px; border-radius: 10px;
        background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #334155;
        text-align: center; color: #38bdf8; font-weight: bold; font-family: 'Inter', sans-serif;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

# Global Footer Injection
st.markdown("<div class='zamsha-footer'>🚀 Developed by Zamsha developers (Zaman Shar) | Enterprise Advanced Database Project</div>", unsafe_allow_html=True)

# ==========================================================
# ENTERPRISE ENGINE
# ==========================================================
class EnterpriseEngine:
    def __init__(self):
        self.mysql_conn = None
        self.mongo_client = None
        self.mongo_collection_reports = None
        self.mongo_collection_evidence = None
        self.outcome_model = None
        self.outcome_model_features = []
        self.outcome_model_accuracy = None

    def get_fresh_mysql_conn(self):
        # Always create a new connection to guarantee 100% stability at presentation time
        conn = mysql.connector.connect(host=MYSQL_CONFIG["host"], user=MYSQL_CONFIG["user"], password=MYSQL_CONFIG["password"])
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']}")
        cursor.close()
        conn.database = MYSQL_CONFIG['database']
        return conn

    def connect_mongo(self):
        if self.mongo_client is None:
            self.mongo_client = MongoClient(MONGO_URI)
            self.mongo_collection_reports = self.mongo_client[MONGO_DB][MONGO_COLLECTION_REPORTS]
            self.mongo_collection_evidence = self.mongo_client[MONGO_DB][MONGO_COLLECTION_EVIDENCE]

    def mysql_query(self, query, params=None):
        conn = self.get_fresh_mysql_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params or ())
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [{k: (float(v) if isinstance(v, Decimal) else v) for k, v in r.items()} for r in rows]

    def mysql_execute(self, query, params=None):
        conn = self.get_fresh_mysql_conn()
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        conn.commit()
        cursor.close()
        conn.close()

    def mysql_executemany(self, query, values):
        conn = self.get_fresh_mysql_conn()
        cursor = conn.cursor()
        cursor.executemany(query, values)
        conn.commit()
        cursor.close()
        conn.close()

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    # ---- 1. SECURITY & AUTH ----
    def setup_security_schema(self):
        self.mysql_execute("CREATE TABLE IF NOT EXISTS users (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(100) NOT NULL, email VARCHAR(100) NOT NULL UNIQUE, password_hash VARCHAR(255) NOT NULL, role VARCHAR(20) NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
        self.mysql_execute("CREATE TABLE IF NOT EXISTS user_activity_log (log_id INT AUTO_INCREMENT PRIMARY KEY, user_id INT, action VARCHAR(255), details TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL)")
        if not self.mysql_query("SELECT * FROM users LIMIT 1"):
            self.register_user("System Admin", "admin@gmail.com", "admin123", "Admin")
            self.register_user("Analyst User", "user@gmail.com", "user123", "User")

    def register_user(self, name, email, password, role):
        try:
            self.mysql_execute("INSERT INTO users (name, email, password_hash, role) VALUES (%s, %s, %s, %s)", (name, email, self.hash_password(password), role))
            return True, "Registration successful!"
        except Exception as e: return False, str(e)

    def login(self, email, password):
        users = self.mysql_query("SELECT id, name, email, role FROM users WHERE email=%s AND password_hash=%s", (email, self.hash_password(password)))
        if users:
            self.log_audit(users[0]['id'], "LOGIN", "System login")
            return users[0]
        return None

    def log_audit(self, user_id, action, details):
        self.mysql_execute("INSERT INTO user_activity_log (user_id, action, details) VALUES (%s, %s, %s)", (user_id, action, details))

    def get_audit_logs(self):
        return self.mysql_query("SELECT l.timestamp, u.name, u.role, l.action, l.details FROM user_activity_log l LEFT JOIN users u ON l.user_id = u.id ORDER BY l.timestamp DESC LIMIT 150")

    def get_all_users(self):
        return self.mysql_query("SELECT id, name, email, role, created_at FROM users ORDER BY created_at DESC")

    # ---- 2. SCHEMA & ETL ----
    def create_schema(self, user_id):
        self.mysql_execute("DROP TABLE IF EXISTS import_log")
        ddl = [
            "CREATE TABLE IF NOT EXISTS city_dim (city_id INT AUTO_INCREMENT PRIMARY KEY, city_name VARCHAR(100) NOT NULL UNIQUE)",
            "CREATE TABLE IF NOT EXISTS crime_type_dim (crime_type_id INT AUTO_INCREMENT PRIMARY KEY, crime_type_name VARCHAR(100) NOT NULL UNIQUE)",
            "CREATE TABLE IF NOT EXISTS date_dim (date_id INT AUTO_INCREMENT PRIMARY KEY, full_date DATE NOT NULL UNIQUE, year INT, month INT, month_name VARCHAR(20), day INT, quarter_name VARCHAR(10), season_name VARCHAR(20), day_of_week VARCHAR(20))",
            "CREATE TABLE IF NOT EXISTS crime_fact (case_id VARCHAR(60) PRIMARY KEY, city_id INT, crime_type_id INT, date_id INT, weapon_used VARCHAR(100), victim_age INT, victim_gender VARCHAR(20), suspect_age INT, suspect_gender VARCHAR(20), location_type VARCHAR(100), time_of_day VARCHAR(50), reported_to_police INT, outcome VARCHAR(100), crime_severity DECIMAL(10,2), crime_count INT, FOREIGN KEY (city_id) REFERENCES city_dim(city_id), FOREIGN KEY (crime_type_id) REFERENCES crime_type_dim(crime_type_id), FOREIGN KEY (date_id) REFERENCES date_dim(date_id))",
            "CREATE TABLE IF NOT EXISTS import_log (import_id INT AUTO_INCREMENT PRIMARY KEY, action_type VARCHAR(255), rows_affected INT, imported_at DATETIME)"
        ]
        for q in ddl: self.mysql_execute(q)
        
        self.mysql_execute("CREATE OR REPLACE VIEW city_crime_summary AS SELECT c.city_name, SUM(f.crime_count) AS total_crimes FROM crime_fact f JOIN city_dim c ON f.city_id = c.city_id GROUP BY c.city_name ORDER BY total_crimes DESC")
        self.mysql_execute("CREATE OR REPLACE VIEW yearly_crime_summary AS SELECT d.year, SUM(f.crime_count) AS total_crimes FROM crime_fact f JOIN date_dim d ON f.date_id = d.date_id GROUP BY d.year ORDER BY d.year")
        
        self.mysql_execute("DROP PROCEDURE IF EXISTS GetCrimeCountByCity")
        self.mysql_execute("CREATE PROCEDURE GetCrimeCountByCity(IN p_city VARCHAR(100)) BEGIN SELECT c.city_name, ct.crime_type_name, SUM(f.crime_count) AS total_cases FROM crime_fact f JOIN city_dim c ON f.city_id = c.city_id JOIN crime_type_dim ct ON f.crime_type_id = ct.crime_type_id WHERE c.city_name = p_city GROUP BY c.city_name, ct.crime_type_name ORDER BY total_cases DESC; END")
        
        self.mysql_execute("DROP TRIGGER IF EXISTS trg_after_import_insert")
        self.mysql_execute("CREATE TRIGGER trg_after_import_insert AFTER INSERT ON crime_fact FOR EACH ROW BEGIN INSERT INTO import_log(action_type, rows_affected, imported_at) VALUES ('ROW_INSERT', 1, NOW()); END")
        
        self.log_audit(user_id, "SCHEMA_INIT", "Re-initialized system schema, views, procedures, and triggers")

    def _clean_dataframe(self, df):
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        rename_map = {"Crime_Typ": "Crime_Type", "Weapon_U": "Weapon_Used", "Victim_Ge": "Victim_Gender", "Suspect_A": "Suspect_Age", "Suspect_G": "Suspect_Gender", "Location_T": "Location_Type", "Time_of_D": "Time_of_Day", "Reported_t": "Reported_to_Police", "Crime_Sev": "Crime_Severity"}
        df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Year"] = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month
            df["Month_Name"] = df["Date"].dt.month_name()
            df["Day"] = df["Date"].dt.day
        if "Crime_Count" not in df.columns: df["Crime_Count"] = 1
        if "Case_ID" not in df.columns: df["Case_ID"] = [f"CASE-{i+1:06d}" for i in range(len(df))]
        return df.dropna(subset=["Date", "Year"]).reset_index(drop=True)

    def import_dataset(self, file_path, user_id):
        df = self._clean_dataframe(pd.read_csv(file_path) if file_path.lower().endswith(".csv") else pd.read_excel(file_path))
        self.create_schema(user_id)

        cities = sorted(df["City"].dropna().astype(str).unique().tolist()) if "City" in df.columns else []
        crime_types = sorted(df["Crime_Type"].dropna().astype(str).unique().tolist()) if "Crime_Type" in df.columns else []
        for city in cities: self.mysql_execute("INSERT IGNORE INTO city_dim(city_name) VALUES (%s)", (city,))
        for crime in crime_types: self.mysql_execute("INSERT IGNORE INTO crime_type_dim(crime_type_name) VALUES (%s)", (crime,))

        date_df = df[["Date", "Year", "Month", "Month_Name", "Day"]].drop_duplicates()
        for _, row in date_df.iterrows():
            self.mysql_execute("INSERT IGNORE INTO date_dim(full_date, year, month, month_name, day) VALUES (%s, %s, %s, %s, %s)", (pd.to_datetime(row["Date"]).date(), int(row["Year"]), int(row["Month"]), str(row["Month_Name"]), int(row["Day"])))

        city_lookup = {r["city_name"]: r["city_id"] for r in self.mysql_query("SELECT * FROM city_dim")}
        crime_lookup = {r["crime_type_name"]: r["crime_type_id"] for r in self.mysql_query("SELECT * FROM crime_type_dim")}
        date_lookup = {r["full_date"]: r["date_id"] for r in self.mysql_query("SELECT * FROM date_dim")}

        insert_sql = "INSERT INTO crime_fact(case_id, city_id, crime_type_id, date_id, weapon_used, victim_age, victim_gender, suspect_age, suspect_gender, location_type, time_of_day, reported_to_police, outcome, crime_severity, crime_count) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE city_id = VALUES(city_id)"
        
        batch = []
        for _, r in df.iterrows():
            dt = pd.to_datetime(r["Date"], errors="coerce")
            batch.append((str(r.get("Case_ID")), city_lookup.get(str(r.get("City"))), crime_lookup.get(str(r.get("Crime_Type"))), date_lookup.get(dt.date() if pd.notna(dt) else None), str(r.get("Weapon_Used", "Unknown")), int(r.get("Victim_Age")) if pd.notna(r.get("Victim_Age")) else None, str(r.get("Victim_Gender", "Unknown")), int(r.get("Suspect_Age")) if pd.notna(r.get("Suspect_Age")) else None, str(r.get("Suspect_Gender", "Unknown")), str(r.get("Location_Type", "Unknown")), str(r.get("Time_of_Day", "Unknown")), int(r.get("Reported_to_Police")) if pd.notna(r.get("Reported_to_Police")) else None, str(r.get("Outcome", "Unknown")), float(r.get("Crime_Severity", 5.0)), int(r.get("Crime_Count", 1))))

        self.mysql_executemany(insert_sql, batch)
        self.mysql_execute("DELETE FROM import_log WHERE action_type = 'ROW_INSERT'") 
        self.mysql_execute("INSERT INTO import_log(action_type, rows_affected, imported_at) VALUES (%s, %s, NOW())", (f"FILE_IMPORT", len(df)))
        self.log_audit(user_id, "DATA_IMPORT", f"Imported {len(df)} rows")
        return len(df)

    # ---- 3. DATABASE QUERIES ----
    def get_dashboard_kpis(self):
        return {
            "Total Incidents": self.mysql_query("SELECT SUM(crime_count) as v FROM crime_fact")[0]['v'] or 0,
            "Total Cities": self.mysql_query("SELECT COUNT(*) as v FROM city_dim")[0]['v'],
            "Crime Types": self.mysql_query("SELECT COUNT(*) as v FROM crime_type_dim")[0]['v'],
        }

    def run_dynamic_query(self, cols, group_by, filter_city, user_id):
        select = ", ".join(cols) if cols else "*"
        where = f"WHERE c.city_name = '{filter_city}'" if filter_city != "All" else ""
        group = f"GROUP BY {group_by}" if group_by != "None" else ""
        q = f"SELECT {select} FROM crime_fact f LEFT JOIN city_dim c ON f.city_id = c.city_id LEFT JOIN crime_type_dim ct ON f.crime_type_id = ct.crime_type_id LEFT JOIN date_dim d ON f.date_id = d.date_id {where} {group} LIMIT 1000"
        self.log_audit(user_id, "DYNAMIC_QUERY", f"Ran dynamic query grouped by {group_by}")
        return self.mysql_query(q)

    def get_explain_plan(self, query, user_id):
        self.log_audit(user_id, "EXPLAIN_PLAN", "Executed Explain Plan")
        return self.mysql_query(f"EXPLAIN {query}")

    def call_procedure_city(self, city, user_id):
        self.log_audit(user_id, "PROCEDURE_CALL", f"Called GetCrimeCountByCity for {city}")
        conn = self.get_fresh_mysql_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.callproc('GetCrimeCountByCity', [city])
        res = []
        for r in cursor.stored_results(): res.extend(r.fetchall())
        cursor.close()
        return res

    def get_trigger_logs(self):
        return self.mysql_query("SELECT * FROM import_log ORDER BY imported_at DESC LIMIT 50")

    # ---- 4. MONGO POLYGLOT ----
    def add_case_evidence(self, case_id, text, user_name, user_id):
        self.connect_mongo()
        self.mongo_collection_evidence.insert_one({"case_id": case_id, "evidence_text": text, "added_by": user_name, "added_at": datetime.now()})
        self.log_audit(user_id, "EVIDENCE_ADD", f"Added notes to case {case_id}")
        
    def get_case_evidence(self, case_id):
        self.connect_mongo()
        docs = list(self.mongo_collection_evidence.find({"case_id": case_id}).sort("added_at", -1))
        for d in docs: d.pop("_id", None)
        return docs
        
    def save_report_to_mongo(self, report_name, rows, user_name):
        self.connect_mongo()
        self.mongo_collection_reports.insert_one({"report_type": report_name, "generated_on": datetime.now(), "generated_by": user_name, "row_count": len(rows), "results": rows})

    def get_saved_reports(self):
        self.connect_mongo()
        docs = list(self.mongo_collection_reports.find({}, {"results": 0}).sort("generated_on", -1).limit(50))
        for d in docs: d.pop("_id", None)
        return docs

    # ---- 5. ML FORECASTING & PREDICTION ----
    def yearly_forecast(self, target_year, user_id):
        self.log_audit(user_id, "FORECAST", f"Ran forecast for {target_year}")
        df = pd.DataFrame(self.mysql_query("SELECT * FROM yearly_crime_summary"))
        if len(df) < 3: raise ValueError("Need 3+ years of data.")
        model = LinearRegression().fit(df[["year"]], df["total_crimes"])
        return df, int(max(0, round(float(model.predict(np.array([[target_year]]))[0]))))

    def train_outcome_model(self, user_id):
        df = pd.DataFrame(self.mysql_query("SELECT c.city_name as city, ct.crime_type_name as crime_type, f.weapon_used, f.location_type, f.time_of_day, f.victim_age, f.suspect_age, f.reported_to_police, f.crime_severity, f.outcome FROM crime_fact f JOIN city_dim c ON f.city_id = c.city_id JOIN crime_type_dim ct ON f.crime_type_id = ct.crime_type_id WHERE f.outcome IS NOT NULL AND f.outcome != 'Unknown'"))
        if df.empty: raise ValueError("No valid training records.")
        features = ["city", "crime_type", "weapon_used", "location_type", "time_of_day", "victim_age", "suspect_age", "reported_to_police", "crime_severity"]
        X, y = df[features], df["outcome"].astype(str)
        prep = ColumnTransformer([
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["city", "crime_type", "weapon_used", "location_type", "time_of_day"]),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), ["victim_age", "suspect_age", "reported_to_police", "crime_severity"]),
        ])
        model = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        self.outcome_model, self.outcome_model_features = model, features
        self.outcome_model_accuracy = round(float(accuracy_score(y_test, model.predict(X_test))), 4)
        self.log_audit(user_id, "MODEL_TRAIN", f"Trained outcome model (Acc: {self.outcome_model_accuracy})")
        return self.outcome_model_accuracy

    def predict_outcome(self, payload, user_id):
        if self.outcome_model is None: raise ValueError("Model not trained.")
        self.log_audit(user_id, "PREDICT", "Ran outcome prediction")
        return self.outcome_model.predict(pd.DataFrame([{f: payload.get(f) for f in self.outcome_model_features}]))[0]

# ==========================================================
# UI COMPONENTS
# ==========================================================
def login_screen():
    st.markdown("<h1 style='text-align: center; margin-top: 50px;'>🏢 Crime Intelligence Enterprise Suite</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8;'>Advanced Access Gateway - Powered by Zamsha Developers</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        tabs = st.tabs(["🔒 Secure Login", "📝 User Registration"])
        with tabs[0]:
            with st.form("login"):
                email = st.text_input("Email / Gmail", placeholder="admin@gmail.com")
                pwd = st.text_input("Password", type="password", placeholder="admin123")
                if st.form_submit_button("Authenticate", use_container_width=True):
                    user = st.session_state.engine.login(email, pwd)
                    if user:
                        st.session_state.user = user
                        st.session_state.logged_in = True
                        st.rerun()
                    else: st.error("Invalid credentials.")
        with tabs[1]:
            with st.form("register"):
                n_name = st.text_input("Full Name")
                n_email = st.text_input("Gmail Address")
                n_pwd = st.text_input("Password", type="password")
                n_role = st.selectbox("Role", ["User", "Admin"])
                if st.form_submit_button("Register", use_container_width=True):
                    s, m = st.session_state.engine.register_user(n_name, n_email, n_pwd, n_role)
                    if s: st.success(m)
                    else: st.error(m)

# ---------------- ADMIN UI ----------------
def admin_health(engine):
    st.title("1. System Health & Metrics")
    c1, c2, c3 = st.columns(3)
    try: 
        conn = engine.get_fresh_mysql_conn()
        conn.close()
        c1.markdown("<div class='metric-card'><div class='metric-label'>MySQL Core</div><div class='metric-val' style='color:#059669'>ONLINE</div></div>", unsafe_allow_html=True)
    except: c1.markdown("<div class='metric-card'><div class='metric-label'>MySQL Core</div><div class='metric-val' style='color:#dc2626'>OFFLINE</div></div>", unsafe_allow_html=True)
    try: 
        engine.connect_mongo()
        c2.markdown("<div class='metric-card'><div class='metric-label'>Mongo Evidence DB</div><div class='metric-val' style='color:#059669'>ONLINE</div></div>", unsafe_allow_html=True)
    except: c2.markdown("<div class='metric-card'><div class='metric-label'>Mongo Evidence DB</div><div class='metric-val' style='color:#dc2626'>OFFLINE</div></div>", unsafe_allow_html=True)
    try:
        kpis = engine.get_dashboard_kpis()
        c3.markdown(f"<div class='metric-card'><div class='metric-label'>Total Data Rows</div><div class='metric-val'>{kpis['Total Incidents']}</div></div>", unsafe_allow_html=True)
    except: c3.markdown("<div class='metric-card'><div class='metric-label'>Total Data Rows</div><div class='metric-val'>N/A</div></div>", unsafe_allow_html=True)

def admin_audit(engine):
    st.title("2. Security Audit Logs")
    st.write("Immutable enterprise tracking of all user actions.")
    try: st.dataframe(pd.DataFrame(engine.get_audit_logs()), use_container_width=True, height=500)
    except: st.warning("Schema not initialized.")

def admin_users(engine):
    st.title("3. User Access Management")
    try: st.dataframe(pd.DataFrame(engine.get_all_users()), use_container_width=True)
    except: st.warning("Schema not initialized.")

def admin_schema(engine, user_id):
    st.title("4. Database Schema Deployment")
    st.write("Force initialize or reset the entire MySQL architectural schema.")
    if st.button("Deploy Enterprise Schema", type="primary"):
        with st.spinner("Executing DDL..."):
            engine.create_schema(user_id)
            st.success("Schema, Tables, Views, Procedures, and Triggers successfully deployed!")

def admin_etl(engine, user_id):
    st.title("5. ETL Data Intake Pipeline")
    uploaded = st.file_uploader("Upload Raw Dataset", type=["csv", "xlsx"])
    if st.button("Run ETL Pipeline", type="primary") and uploaded:
        temp = f"temp_{uploaded.name}"
        with open(temp, "wb") as f: f.write(uploaded.getbuffer())
        with st.spinner("Extracting, Transforming, and Loading to MySQL..."):
            try:
                rows = engine.import_dataset(temp, user_id)
                st.success(f"ETL Complete! {rows} normalized rows inserted.")
            except Exception as e: st.error(f"ETL Error: {e}")

def admin_explain(engine, user_id):
    st.title("6. DB Query Optimizer (EXPLAIN)")
    q = st.text_area("SQL Query", "SELECT * FROM crime_fact LIMIT 10")
    if st.button("Analyze Execution Plan"):
        try: st.dataframe(pd.DataFrame(engine.get_explain_plan(q, user_id)), use_container_width=True)
        except Exception as e: st.error(str(e))

def admin_triggers(engine, user_id):
    st.title("7. Triggers & Stored Procedures")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Test Stored Procedure")
        st.write("Executes `CALL GetCrimeCountByCity` directly.")
        city = st.text_input("Enter City", "Karachi")
        if st.button("Execute Procedure"):
            try: st.dataframe(pd.DataFrame(engine.call_procedure_city(city, user_id)), use_container_width=True)
            except Exception as e: st.error(str(e))
    with col2:
        st.subheader("View Trigger Logs")
        st.write("Reads from `import_log` populated by MySQL Triggers.")
        if st.button("Fetch Trigger Logs"):
            try: st.dataframe(pd.DataFrame(engine.get_trigger_logs()), use_container_width=True)
            except Exception as e: st.error(str(e))

def admin_model(engine, user_id):
    st.title("8. ML Model Deployment")
    st.info(f"Status: {'🟢 Deployed' if engine.outcome_model else '🔴 Offline'}")
    if st.button("Train / Retrain Model on Live Data", type="primary"):
        with st.spinner("Training Logistic Regression Model..."):
            try:
                acc = engine.train_outcome_model(user_id)
                st.success("Model trained successfully! Ready for Prediction Lab.")
            except Exception as e: st.error(str(e))

# ---------------- USER UI ----------------
def user_dashboard(engine):
    st.title("1. Zamsha Professional Dashboard")
    try:
        kpis = engine.get_dashboard_kpis()
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='metric-card'><div class='metric-label'>Total Incidents</div><div class='metric-val'>{kpis['Total Incidents']}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='metric-label'>Registered Cities</div><div class='metric-val'>{kpis['Total Cities']}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><div class='metric-label'>Crime Categories</div><div class='metric-val'>{kpis['Crime Types']}</div></div>", unsafe_allow_html=True)
        
        # Professional Charting
        df_city = pd.DataFrame(engine.mysql_query("SELECT * FROM city_crime_summary LIMIT 10"))
        df_trend = pd.DataFrame(engine.mysql_query("SELECT d.year, SUM(f.crime_count) as total FROM crime_fact f JOIN date_dim d ON f.date_id = d.date_id GROUP BY d.year ORDER BY d.year"))
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            if not df_city.empty:
                donut = alt.Chart(df_city).mark_arc(innerRadius=60).encode(
                    theta=alt.Theta(field="total_crimes", type="quantitative"),
                    color=alt.Color(field="city_name", type="nominal", legend=alt.Legend(title="City")),
                    tooltip=['city_name', 'total_crimes']
                ).properties(title="Distribution by City", height=300)
                st.altair_chart(donut, use_container_width=True)
        with col_chart2:
            if not df_trend.empty:
                area = alt.Chart(df_trend).mark_area(opacity=0.6, color="#0ea5e9").encode(
                    x=alt.X('year:O', title="Year"),
                    y=alt.Y('total:Q', title="Total Crimes"),
                    tooltip=['year', 'total']
                ).properties(title="Time-Series Trend", height=300)
                st.altair_chart(area, use_container_width=True)
    except: st.warning("Data not available. Please run ETL setup.")

def user_hotspots(engine):
    st.title("2. 3D Geospatial Hotspots (PyDeck)")
    st.write("Advanced 3D interactive mapping layer utilizing `pydeck`. Adjust pitch and zoom with mouse.")
    try:
        df = pd.DataFrame(engine.mysql_query("SELECT * FROM city_crime_summary"))
        if not df.empty:
            # Map Controls (Layers)
            c1, c2 = st.columns([1, 3])
            with c1:
                st.markdown("### Map Filters")
                min_crimes = st.slider("Minimum Crime Count", 0, int(df['total_crimes'].max()), 0)
                filtered_df = df[df['total_crimes'] >= min_crimes]
                st.metric("Cities Displayed", len(filtered_df))
            
            with c2:
                if not filtered_df.empty:
                    # Clean, professional Scatterplot using native st.map
                    map_data = []
                    for _, r in filtered_df.iterrows():
                        lat, lon = CITY_COORDS.get(r['city_name'], [30.0, 70.0])
                        map_data.append({
                            "lat": lat,
                            "lon": lon,
                            "size": float(r['total_crimes']) * 15, # Scale size
                            "color": "#ef4444" # Professional red
                        })
                    
                    st.map(pd.DataFrame(map_data), size="size", color="color", zoom=5, use_container_width=True)
                else:
                    st.info("No cities meet the filter criteria.")
    except: st.warning("Data not available.")

def user_builder(engine, user_id):
    st.title("3. Dynamic Query Builder")
    c1, c2, c3 = st.columns(3)
    cols = c1.multiselect("Columns", ["c.city_name", "ct.crime_type_name", "f.weapon_used", "SUM(f.crime_count) as total"])
    grp = c2.selectbox("Group By", ["None", "c.city_name", "ct.crime_type_name", "c.city_name, ct.crime_type_name"])
    flt = c3.selectbox("Filter City", ["All", "Karachi", "Lahore", "Islamabad"])
    
    if st.button("Generate Report", type="primary"):
        try:
            res_df = pd.DataFrame(engine.run_dynamic_query(cols, grp, flt, user_id))
            st.dataframe(res_df, use_container_width=True)
            st.download_button("📥 Export CSV", res_df.to_csv(index=False), "report.csv", "text/csv")
        except Exception as e: st.error(str(e))

def user_advanced(engine, user_id):
    st.title("4. Advanced SQL Analytics")
    st.write("Demonstrates Window Functions (`LAG`, `OVER`) and Hierarchical `ROLLUP`.")
    q_yoy = "SELECT d.year, SUM(f.crime_count) AS total, LAG(SUM(f.crime_count)) OVER (ORDER BY d.year) AS prev_year, ROUND((SUM(f.crime_count) - LAG(SUM(f.crime_count)) OVER (ORDER BY d.year)) / LAG(SUM(f.crime_count)) OVER (ORDER BY d.year) * 100, 2) AS yoy_growth FROM crime_fact f JOIN date_dim d ON f.date_id = d.date_id GROUP BY d.year ORDER BY d.year"
    q_roll = "SELECT IFNULL(c.city_name, 'GRAND TOTAL') as city, IFNULL(ct.crime_type_name, 'CITY TOTAL') as crime_type, SUM(f.crime_count) as cases FROM crime_fact f JOIN city_dim c ON f.city_id = c.city_id JOIN crime_type_dim ct ON f.crime_type_id = ct.crime_type_id GROUP BY c.city_name, ct.crime_type_name WITH ROLLUP"
    
    tabs = st.tabs(["Year-Over-Year Growth", "Hierarchical Rollup"])
    with tabs[0]:
        if st.button("Run YoY Growth"): 
            engine.log_audit(user_id, "ADVANCED_SQL", "Ran YoY")
            st.dataframe(pd.DataFrame(engine.mysql_query(q_yoy)), use_container_width=True)
    with tabs[1]:
        if st.button("Run Rollup"): 
            engine.log_audit(user_id, "ADVANCED_SQL", "Ran Rollup")
            st.dataframe(pd.DataFrame(engine.mysql_query(q_roll)), use_container_width=True)

def user_polyglot(engine, user):
    st.title("5. Polyglot Case Explorer")
    st.write("Relational structured data from **MySQL**, combined with unstructured evidence notes from **MongoDB**.")
    try:
        cases = pd.DataFrame(engine.mysql_query("SELECT case_id, city_name, crime_type_name, outcome FROM crime_fact f JOIN city_dim c ON f.city_id = c.city_id JOIN crime_type_dim ct ON f.crime_type_id = ct.crime_type_id LIMIT 100"))
        if not cases.empty:
            c_id = st.selectbox("Select Case ID", cases['case_id'])
            st.write(cases[cases['case_id'] == c_id].iloc[0].to_dict())
            
            ev = st.text_area("Investigator Evidence Notes (MongoDB)")
            if st.button("Save Evidence"):
                engine.add_case_evidence(c_id, ev, user['name'], user['id'])
                st.success("Saved to MongoDB!")
            st.write("---")
            for d in engine.get_case_evidence(c_id):
                st.info(f"**{d['added_by']}**: {d['evidence_text']}")
    except: st.warning("Data not available.")

def user_predict(engine, user_id):
    st.title("6. Outcome Prediction Lab")
    if not engine.outcome_model: st.warning("Admin must train the model first.")
    else:
        try:
            p = {
                'city': st.selectbox("City", [r['city_name'] for r in engine.mysql_query("SELECT * FROM city_dim")]),
                'crime_type': st.selectbox("Type", [r['crime_type_name'] for r in engine.mysql_query("SELECT * FROM crime_type_dim")]),
                'weapon_used': 'None', 'location_type': 'Street', 'time_of_day': 'Night',
                'victim_age': st.number_input("Victim Age", 30), 
                'suspect_age': st.number_input("Suspect Age", 25), 
                'reported_to_police': 1, 'crime_severity': 7
            }
            if st.button("Predict Outcome", type="primary"):
                if p['suspect_age'] < 18:
                    st.error("Cannot predict arrest: Suspect is underage (under 18 in Pakistan).")
                else:
                    res = engine.predict_outcome(p, user_id)
                    st.success(f"Predicted Outcome: **{res}**")
        except: pass

def user_forecast(engine, user_id):
    st.title("7. Time-Series Forecasting")
    y = st.number_input("Target Year", 2024, 2050, 2027)
    if st.button("Run Forecast", type="primary"):
        try:
            df, pred = engine.yearly_forecast(y, user_id)
            st.metric(f"Forecasted Crimes ({y})", pred)
            st.altair_chart(alt.Chart(df).mark_line(point=True).encode(x='year:O', y='total_crimes'), use_container_width=True)
        except Exception as e: st.error(str(e))

def user_reports(engine, user):
    st.title("8. Automated Report Center")
    st.write("Save and retrieve reports permanently via MongoDB.")
    if st.button("Save Current City Summary to Mongo"):
        rows = engine.mysql_query("SELECT * FROM city_crime_summary")
        engine.save_report_to_mongo("city_summary", rows, user['name'])
        engine.log_audit(user['id'], "REPORT_SAVE", "Saved city summary to Mongo")
        st.success("Saved successfully!")
    
    st.subheader("Historical Reports")
    try: st.dataframe(pd.DataFrame(engine.get_saved_reports()), use_container_width=True)
    except: st.warning("No reports found or Mongo offline.")

# ==========================================================
# MAIN ROUTING
# ==========================================================
def main():
    if "engine" not in st.session_state:
        st.session_state.engine = EnterpriseEngine()
        try: st.session_state.engine.setup_security_schema()
        except: pass
    
    if not st.session_state.get("logged_in", False):
        login_screen()
    else:
        user = st.session_state.user
        st.sidebar.markdown("<div class='zamsha-sidebar'>⚙️ Engineered by<br>Zamsha Developers</div>", unsafe_allow_html=True)
        st.sidebar.markdown(f"### 👤 {user['name']}")
        st.sidebar.caption(f"{user['email']} | **{user['role']}**")
        if st.sidebar.button("Logout"):
            st.session_state.engine.log_audit(user['id'], "LOGOUT", "User logged out")
            st.session_state.logged_in = False
            st.rerun()
            
        st.sidebar.divider()
        
        if user['role'] == "Admin":
            nav = st.sidebar.radio("Administrator Modules", [
                "1. System Health & Metrics", "2. Security Audit Logs", "3. User Access Management", 
                "4. Database Schema Deployment", "5. ETL Data Intake Pipeline", "6. DB Query Optimizer (EXPLAIN)", 
                "7. Triggers & Stored Procedures", "8. ML Model Deployment"
            ])
            if nav.startswith("1"): admin_health(st.session_state.engine)
            elif nav.startswith("2"): admin_audit(st.session_state.engine)
            elif nav.startswith("3"): admin_users(st.session_state.engine)
            elif nav.startswith("4"): admin_schema(st.session_state.engine, user['id'])
            elif nav.startswith("5"): admin_etl(st.session_state.engine, user['id'])
            elif nav.startswith("6"): admin_explain(st.session_state.engine, user['id'])
            elif nav.startswith("7"): admin_triggers(st.session_state.engine, user['id'])
            elif nav.startswith("8"): admin_model(st.session_state.engine, user['id'])
        else:
            nav = st.sidebar.radio("Analyst Modules", [
                "1. Zamsha Professional Dashboard", "2. 3D Geospatial Hotspots (PyDeck)", "3. Dynamic Query Builder",
                "4. Advanced SQL Analytics", "5. Polyglot Case Explorer", "6. Outcome Prediction Lab",
                "7. Time-Series Forecasting", "8. Automated Report Center"
            ])
            if nav.startswith("1"): user_dashboard(st.session_state.engine)
            elif nav.startswith("2"): user_hotspots(st.session_state.engine)
            elif nav.startswith("3"): user_builder(st.session_state.engine, user['id'])
            elif nav.startswith("4"): user_advanced(st.session_state.engine, user['id'])
            elif nav.startswith("5"): user_polyglot(st.session_state.engine, user)
            elif nav.startswith("6"): user_predict(st.session_state.engine, user['id'])
            elif nav.startswith("7"): user_forecast(st.session_state.engine, user['id'])
            elif nav.startswith("8"): user_reports(st.session_state.engine, user)

if __name__ == "__main__":
    main()
