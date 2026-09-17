import io
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

st.set_page_config(page_title="AstraBurn AI — Semiconductor Screening", page_icon="🛰️", layout="wide")

st.markdown("## 🛰️ AstraBurn AI — Autonomous Semiconductor Screening System")
st.caption("Mission-Critical Latent Defect Identification & Reliability Intelligence Platform")

# --- FUNCTION TO CREATE DEFAULT SYNTHETIC DATASET ---
def generate_default_data():
    np.random.seed(42)
    grid_size = 13
    radius = grid_size // 2
    records = []
    comp_counter = 1
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            dist = np.sqrt(x**2 + y**2)
            if dist <= radius:
                base_leakage = np.random.normal(10.0, 1.2)
                v_th = np.random.normal(1.8, 0.05)
                v_br = np.random.normal(32.0, 1.1)
                is_edge = dist > (radius - 1.5)
                thermal_drift = np.random.normal(0.4, 0.15)
                if is_edge and np.random.rand() > 0.65:
                    base_leakage += np.random.uniform(25.0, 36.0)
                    v_th -= np.random.uniform(0.3, 0.5)
                    v_br -= np.random.uniform(5.0, 8.0)
                    thermal_drift += np.random.uniform(8.0, 18.0)
                records.append({
                    'Component_ID': f"W01_D{comp_counter:03d}",
                    'Lot_ID': 'Lot_L10',
                    'Wafer_X': x,
                    'Wafer_Y': y,
                    'Pre_BurnIn_uA': round(float(base_leakage), 2),
                    'Post_BurnIn_uA': round(float(base_leakage + thermal_drift), 2),
                    'Vth_Volts': round(float(v_th), 3),
                    'Vbr_Volts': round(float(v_br), 2)
                })
                comp_counter += 1
    return pd.DataFrame(records)

# --- SESSION STATE INITIALIZATION ---
if 'dataset' not in st.session_state:
    st.session_state.dataset = generate_default_data()
    st.session_state.is_custom_uploaded = False

# --- SIDEBAR: CONTROLS ---
st.sidebar.title("🎛️ Mission Configuration")
menu = st.sidebar.radio("Navigation", [
    "📊 Officer Executive Dashboard", 
    "📦 Dual-Bin Hardware Inspector",
    "⚡ Real-Time Burn-In Chamber Sim",
    "🗺️ Wafer Spatial Defect Heatmap",
    "🧠 Explainable AI (XAI) Diagnostics",
    "⏳ Thermal Stress & Lifetime Reliability",
    "🚨 Live Audio Alerts & Quarantine Hub",
    "📥 Data Ingestion (Upload / Manual)", 
    "📈 Lot Variation & Distribution", 
    "🔥 Burn-In Drift Prediction"
])

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Screening Profile")

mission_mode = st.sidebar.selectbox(
    "Mission Criticality Preset", 
    [
        "Deep Space / Manned (Ultra-Strict |Z| > 2.0)", 
        "Standard Satellite Payload (|Z| > 2.5)", 
        "Commercial / Student Payload (|Z| > 3.0)",
        "Custom Manual Tuning"
    ]
)

default_z = 2.0 if "Deep Space" in mission_mode else (3.0 if "Commercial" in mission_mode else 2.5)

user_z_threshold = st.sidebar.slider(
    "Z-Score Outlier Cutoff (|Z|):", 
    min_value=1.5, 
    max_value=4.0, 
    value=default_z, 
    step=0.1
)

user_static_limit = st.sidebar.number_input(
    "Static Pass Ceiling (µA):", 
    min_value=1.0, 
    max_value=100.0, 
    value=50.0, 
    step=1.0
)

if st.sidebar.button("🔄 Reset to Default Wafer Lot"):
    st.session_state.dataset = generate_default_data()
    st.session_state.is_custom_uploaded = False
    st.rerun()

if st.session_state.get('is_custom_uploaded', False):
    st.sidebar.success("📂 Using Custom Uploaded Lot")

# --- CORE SCREENING ENGINE ---
def process_data(data, static_limit, z_thresh):
    d = data.copy()
    if 'Lot_ID' not in d.columns:
        d['Lot_ID'] = 'Lot_Uploaded'
    
    d['Lot_Mean'] = d.groupby('Lot_ID')['Pre_BurnIn_uA'].transform('mean')
    d['Lot_Std'] = d.groupby('Lot_ID')['Pre_BurnIn_uA'].transform('std').replace(0, 0.001).fillna(0.001)
    d['Z_Score'] = (d['Pre_BurnIn_uA'] - d['Lot_Mean']) / d['Lot_Std']
    d['Static_Status'] = np.where(d['Pre_BurnIn_uA'] <= static_limit, 'PASS', 'FAIL')
    d['AstraBurn_Anomaly'] = d['Z_Score'].abs() > z_thresh
    d['Defect_Category'] = 'Flight Qualified'
    d.loc[d['Static_Status'] == 'FAIL', 'Defect_Category'] = 'Catastrophic Fail'
    d.loc[(d['Static_Status'] == 'PASS') & (d['AstraBurn_Anomaly']), 'Defect_Category'] = 'Latent Defect (Outlier)'
    
    if 'Post_BurnIn_uA' not in d.columns:
        d['Post_BurnIn_uA'] = d['Pre_BurnIn_uA'] + 0.3
    
    d['Drift_uA'] = d['Post_BurnIn_uA'] - d['Pre_BurnIn_uA']
    d['Drift_Percent'] = (d['Drift_uA'] / d['Pre_BurnIn_uA'].replace(0, 0.001)) * 100
    z_abs = d['Z_Score'].abs()
    d['Risk_Probability'] = np.clip((z_abs / 4.0) * 60 + (d['Drift_Percent'] / 50.0) * 40, 5, 99)
    return d

processed_df = process_data(st.session_state.dataset, user_static_limit, user_z_threshold)

# --- PDF GENERATOR ---
def generate_pdf_report(dataframe, mission, static_lim, z_cut):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=17, textColor=colors.HexColor('#0b2240'), alignment=1)
    elements = [
        Paragraph("FLIGHT CLEARANCE INSPECTION REPORT", title_style),
        Paragraph("AstraBurn AI Autonomous Semiconductor Screening System", styles['Normal']),
        Spacer(1, 14)
    ]
    total_parts = len(dataframe)
    static_bad = (dataframe['Static_Status'] == 'FAIL').sum()
    latent_bad = (dataframe['Defect_Category'] == 'Latent Defect (Outlier)').sum()
    accepted = total_parts - static_bad - latent_bad
    yield_val = (accepted / total_parts) * 100 if total_parts > 0 else 0
    status_stamp = "FLIGHT QUALIFIED" if latent_bad == 0 and static_bad == 0 else "BATCH QUARANTINE / REJECT"
    summary_data = [
        ['Parameter', 'Specification / Test Result'],
        ['Mission Profile', mission],
        ['Static Ceiling Limit', f"{static_lim} µA"],
        ['Statistical Cutoff Threshold', f"|Z| > {z_cut:.1f}σ"],
        ['Total Screened Units', str(total_parts)],
        ['Static Failures (> Ceiling)', str(static_bad)],
        ['Latent Defects Flagged', str(latent_bad)],
        ['Flight Qualified Yield', f"{yield_val:.2f}%"],
        ['Batch Certification', status_stamp]
    ]
    t = Table(summary_data, colWidths=[240, 280])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0b2240')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
        ('TEXTCOLOR', (0, -1), (-1, -1), colors.HexColor('#c53030') if status_stamp != "FLIGHT QUALIFIED" else colors.HexColor('#276749')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold')
    ]))
    elements.append(t)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# --- 1. EXECUTIVE DASHBOARD ---
if menu == "📊 Officer Executive Dashboard":
    st.subheader("Executive Screening Overview")
    st.info(f"Active Thresholds: **Static Limit = {user_static_limit} µA** | **Outlier Cutoff = |Z| > {user_z_threshold:.1f}σ**")

    m1, m2, m3, m4 = st.columns(4)
    total = len(processed_df)
    static_fails = (processed_df['Static_Status'] == 'FAIL').sum()
    latent_defects = (processed_df['Defect_Category'] == 'Latent Defect (Outlier)').sum()
    yield_rate = ((total - static_fails - latent_defects) / total) * 100 if total > 0 else 0
    
    m1.metric("Total Tested", f"{total}")
    m2.metric(f"Static Fails (>{user_static_limit}µA)", f"{static_fails}")
    m3.metric(f"Latent Defects (|Z|>{user_z_threshold:.1f})", f"{latent_defects}", delta="High Risk" if latent_defects > 0 else "Nominal", delta_color="inverse")
    m4.metric("Flight Qualified Yield", f"{yield_rate:.1f}%")
    
    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.histogram(
            processed_df, 
            x='Pre_BurnIn_uA', 
            color='Defect_Category',
            color_discrete_map={'Flight Qualified': '#0984e3', 'Latent Defect (Outlier)': '#d63031', 'Catastrophic Fail': '#e17055'},
            title=f"Statistical Distribution (Dynamic Limit: {user_static_limit} µA)"
        )
        fig.add_vline(x=user_static_limit, line_dash="dash", line_color="black", annotation_text=f"Limit ({user_static_limit} µA)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.write("##### Flagged Latent Defects")
        flagged = processed_df[processed_df['Defect_Category'] == 'Latent Defect (Outlier)'][['Component_ID', 'Lot_ID', 'Pre_BurnIn_uA', 'Z_Score']]
        st.dataframe(flagged, use_container_width=True)
        pdf_bytes = generate_pdf_report(processed_df, mission_mode, user_static_limit, user_z_threshold)
        st.download_button(
            label="📄 Download Flight Clearance Certificate (PDF)",
            data=pdf_bytes,
            file_name="AstraBurn_Flight_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# --- 2. DUAL-BIN HARDWARE INSPECTOR ---
elif menu == "📦 Dual-Bin Hardware Inspector":
    st.subheader("Physical Hardware Binning & Interactive Defect Inspector")
    st.caption("Segregated cleanroom sorting bins: Flight-Approved Hardware vs Quarantined Outliers")
    
    good_hardware = processed_df[processed_df['Defect_Category'] == 'Flight Qualified']
    defect_hardware = processed_df[processed_df['Defect_Category'] != 'Flight Qualified']
    
    bin_col1, bin_col2 = st.columns([1, 1])
    
    with bin_col1:
        st.markdown(f"#### 🟢 Bin A: Flight Qualified Hardware ({len(good_hardware)} units)")
        st.success("Components verified safe for space payloads.")
        st.dataframe(
            good_hardware[['Component_ID', 'Lot_ID', 'Pre_BurnIn_uA', 'Z_Score', 'Risk_Probability']].rename(
                columns={'Pre_BurnIn_uA': 'Leakage (µA)', 'Z_Score': 'Z-Score (σ)', 'Risk_Probability': 'Risk (%)'}
            ), 
            height=320, 
            use_container_width=True
        )
        
    with bin_col2:
        st.markdown(f"#### 🔴 Bin B: Quarantined Defect Hardware ({len(defect_hardware)} units)")
        st.error("Components flagged for latent breakdown or static failure.")
        st.dataframe(
            defect_hardware[['Component_ID', 'Lot_ID', 'Pre_BurnIn_uA', 'Z_Score', 'Defect_Category']].rename(
                columns={'Pre_BurnIn_uA': 'Leakage (µA)', 'Z_Score': 'Z-Score (σ)', 'Defect_Category': 'Status'}
            ), 
            height=320, 
            use_container_width=True
        )
    
    st.markdown("---")
    st.subheader("🔍 Interactive Defect Reason Inspector")
    st.markdown("Select any defective component from **Bin B** to view exact root-cause failure analysis:")
    
    if len(defect_hardware) > 0:
        defect_ids = list(defect_hardware['Component_ID'])
        selected_defect_id = st.selectbox("Select Quarantined Component ID to Inspect:", defect_ids)
        target = defect_hardware[defect_hardware['Component_ID'] == selected_defect_id].iloc[0]
        
        card_col1, card_col2, card_col3, card_col4 = st.columns(4)
        card_col1.metric("Component ID", target['Component_ID'])
        card_col2.metric("Pre-Burn Leakage", f"{target['Pre_BurnIn_uA']} µA")
        card_col3.metric("Lot Outlier Score", f"+{target['Z_Score']:.2f} σ")
        card_col4.metric("Infant Failure Risk", f"{target['Risk_Probability']:.1f}%")
        
        st.markdown("##### 🔬 Root-Cause Technical Diagnostic:")
        if target['Defect_Category'] == 'Latent Defect (Outlier)':
            st.error(
                f"🚨 **LATENT DEFECT DETECTED:** While this chip passed the static threshold ({target['Pre_BurnIn_uA']} µA <= {user_static_limit} µA), "
                f"it deviates by **+{target['Z_Score']:.2f}σ** above its manufacturing lot mean. "
                f"**Root Cause:** Early microscopic gate oxide pinholes or substrate micro-cracks. "
                f"**Failure Mode:** Severe thermal runaway / infant mortality under orbital temperature cycling. "
                f"**Recommendation:** DO NOT deploy to flight payload. Route to physical destruct lab."
            )
        else:
            st.error(
                f"❌ **HARD STATIC FAILURE:** Leakage current ({target['Pre_BurnIn_uA']} µA) exceeded allowable ceiling ({user_static_limit} µA). "
                f"**Root Cause:** Short-circuit or catastrophic junction breakdown. "
                f"**Recommendation:** Immediate scrap."
            )
    else:
        st.success("No defective hardware found in this manufacturing lot!")

# --- 3. REAL-TIME BURN-IN CHAMBER SIMULATOR ---
elif menu == "⚡ Real-Time Burn-In Chamber Sim":
    st.subheader("Thermal Stress Chamber Telemetry Simulator")
    col_s1, col_s2, col_s3 = st.columns(3)
    start_sim = col_s1.button("▶️ Start 168-Hr Chamber Stress Run")
    chamber_temp = col_s2.empty()
    chamber_status = col_s3.empty()
    plot_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    if start_sim:
        sample_size = min(15, len(processed_df)) if len(processed_df) > 0 else 1
        sim_df = processed_df.sample(sample_size, random_state=42).copy()
        for step in range(1, 101):
            temp = 25 + (step / 100.0) * 100.0
            chamber_temp.metric("Chamber Core Temp", f"{temp:.1f} °C")
            chamber_status.info("🔥 Thermal Bake In Progress...")
            progress_bar.progress(step)
            sim_df['Sim_Current_uA'] = sim_df['Pre_BurnIn_uA'] + (temp - 25) * 0.15 * (sim_df['Risk_Probability'] / 50.0)
            live_fig = px.bar(
                sim_df, 
                x='Component_ID', 
                y='Sim_Current_uA',
                color='Sim_Current_uA',
                color_continuous_scale='Turbo',
                range_y=[0, 80],
                title=f"Telemetry Stream at {temp:.1f} °C (Static Fail Ceiling = {user_static_limit} µA)"
            )
            live_fig.add_hline(y=user_static_limit, line_dash="dash", line_color="red")
            plot_placeholder.plotly_chart(live_fig, use_container_width=True)
            time.sleep(0.04)
        chamber_status.success("✅ Chamber Run Completed!")
        st.balloons()
    else:
        st.info("Click 'Start 168-Hr Chamber Stress Run' above to watch live thermal stress acceleration in action.")

# --- 4. WAFER SPATIAL HEATMAP ---
elif menu == "🗺️ Wafer Spatial Defect Heatmap":
    st.subheader("Silicon Wafer Spatial Anomaly Mapping")
    col_w1, col_w2 = st.columns([3, 1])
    with col_w1:
        w_fig = px.scatter(
            processed_df,
            x='Wafer_X',
            y='Wafer_Y',
            color='Defect_Category',
            color_discrete_map={'Flight Qualified': '#10ac84', 'Latent Defect (Outlier)': '#ee5253', 'Catastrophic Fail': '#222f3e'},
            size='Pre_BurnIn_uA',
            hover_data=['Component_ID', 'Pre_BurnIn_uA', 'Z_Score'],
            title="Interactive Silicon Wafer Die Map"
        )
        w_fig.update_layout(width=650, height=650, plot_bgcolor="#f1f2f6")
        st.plotly_chart(w_fig, use_container_width=True)
    with col_w2:
        st.markdown("#### Wafer Yield Diagnostics")
        st.metric("Total Dies on Wafer", len(processed_df))
        st.metric("Dies Passed (Green)", (processed_df['Defect_Category'] == 'Flight Qualified').sum())
        st.metric("Spatial Rejections (Red)", (processed_df['Defect_Category'] != 'Flight Qualified').sum())

# --- 5. EXPLAINABLE AI DIAGNOSTICS ---
elif menu == "🧠 Explainable AI (XAI) Diagnostics":
    st.subheader("Component Deep-Dive Root Cause Explainer")
    comp_list = list(processed_df['Component_ID'])
    selected_comp = st.selectbox("Select Component ID for Audit:", comp_list)
    comp_row = processed_df[processed_df['Component_ID'] == selected_comp].iloc[0]
    st.markdown("---")
    x1, x2, x3 = st.columns(3)
    x1.metric("Pre-Burn Leakage", f"{comp_row['Pre_BurnIn_uA']:.2f} µA")
    x2.metric("Lot-Relative Z-Score", f"{comp_row['Z_Score']:.2f} σ")
    x3.metric("Infant Mortality Risk", f"{comp_row['Risk_Probability']:.1f}%")
    st.markdown("### 📋 Root Cause Forensic Breakdown")
    if comp_row['Defect_Category'] == 'Flight Qualified':
        st.success(f"✅ **CERTIFIED QUALIFIED:** Nominal statistical behavior.")
    elif comp_row['Defect_Category'] == 'Latent Defect (Outlier)':
        st.error(f"⚠️ **REJECTED (LATENT DEFECT):** Passed static limit, but anomalous deviation.")
    else:
        st.error(f"❌ **HARD REJECTION:** Static leakage exceeded ceiling.")

# --- 6. LIFETIME RELIABILITY ---
elif menu == "⏳ Thermal Stress & Lifetime Reliability":
    st.subheader("Mission Lifetime Reliability & Arrhenius Degradation")
    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        hours = np.linspace(0, 10000, 100)
        baseline_survival = np.exp(-0.00003 * hours) * 100
        degraded_survival = np.exp(-0.00018 * hours) * 100
        surv_fig = go.Figure()
        surv_fig.add_trace(go.Scatter(x=hours, y=baseline_survival, mode='lines', name='Flight Qualified Units', line=dict(color='#10ac84', width=3)))
        surv_fig.add_trace(go.Scatter(x=hours, y=degraded_survival, mode='lines', name='Latent Defect Suspects', line=dict(color='#ee5253', width=3, dash='dash')))
        surv_fig.update_layout(title="Projected On-Orbit Survival Rate (%) Over 10,000 Hours", xaxis_title="Flight Hours", yaxis_title="Survival Probability (%)")
        st.plotly_chart(surv_fig, use_container_width=True)
    with col_r2:
        st.metric("Lot Thermal Drift (Mean)", f"+{processed_df['Drift_uA'].mean():.2f} µA")
        st.metric("Worst Thermal Drift", f"+{processed_df['Drift_uA'].max():.2f} µA", delta="Degraded", delta_color="inverse")

# --- 7. LIVE AUDIO ALERTS & QUARANTINE HUB ---
elif menu == "🚨 Live Audio Alerts & Quarantine Hub":
    st.subheader("Automated Sound Annunciator & Physical Pick-and-Place Quarantine")
    latent_defects_df = processed_df[processed_df['Defect_Category'] == 'Latent Defect (Outlier)']
    defect_count = len(latent_defects_df)
    col_a1, col_a2 = st.columns([2, 1])
    with col_a1:
        st.markdown("#### 🔊 Audio Annunciation")
        if defect_count > 0:
            st.error(f"🚨 **CRITICAL ALERT:** {defect_count} Latent Defects identified in Batch.")
            st.components.v1.html(
                """
                <script>
                function playAlert() {
                    var ctx = new (window.AudioContext || window.webkitAudioContext)();
                    var osc = ctx.createOscillator();
                    var gain = ctx.createGain();
                    osc.type = 'sawtooth';
                    osc.frequency.setValueAtTime(800, ctx.currentTime);
                    osc.frequency.exponentialRampToValueAtTime(400, ctx.currentTime + 0.5);
                    gain.gain.setValueAtTime(0.3, ctx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
                    osc.connect(gain);
                    gain.connect(ctx.destination);
                    osc.start();
                    osc.stop(ctx.currentTime + 0.5);
                }
                playAlert();
                </script>
                <button onclick="playAlert()" style="background-color:#ee5253; color:white; border:none; padding:10px 18px; border-radius:5px; font-weight:bold; cursor:pointer;">
                    🔊 Replay Cleanroom Siren
                </button>
                """,
                height=70
            )
        else:
            st.success("✅ Cleanroom Status: Zero anomalies detected in lot.")
    with col_a2:
        st.markdown("#### 📦 Robotic Quarantine Export")
        quarantine_csv = latent_defects_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Quarantined Dies (CSV)",
            data=quarantine_csv,
            file_name="Quarantined_Dies_Report.csv",
            mime="text/csv",
            use_container_width=True
        )
    st.markdown("---")
    st.dataframe(latent_defects_df[['Component_ID', 'Lot_ID', 'Wafer_X', 'Wafer_Y', 'Pre_BurnIn_uA', 'Z_Score', 'Risk_Probability']], use_container_width=True)

# --- 8. DATA INGESTION (PERSISTENT STATE FIX) ---
elif menu == "📥 Data Ingestion (Upload / Manual)":
    st.subheader("Data Management & Dynamic Batch Ingestion")
    
    if st.session_state.get('is_custom_uploaded', False):
        st.info(f"Currently active custom batch has **{len(st.session_state.dataset)} components** loaded.")
    
    t1, t2 = st.tabs(["Upload CSV/Excel", "Manual Entry"])
    with t1:
        up_file = st.file_uploader("Upload screening file (.csv, .xlsx)", type=['csv', 'xlsx'])
        if up_file is not None:
            up_df = pd.read_csv(up_file) if up_file.name.endswith('.csv') else pd.read_excel(up_file)
            st.write("Preview of Uploaded File:")
            st.dataframe(up_df.head(5), use_container_width=True)
            
            cols = list(up_df.columns)
            id_guess = next((c for c in cols if 'comp' in c.lower() or 'id' in c.lower() or 'chip' in c.lower()), cols[0])
            lot_guess = next((c for c in cols if 'lot' in c.lower() or 'batch' in c.lower()), cols[0])
            leak_guess = next((c for c in cols if 'pre' in c.lower() or 'leak' in c.lower() or 'current' in c.lower() or 'ua' in c.lower()), cols[-1])
            
            col_m1, col_m2, col_m3 = st.columns(3)
            c_id = col_m1.selectbox("Component ID Column:", cols, index=cols.index(id_guess))
            c_lot = col_m2.selectbox("Lot ID Column:", cols, index=cols.index(lot_guess))
            c_val = col_m3.selectbox("Pre-Burn Leakage Column:", cols, index=cols.index(leak_guess))
            
            if st.button("🚀 Ingest & Apply to All AstraBurn Features"):
                clean_pre = pd.to_numeric(up_df[c_val], errors='coerce').fillna(0.0)
                n = len(up_df)
                
                # Check for existing wafer / post columns, otherwise auto-populate
                x_coords = up_df['Wafer_X'] if 'Wafer_X' in up_df.columns else np.random.randint(-6, 7, n)
                y_coords = up_df['Wafer_Y'] if 'Wafer_Y' in up_df.columns else np.random.randint(-6, 7, n)
                post_vals = pd.to_numeric(up_df['Post_BurnIn_uA'], errors='coerce') if 'Post_BurnIn_uA' in up_df.columns else (clean_pre + np.random.normal(0.4, 0.2, n))
                
                new_dataset = pd.DataFrame({
                    'Component_ID': up_df[c_id].astype(str),
                    'Lot_ID': up_df[c_lot].astype(str) if c_lot != c_id else 'Uploaded_Lot',
                    'Wafer_X': x_coords,
                    'Wafer_Y': y_coords,
                    'Pre_BurnIn_uA': clean_pre,
                    'Post_BurnIn_uA': post_vals.round(2),
                    'Vth_Volts': 1.8,
                    'Vbr_Volts': 32.0
                })
                
                # Update persistent state
                st.session_state.dataset = new_dataset
                st.session_state.is_custom_uploaded = True
                st.success(f"✅ Successfully loaded {n} components! Go to any tab (Dashboard, Dual-Bin, etc.) to see results.")
                st.rerun()

    with t2:
        with st.form("manual_entry_form"):
            st.write("Add individual component manually to current batch:")
            cid = st.text_input("Component ID", f"COMP_{len(st.session_state.dataset)+1:03d}")
            lid = st.selectbox("Lot ID", ["Lot_L10", "Lot_L11", "Lot_Custom"])
            pre = st.number_input("Pre-Burn Leakage Current (µA)", value=12.5)
            post = st.number_input("Post-Burn Leakage Current (µA)", value=13.0)
            if st.form_submit_button("Add to Batch"):
                new_entry = pd.DataFrame([{
                    'Component_ID': cid, 
                    'Lot_ID': lid, 
                    'Wafer_X': np.random.randint(-5, 6), 
                    'Wafer_Y': np.random.randint(-5, 6), 
                    'Pre_BurnIn_uA': pre, 
                    'Post_BurnIn_uA': post, 
                    'Vth_Volts': 1.8, 
                    'Vbr_Volts': 32.0
                }])
                st.session_state.dataset = pd.concat([st.session_state.dataset, new_entry], ignore_index=True)
                st.success("Component appended to batch!")
                st.rerun()

# --- 9. LOT VARIATION ---
elif menu == "📈 Lot Variation & Distribution":
    st.subheader("Lot-to-Lot Spread Analysis")
    b_fig = px.box(processed_df, x='Lot_ID', y='Pre_BurnIn_uA', color='Lot_ID', title="Spread across Lots")
    st.plotly_chart(b_fig, use_container_width=True)
    summary = processed_df.groupby('Lot_ID')['Pre_BurnIn_uA'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    st.dataframe(summary.rename(columns={'count': 'Total Units', 'mean': 'Mean (µA)', 'std': 'Std Dev (σ)'}), use_container_width=True)

# --- 10. BURN-IN DRIFT ---
elif menu == "🔥 Burn-In Drift Prediction":
    st.subheader("Burn-In Degradation Drift (Isolation Forest ML)")
    features = processed_df[['Pre_BurnIn_uA', 'Post_BurnIn_uA', 'Drift_uA']].fillna(0)
    iso = IsolationForest(contamination=0.03, random_state=42)
    processed_df['Drift_Anomaly'] = iso.fit_predict(features) == -1
    d_fig = px.scatter(
        processed_df, 
        x='Pre_BurnIn_uA', 
        y='Post_BurnIn_uA', 
        color='Drift_Anomaly',
        color_discrete_map={False: '#0984e3', True: '#d63031'}, 
        hover_data=['Component_ID', 'Lot_ID'],
        title="Pre-Burn vs Post-Burn Thermal Drift"
    )
    st.plotly_chart(d_fig, use_container_width=True)
