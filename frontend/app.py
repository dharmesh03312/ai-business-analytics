import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="AI Business Analytics", page_icon="📊", layout="wide")

# 2. Custom CSS styling
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Main Header
st.title("📊 AI Business Analytics Dashboard")
st.markdown("Automated insights and performance tracking overview.")
st.divider()

API_BASE = "https://ai-business-analytics.onrender.com"

# 4. State Management
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None
if 'pdf_report' not in st.session_state:
    st.session_state.pdf_report = None

# 5. Sidebar Navigation / Controls
with st.sidebar:
    st.header("⚙️ Control Panel")
    st.write("Upload a CSV dataset and process it through the FastAPI backend.")
    st.write("") # spacing
    uploaded_file = st.file_uploader("Upload Sales CSV", type=["csv"])
    analyze_btn = st.button("🚀 Analyze Data", use_container_width=True, type="primary", disabled=uploaded_file is None)
    
    # Present isolated download button conditional on successful downstream fetch cache
    if st.session_state.pdf_report:
        st.write("")
        st.download_button(
            label="📄 Download PDF Report",
            data=st.session_state.pdf_report,
            file_name="Business_Analytics_Report.pdf",
            mime="application/pdf",
            type="secondary",
            use_container_width=True
        )

# 6. API Interaction
if analyze_btn and uploaded_file is not None:
    with st.spinner("Processing data through backend microservices..."):
        try:
            # Send file across all modular API endpoints seamlessly via internal buffer
            file_bytes = uploaded_file.getvalue()
            
            def make_file_payload():
                return {"file": ("data.csv", file_bytes, "text/csv")}
                
            resp_analyze = requests.post(f"{API_BASE}/analyze", files=make_file_payload(),timeout=10)
            resp_forecast = requests.post(f"{API_BASE}/forecast", files=make_file_payload(),timeout=10)
            resp_anomalies = requests.post(f"{API_BASE}/anomalies", files=make_file_payload(),timeout=10)
            resp_insights = requests.post(f"{API_BASE}/insights", files=make_file_payload(),timeout=10)
            
            if resp_analyze.status_code == 200:
                combined_data = resp_analyze.json()
                combined_data.update(resp_forecast.json() if resp_forecast.status_code == 200 else {})
                combined_data.update(resp_anomalies.json() if resp_anomalies.status_code == 200 else {})
                combined_data.update(resp_insights.json() if resp_insights.status_code == 200 else {})
                
                st.session_state.analyzed_data = combined_data
                st.toast('Analysis pipeline completely executed!', icon='✅')
                
                # Fetch dynamically compiled PDF structurally parsing the uploaded bytes identically via POST
                try:
                    pdf_request = requests.post(f"{API_BASE}/download-report", files=make_file_payload())
                    if pdf_request.status_code == 200:
                        st.session_state.pdf_report = pdf_request.content
                except Exception:
                    pass # Graceful fail state
            else:
                st.error(f"Backend API Error [{resp_analyze.status_code}]: {resp_analyze.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to backend! Ensure FastAPI is running on {API_BASE}. Exception: {e}")

# 7. Dashboard Layout rendering
if st.session_state.analyzed_data:
    data = st.session_state.analyzed_data
    
    total_sales = data.get("total_sales", 0)
    top_product = data.get("top_product", "N/A")
    monthly_sales = data.get("monthly_sales", {})
    top_products = data.get("top_products", {})
    forecast = data.get("forecast", [])
    anomalies = data.get("anomalies", [])
    insights = data.get("insights", "")
    
    # --- ALERTS & ANOMALIES ---
    if anomalies:
        st.error("🚨 **System Alert:** Significant statistical anomalies detected in historical dataset! (Variance breached ±2.0 Z-Score)")
        alert_cols = st.columns(min(len(anomalies), 4))
        for i, anomaly in enumerate(anomalies[:4]): # Display top 4 to maintain UI uniformity
            with alert_cols[i]:
                st.metric(label=f"Date: {anomaly['date']}", value=f"${anomaly['sales']:,.2f}", delta="Outlier")
        st.write("")
        st.divider()
        
    # --- EXECUTIVE SUMMARY ---
    st.header("📌 Executive Summary")
    st.write("") 
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.metric(label="Total Pipeline Revenue", value=f"${total_sales:,.2f}", delta="All Time")
    with m2:
        st.metric(label="Top Performing Product", value=top_product, delta="Rank #1")
    with m3:
        st.metric(label="Recognized Entities", value=len(top_products))
        
    st.write("")
    st.write("")
    
    # --- AI INSIGHTS ---
    st.header("🧠 Applied AI Insights")
    st.write("")
    # Determine the status nicely
    if "Failed" in insights or "Not Generated" in insights:
        st.warning(insights)
    else:
        # Wrap the LLM Markdown in a nice successful informative highlight
        st.info(insights)
        
    st.write("")
    st.write("")
    
    # --- VISUALIZATIONS ---
    st.header("📈 Deep Dive Analysis")
    st.write("")
    c1, c2 = st.columns(2, gap="large")
    
    with c1:
        st.subheader("Monthly Sales Trend")
        if monthly_sales:
            df_monthly = pd.DataFrame(list(monthly_sales.items()), columns=["Month", "Sales"])
            df_monthly = df_monthly.sort_values(by="Month") 
            
            fig_line = px.line(
                df_monthly, 
                x="Month", 
                y="Sales", 
                markers=True,
                line_shape="spline",
                template="plotly_white"
            )
            fig_line.update_traces(line_color="#1f77b4", line_width=3, marker_size=8)
            fig_line.update_layout(
                xaxis_title="Time Period", 
                yaxis_title="Revenue ($)", 
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Insufficient spatial data.")
            
    with c2:
        st.subheader("Top Selling Products")
        if top_products:
            df_products = pd.DataFrame(list(top_products.items()), columns=["Product", "Sales"])
            # Sort ascending for horizontal bar chart so highest is on top
            df_products = df_products.sort_values(by="Sales", ascending=True) 
            
            fig_bar = px.bar(
                df_products, 
                y="Product", 
                x="Sales", 
                orientation='h',
                text_auto='.2s',
                template="plotly_white",
                color="Sales",
                color_continuous_scale="Blues"
            )
            fig_bar.update_layout(
                xaxis_title="Total Revenue ($)", 
                yaxis_title="Product Tag", 
                margin=dict(l=0, r=0, t=20, b=0), 
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Insufficient clustering data.")

    # --- FORECASTING ---
    st.write("")
    st.divider()
    st.header("🔮 Sales Forecast (Linear Prediction)")
    st.write("")
    
    if forecast:
        df_forecast = pd.DataFrame(forecast)
        
        fig_forecast = px.line(
            df_forecast,
            x="date",
            y="predicted_sales",
            markers=True,
            line_shape="spline",
            template="plotly_white"
        )
        fig_forecast.update_traces(line_color="#ff7f0e", line_width=3, marker_size=6)
        fig_forecast.update_layout(
            xaxis_title="Predicted Date", 
            yaxis_title="Expected Revenue ($)", 
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.info("No forecast data generated.")

else:
    # --- EMPTY STATE ---
    st.info("👈 Please execute **Analyze Data** in the Control Panel to populate dashboard.")
    # Placeholder using UI design patterns
    st.markdown('''
        <div style="border: 2px dashed #d3d3d3; padding: 50px; text-align: center; border-radius: 10px; margin-top: 20px;">
            <h3 style="color: grey;">Waiting for initial execution...</h3>
            <p style="color: grey;">The analytical charts and dynamic matrices will materialize here.</p>
        </div>
    ''', unsafe_allow_html=True)
