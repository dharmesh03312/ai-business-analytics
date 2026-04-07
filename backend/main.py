import os
import io
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas
import textwrap


app = FastAPI(title="AI Business Analytics API")
PORT = int(os.environ.get("PORT", 8000))

def load_df(file: UploadFile):
    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # 1. Normalize column names (lowercase, strip whitespace)
        df.columns = [str(col).lower().strip() for col in df.columns]
        
        required_cols = ['sales', 'product', 'date']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Required columns {required_cols} are missing")
            
        initial_rows = len(df)
        
        # 2a. Clean 'product' column: String type, trim spaces, uppercase
        df['product'] = df['product'].astype(str).str.strip().str.upper()
        df = df[df['product'] != '']
        df = df[df['product'] != 'NAN']
        
        # 2b. Clean 'date' column: Parse safely parsing mix formats, coerce errors, drop NaT
        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce', dayfirst=True)
        date_valid = df['date'].notna()
        date_removed_count = initial_rows - date_valid.sum()
        
        # 2c. Clean 'sales' column: numeric only, >= 0, drop NaN
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        sales_valid = df['sales'].notna() & (df['sales'] >= 0)
        sales_removed_count = initial_rows - sales_valid.sum()
        
        # 4. Drop rows where critical fields lack data (apply validated masks)
        df = df[date_valid & sales_valid]
        
        # 3. Remove duplicate rows completely
        rows_before_dedup = len(df)
        df = df.drop_duplicates()
        duplicates_removed_count = rows_before_dedup - len(df)
        
        final_rows = len(df)
        rows_removed_total = initial_rows - final_rows
        
        # 5. Output quality tracking
        metadata = {
            "rows_removed": rows_removed_total,
            "rows_remaining": final_rows,
            "details": {
                "invalid_date": int(date_removed_count),
                "invalid_sales": int(sales_removed_count),
                "duplicates_dropped": int(duplicates_removed_count)
            }
        }
        
        if final_rows == 0:
            raise ValueError("All rows were removed during the cleaning process due to invalid formats.")
            
        return df, metadata
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

def get_forecast_data(df: pd.DataFrame):
    daily_sales = df.groupby('date')['sales'].sum().reset_index().sort_values('date')
    min_date = daily_sales['date'].min()
    daily_sales['days_since_start'] = (daily_sales['date'] - min_date).dt.days
    
    X = daily_sales['days_since_start'].values.reshape(-1, 1)
    y = daily_sales['sales'].values
    
    model = LinearRegression().fit(X, y)
    
    max_date = daily_sales['date'].max()
    next_30_dates = [max_date + pd.Timedelta(days=i) for i in range(1, 31)]
    next_30_days_since = np.array([(d - min_date).days for d in next_30_dates]).reshape(-1, 1)
    
    predictions = model.predict(next_30_days_since)
    return [{"date": d.strftime('%Y-%m-%d'), "predicted_sales": round(float(p), 2)} for d, p in zip(next_30_dates, predictions)]

def get_anomalies_data(df: pd.DataFrame):
    daily_sales = df.groupby('date')['sales'].sum().reset_index().sort_values('date')
    sales_mean = daily_sales['sales'].mean()
    sales_std = daily_sales['sales'].std()
    
    daily_sales['z_score'] = 0.0
    if sales_std > 0:
        daily_sales['z_score'] = (daily_sales['sales'] - sales_mean) / sales_std
        
    anomalies_df = daily_sales[daily_sales['z_score'].abs() > 2]
    return [{"date": row['date'].strftime('%Y-%m-%d'), "sales": round(float(row['sales']), 2)} for _, row in anomalies_df.iterrows()]

def generate_insights_text(total_sales, top_product, monthly_sales_dict, forecast_data, anomalies_data):
    # Rule-Based Insight Pipeline (Fallback/Default)
    months = sorted(list(monthly_sales_dict.keys()))
    trend_text = "remains stable"
    if len(months) >= 2:
        if monthly_sales_dict[months[-1]] > monthly_sales_dict[months[0]]:
            trend_text = "indicates an upward trajectory"
        elif monthly_sales_dict[months[-1]] < monthly_sales_dict[months[0]]:
            trend_text = "shows a downward decline"
            
    avg_forecast = sum(f['predicted_sales'] for f in forecast_data) / len(forecast_data) if forecast_data else 0
    anomaly_text = f"Risk Analysis highlights {len(anomalies_data)} statistical anomalies requiring attention." if anomalies_data else "Risk Analysis indicates stable historical variance with no significant anomalies."
    
    rule_based_insights = (
        f"**Key trends**: Overall revenue momentum {trend_text} across tracked periods, accumulating ${total_sales:,.2f} in total sales.\n\n"
        f"**Driving forces**: The item '{top_product}' remains the primary operational catalyst driving this volume.\n\n"
        f"**Risk Analysis**: {anomaly_text}\n\n"
        f"**Future Outlook**: Predictive models forecast a stable average of ${avg_forecast:,.2f} per day over the incoming 30-day timeline."
    )
    return rule_based_insights


@app.post("/analyze")
def analyze_sales(file: UploadFile = File(...)):
    df, metadata = load_df(file)
    df['month'] = df['date'].dt.strftime('%Y-%m')

    total_sales = float(df['sales'].sum())
    product_sales = df.groupby('product')['sales'].sum()
    top_product = str(product_sales.idxmax())
    
    top_5_products = product_sales.sort_values(ascending=False).head(5)
    top_products_dict = {str(k): float(v) for k, v in top_5_products.items()}
    
    monthly_sales = df.groupby('month')['sales'].sum()
    monthly_sales_dict = {str(k): float(v) for k, v in monthly_sales.items()}
    
    return {
        "total_sales": total_sales,
        "top_product": top_product,
        "monthly_sales": monthly_sales_dict,
        "top_products": top_products_dict,
        "data_quality": metadata
    }

@app.post("/forecast")
def forecast_sales(file: UploadFile = File(...)):
    df, _ = load_df(file)
    return {"forecast": get_forecast_data(df)}

@app.post("/anomalies")
def detect_anomalies(file: UploadFile = File(...)):
    df, _ = load_df(file)
    return {"anomalies": get_anomalies_data(df)}

@app.post("/insights")
def generate_insights(file: UploadFile = File(...)):
    try:
        df, _ = load_df(file)
        
        # Build necessary context
        total_sales = float(df['sales'].sum())
        top_product = str(df.groupby('product')['sales'].sum().idxmax())
        df['month'] = df['date'].dt.strftime('%Y-%m')
        monthly_sales_dict = {str(k): float(v) for k, v in df.groupby('month')['sales'].sum().items()}
        
        forecast_data = get_forecast_data(df)
        anomalies_data = get_anomalies_data(df)
        
        insights_text = generate_insights_text(
            total_sales, top_product, monthly_sales_dict, forecast_data, anomalies_data
        )
        
        return {"insights": insights_text}
    except Exception as e:
        return {"insights": f"⚠️ **Analytics Core Failed:** unable to process insights ({str(e)})."}

@app.post("/download-report")
def download_report(file: UploadFile = File(...)):
    try:
        df, _ = load_df(file)
        
        total_sales = float(df['sales'].sum())
        top_product = str(df.groupby('product')['sales'].sum().idxmax())
        df['month'] = df['date'].dt.strftime('%Y-%m')
        monthly_sales = {str(k): float(v) for k, v in df.groupby('month')['sales'].sum().items()}
        forecast = get_forecast_data(df)
        anomalies = get_anomalies_data(df)
        
        # Fetch structured rule-based or AI generated summary flawlessly
        insights_text = generate_insights_text(
            total_sales, top_product, monthly_sales, forecast, anomalies
        )
        
        # Platypus Document Construction Statelessly using BytesIO
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
        
        # Structural PDF Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(name='TitleStyle', parent=styles['Heading1'], alignment=1, fontSize=24, spaceAfter=20, textColor=colors.black)
        heading_style = ParagraphStyle(name='HeadingStyle', parent=styles['Heading2'], fontSize=16, spaceBefore=20, spaceAfter=10, textColor=colors.darkblue)
        normal_style = ParagraphStyle(name='NormalStyle', parent=styles['Normal'], fontSize=11, spaceAfter=8, leading=16)
        
        elements = []
        
        # 1. Main Title
        elements.append(Paragraph("<b>Business Analytics Report</b>", title_style))
        elements.append(Spacer(1, 10))
        
        # 2. Executive Summary - AI Insights
        elements.append(Paragraph("<b>AI Executive Summary</b>", heading_style))
        # Ensure LLM markdown cleanly translates visually 
        clean_insights = insights_text.replace('\n', '<br/>')
        elements.append(Paragraph(clean_insights, normal_style))
        elements.append(Spacer(1, 10))
        
        # 3. Key Metrics
        elements.append(Paragraph("<b>Key Quantitative Metrics</b>", heading_style))
        elements.append(Paragraph(f"<b>Total Sales Volume:</b> ${total_sales:,.2f}", normal_style))
        elements.append(Paragraph(f"<b>Top Performing Product:</b> {top_product}", normal_style))
        
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("<b>Monthly Tranches:</b>", normal_style))
        monthly_table_data = [["Month", "Total Sales Volume ($)"]]
        for k, v in monthly_sales.items():
            monthly_table_data.append([str(k), f"${v:,.2f}"])
            
        m_table = Table(monthly_table_data, colWidths=[120, 150], hAlign='LEFT')
        m_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ]))
        elements.append(m_table)
        elements.append(Spacer(1, 15))
        
        # 4. Forecast Summary
        elements.append(Paragraph("<b>Predictive Forecast (Sample Output)</b>", heading_style))
        forecast_table_data = [["Target Date", "Algorithmic Prediction ($)"]]
        for f in forecast[:5]:
            forecast_table_data.append([f['date'], f"${f['predicted_sales']:,.2f}"])
            
        f_table = Table(forecast_table_data, colWidths=[120, 150], hAlign='LEFT')
        f_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
        ]))
        elements.append(f_table)
        elements.append(Spacer(1, 15))
        
        # 5. Anomalies
        elements.append(Paragraph("<b>Statistical Anomalies</b>", heading_style))
        if anomalies:
            anom_table_data = [["Flagged Date", "Actual Sales Variance ($)"]]
            for a in anomalies[:5]:
                anom_table_data.append([a['date'], f"${a['sales']:,.2f}"])
                
            a_table = Table(anom_table_data, colWidths=[120, 150], hAlign='LEFT')
            a_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.mistyrose),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
            ]))
            elements.append(a_table)
            
            if len(anomalies) > 5:
                elements.append(Paragraph(f"<i>(+ {len(anomalies)-5} additional outliers truncated for layout tracking)</i>", normal_style))
        else:
            elements.append(Paragraph("No significant variations detected outside ±2.0 Z-Score tolerances.", normal_style))
            
        # Safely compile dynamic Platypus schema explicitly targeting the buffer
        doc.build(elements)
        
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return Response(
            content=pdf_bytes, 
            media_type="application/pdf", 
            headers={"Content-Disposition": "attachment; filename=Business_Analytics_Report.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
