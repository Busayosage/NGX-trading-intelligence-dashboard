# 📈 NGX Trading Intelligence Dashboard

An AI-powered stock analytics and quantitative trading dashboard for the Nigerian Stock Exchange (NGX).

---

## 🚀 Overview

This project is a full-stack data analytics and quant research system that:

- Processes historical stock data
- Generates trading signals (BUY / SELL / HOLD)
- Builds a ranking engine
- Simulates a dynamic portfolio strategy
- Provides risk and performance analytics
- Includes an AI assistant for market insights

---

## 🧠 Key Features

### 📊 Market Intelligence
- Top-performing stocks
- BUY signal detection
- Risk classification
- Multi-stock comparison

### 🧮 Quant Engine
- Cross-sectional ranking model
- Momentum + volatility + forecast scoring
- Confidence scoring system

### 📈 Portfolio Backtesting
- Dynamic equal-weight portfolio
- Weekly rebalancing
- Transaction cost modelling
- Minimum holding period logic
- Turnover control (Step 5A & 5B)

### 🤖 AI Assistant
- Ask questions like:
  - "Top BUY stocks?"
  - "Compare GTCO and DANGSUGAR"
  - "What is the riskiest stock?"

---

## 🏗️ Tech Stack

- Python
- Pandas / NumPy
- Plotly
- Streamlit
- Quantitative Finance Logic

---

## 📂 Project Structure


stock-market-analytics/
│
├── data/ # Raw stock data
├── outputs/ # Processed outputs & backtests
├── scripts/ # Data processing scripts
├── dashboard.py # Main Streamlit dashboard
├── EDA.py # Exploratory data analysis
├── requirements.txt
└── README.md


---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run dashboard.py
📌 Project Goal

To build a quantitative trading intelligence system that evolves into:

Portfolio analytics tool
AI-powered trading assistant
NGX quant research platform
📈 Future Improvements
Factor models (Fama-French style)
Walk-forward testing
Sector rotation strategies
ML-based signal prediction
👤 Author

Seun Oseola
Data Analyst | Quant Analytics Enthusiast
