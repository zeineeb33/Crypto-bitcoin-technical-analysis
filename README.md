# BTC-Trend-Analyzer

## Introduction  
This project performs a **technical analysis** of Bitcoin (**BTC-USD**) using historical market data from the past two years.  
The goal is to **retrieve, visualize, and analyze** key market indicators to better understand price trends, trading volumes, and volatility.  

We use **Yahoo Finance’s API** (via the `yfinance` Python library) to collect the data and apply several **financial analysis techniques**, including:  
- Price trend visualization  
- Moving Averages (**SMA** & **EMA**)  
- Technical Indicators (**MACD**, trading signals)  
- Volatility Analysis  
- Volume-to-price correlation  
- Monthly performance tracking  

---

## Dataset Description  
The dataset contains daily Bitcoin market data with the following columns:  

| Column     | Description |
|------------|-------------|
| **Open**   | Price at the start of the trading day |
| **High**   | Highest price during the day |
| **Low**    | Lowest price during the day |
| **Close**  | Price at the end of the trading day |
| **Volume** | Number of Bitcoins traded during the day |
| **Dividends** | Dividends paid (always 0 for BTC) |
| **Stock Splits** | Stock splits (always 0 for BTC) |

---

git clone https://github.com/your-username/BTC-Trend-Analyzer.git
cd BTC-Trend-Analyzer
