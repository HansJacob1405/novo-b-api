from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, base64

app = FastAPI(title="Novo B Data API")

TICKER = "NOVO-B.CO"  # Novo Nordisk B på Nasdaq Copenhagen


class NovoRequest(BaseModel):
    period: str = "1y"
    interval: str = "1d"


@app.post("/novo_b_chart")
def novo_b_chart(req: NovoRequest):
    try:
        df = yf.download(
            TICKER,
            period=req.period,
            interval=req.interval,
            progress=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"yfinance fejl: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=404, detail="Ingen data retur fra yfinance")

    # Reset index for 'Date'
    df = df.reset_index()

    # Sortér datoer
    df = df.sort_values("Date")

    # Begræns til sidste 365 dage
    last_date = df["Date"].max()
    cutoff = last_date - pd.Timedelta(days=365)
    df12 = df[df["Date"] >= cutoff].copy()

    if df12.empty:
        raise HTTPException(status_code=404, detail="Ingen data i de sidste 12 måneder")

    # 30-dages glidende gennemsnit
    df12["MA30"] = df12["Close"].rolling(30, min_periods=1).mean()

    # Trendlinje
    x = np.arange(len(df12))
    y = df12["Close"].values
    slope, intercept = np.polyfit(x, y, 1)
    df12["Trend"] = intercept + slope * x

    # Opsummering
    start_date = df12["Date"].min()
    end_date = df12["Date"].max()
    start_price = float(df12.iloc[0]["Close"])
    end_price = float(df12.iloc[-1]["Close"])
    pct_change = (end_price / start_price - 1) * 100

    # Plot
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(df12["Date"], df12["Close"], label="Close")
    ax.plot(df12["Date"], df12["MA30"], label="MA30")
    ax.plot(df12["Date"], df12["Trend"], label="Trend", linestyle="--")
    ax.set_title("Novo B – Daglige lukkepriser (12 mdr)")
    ax.set_xlabel("Dato")
    ax.set_ylabel("Kurs (DKK)")
    ax.legend()
    fig.autofmt_xdate()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "ticker": TICKER,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "start_price": start_price,
        "end_price": end_price,
        "pct_change": pct_change,
        "image_base64": img_b64,
        "preview": df12.tail(5).to_dict(orient="records")
    }
