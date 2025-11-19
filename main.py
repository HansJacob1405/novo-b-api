from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
import io
import base64

app = FastAPI(title="Novo B Data API")

TICKER = "NOVO-B.CO"  # Novo Nordisk B på Nasdaq Copenhagen


class NovoRequest(BaseModel):
    period: str = "1y"
    interval: str = "1d"


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    """Root-endpoint så Render får et hurtigt 200 OK (både GET og HEAD)."""
    return {"status": "ok", "message": "Novo B API is running"}


@app.get("/health")
def health():
    """Simpel health-check."""
    return {"status": "ok"}


@app.post("/novo_b_chart")
def novo_b_chart(req: NovoRequest):
    # Importér matplotlib "lazy", så opstarten bliver hurtigere
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        # 1) Hent data via yfinance
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

        # 2) Forbered data
        df = df.reset_index()  # gør index til kolonne(r)

        # --- NY LOGIK: find datokolonnen robust ---
        date_col = None

        # a) Find en kolonne der allerede er datetime64
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.datetime64):
                date_col = col
                break

        # b) Hvis ingen datetime-kolonne, prøv første kolonne som dato
        if date_col is None:
            first_col = df.columns[0]
            try:
                df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
                if df[first_col].notna().any():
                    date_col = first_col
            except Exception:
                pass

        if date_col is None:
            raise HTTPException(status_code=500, detail="Ingen datokolonne fundet i data fra yfinance")

        # c) Omdøb til 'Date' hvis nødvendigt
        if date_col != "Date":
            df.rename(columns={date_col: "Date"}, inplace=True)

        # Fjern rækker uden gyldig dato
        df = df.dropna(subset=["Date"])
        if df.empty:
            raise HTTPException(status_code=404, detail="Ingen gyldige datoer i data")

        if "Close" not in df.columns:
            raise HTTPException(status_code=500, detail="Close-kolonne ikke fundet i data")

        # Sorter kronologisk
        df = df.sort_values("Date")

        # Begræns til sidste 365 dage
        last_date = df["Date"].max()
        cutoff = last_date - pd.Timedelta(days=365)
        df12 = df[df["Date"] >= cutoff].copy()

        if df12.empty:
            raise HTTPException(status_code=404, detail="Ingen data i de sidste 12 måneder")

        # 3) 30-dages glidende gennemsnit
        df12["MA30"] = df12["Close"].rolling(window=30, min_periods=1).mean()

        # 4) Lineær trendlinje
        x = np.arange(len(df12))
        y = df12["Close"].values

        if len(df12) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            df12["Trend"] = intercept + slope * x
        else:
            df12["Trend"] = df12["Close"]

        # 5) Opsummering
        start_date = df12["Date"].min()
        end_date = df12["Date"].max()
        start_price = float(df12.iloc[0]["Close"])
        end_price = float(df12.iloc[-1]["Close"])
        pct_change = (end_price / start_price - 1) * 100 if start_price != 0 else 0.0

        # 6) Plot
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
        plt.close(fig)

        # 7) JSON-venlig preview (dato som streng)
        df_preview = df12.copy()
        df_preview["Date"] = df_preview["Date"].dt.strftime("%Y-%m-%d")
        preview_rows = df_preview.tail(5)[["Date", "Close", "MA30", "Trend"]].to_dict(orient="records")

        return {
            "ticker": TICKER,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "start_price": start_price,
            "end_price": end_price,
            "pct_change": pct_change,
            "image_base64": img_b64,
            "preview": preview_rows,
        }

    except HTTPException:
        # HTTPException skal bare sendes videre uændret
        raise
    except Exception as e:
        # Alle andre fejl sendes som JSON i 'detail'
        raise HTTPException(status_code=500, detail=f"Serverfejl: {str(e)}")
