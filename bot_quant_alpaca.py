import os, asyncio, warnings, requests, math
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from dotenv import load_dotenv
from flask import Flask
from threading import Thread

# Modelos Estadísticos y ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# Ejecución y APIs
import alpaca_trade_api as tradeapi
from telegram import Bot
from telegram.constants import ParseMode

warnings.filterwarnings("ignore")
load_dotenv()

# ========================= CONFIGURACIÓN =====================

# Credenciales (Configurar en Render Environment Variables)
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets" # Cambiar a https://api.alpaca.markets para Real

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Parámetros Operativos
UMBRAL_SCORE = 75
RISK_PER_TRADE = 0.05  # 5% del capital por operación
UMBRAL_RVOL = 1.5      # Volumen relativo mínimo para breakout

# Inicialización
alpaca = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, BASE_URL, api_version='v2')
telegram_bot = Bot(token=TOKEN) if TOKEN and CHAT_ID else None

# ========================= SERVIDOR FLASK (24/7) =============

app = Flask('')

@app.route('/')
def home():
    return "Bot Quant Operativo 24/7"

def run_flask():
    app.run(host='0.0.0.0', port=8080)

# ========================= ESCÁNER NASDAQ =====================

def get_nasdaq_candidates():
    """Escanea tickers con volumen inusual y ruptura de resistencia."""
    # Lista de alta liquidez para el escáner
    tickers = ["AAPL", "NVDA", "TSLA", "AMD", "MSFT", "META", "AMZN", "GOOGL", "NFLX"]
    cryptos = ["BTCUSD", "ETHUSD", "SOLUSD"]
    
    candidatos = []
    print(f"🔍 Escaneando {len(tickers)} activos del Nasdaq...")
    
    for symbol in tickers + cryptos:
        try:
            bars = alpaca.get_bars(symbol, tradeapi.TimeFrame.Day, 
                                 limit=20).df
            if bars.empty: continue
            
            # Cálculo de RVOL (Volumen Relativo)
            avg_vol = bars['volume'].tail(15).mean()
            current_vol = bars['volume'].iloc[-1]
            rvol = current_vol / avg_vol
            
            # Detección de Breakout (Cierre > Max 15 días previos)
            resistance = bars['high'].tail(15).iloc[:-2].max()
            current_price = bars['close'].iloc[-1]
            
            if current_price > resistance and rvol > UMBRAL_RVOL:
                candidatos.append(symbol)
                print(f"🔥 Breakout en {symbol} | RVOL: {rvol:.2f}")
        except Exception as e:
            continue
            
    return candidatos

# ========================= MOTOR QUANT ========================

def aplicar_indicadores(df):
    """Añade indicadores técnicos al DataFrame."""
    df['RSI'] = ta.rsi(df['close'], 14)
    df['SMA20'] = ta.sma(df['close'], 20)
    df['SMA50'] = ta.sma(df['close'], 50)
    macd = ta.macd(df['close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['RET'] = np.log(df['close']/df['close'].shift(1))
    return df.dropna()

def engine_analisis(df):
    """Ejecuta ARIMA, GARCH y Scoring."""
    # 1. ARIMA Trend
    try:
        model_arima = ARIMA(df['close'].tail(30), order=(1,1,0)).fit()
        pred = model_arima.forecast(5).mean()
        trend = "ALCISTA" if pred > df['close'].iloc[-1] else "BAJISTA"
    except: trend = "NEUTRAL"

    # 2. GARCH Volatilidad
    returns = df['RET'].tail(120) * 100
    try:
        model_garch = arch_model(returns, p=1, q=1)
        res = model_garch.fit(disp="off")
        vol_forecast = np.sqrt(res.forecast(horizon=1).variance.values[-1][0])
    except: vol_forecast = 40.0

    return trend, vol_forecast

class MLPredictor:
    """Modelo de Machine Learning para probabilidad de éxito."""
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()

    def get_prob(self, df):
        df['Target'] = (df['close'].shift(-3) > df['close']).astype(int)
        X = df[['RSI','MACD','SMA20','SMA50']]
        y = df['Target']
        Xs = self.scaler.fit_transform(X[:-1])
        self.model.fit(Xs, y[:-1])
        return self.model.predict_proba(self.scaler.transform(X[-1:]))[0][1]

# ========================= RIESGO Y EJECUCIÓN =================

def calcular_niveles(precio, vol, prob):
    """Calcula SL/TP dinámicos basados en volatilidad GARCH."""
    # Multiplicador basado en volatilidad diaria proyectada
    distancia = (vol / 100) * 1.5
    factor_tp = 1.3 if prob > 0.70 else 1.0
    
    sl = precio * (1 - distancia)
    tp = precio * (1 + (distancia * 2 * factor_tp))
    return round(sl, 2), round(tp, 2)

async def enviar_telegram(msg):
    if telegram_bot:
        await telegram_bot.send_message(CHAT_ID, msg, parse_mode=ParseMode.MARKDOWN)

def ejecutar_orden(symbol, qty, sl, tp):
    """Ejecuta orden Bracket en Alpaca (Entrada + SL + TP)."""
    try:
        alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side='buy',
            type='market',
            time_in_force='gtc',
            order_class='bracket',
            stop_loss={'stop_price': sl},
            take_profit={'limit_price': tp}
        )
        print(f"✅ ORDEN ENVIADA: {symbol} | SL: {sl} | TP: {tp}")
    except Exception as e:
        print(f"❌ Error Alpaca: {e}")

# ========================= MAIN LOOP ==========================

async def main():
    print("🚀 BOT QUANT INICIADO")
    ml = MLPredictor()
    
    while True:
        try:
            # 1. Obtener candidatos del Nasdaq y Cryptos
            candidatos = get_nasdaq_candidates()
            
            # 2. Revisar Capital
            account = alpaca.get_account()
            cash = float(account.cash)
            
            for ticker in candidatos:
                # Obtener data histórica
                bars = alpaca.get_bars(ticker, tradeapi.TimeFrame.Day, limit=150).df
                if bars.empty: continue
                
                df = aplicar_indicadores(bars)
                trend, vol = engine_analisis(df)
                prob = ml.get_prob(df)
                
                # Calcular Score
                score = 0
                score += 30 if prob > 0.65 else 10
                score += 30 if trend == "ALCISTA" else 0
                score += 20 if vol < 35 else 5
                score += 20 if 40 < df['RSI'].iloc[-1] < 65 else 5
                
                if score >= UMBRAL_SCORE:
                    precio = df['close'].iloc[-1]
                    sl, tp = calcular_niveles(precio, vol, prob)
                    
                    # Calcular cantidad (Risk Mgmt)
                    monto_invertir = cash * RISK_PER_TRADE
                    qty = math.floor(monto_invertir / precio)
                    
                    if qty > 0:
                        # Verificar si ya existe posición
                        try:
                            alpaca.get_position(ticker)
                            print(f"⚠️ Posición activa en {ticker}, saltando...")
                        except:
                            ejecutar_orden(ticker, qty, sl, tp)
                            await enviar_telegram(f"🔔 *COMPRA EJECUTADA*\nTicker: {ticker}\nPrecio: {precio}\nSL: {sl} | TP: {tp}\nScore: {score}")

        except Exception as e:
            print(f"⚠️ Error en loop: {e}")
            
        print("⏳ Ciclo completado. Esperando 30 min...")
        await asyncio.sleep(1800)

if __name__ == "__main__":
    # Iniciar Servidor Flask en un hilo separado para Render
    Thread(target=run_flask).start()
    # Iniciar Bot
    asyncio.run(main())
