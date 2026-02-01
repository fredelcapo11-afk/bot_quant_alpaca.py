# 🏦 Bot Quant Institucional - Alpaca & Nasdaq

Este es un bot de trading automático diseñado para operar en **Alpaca Markets**. Utiliza modelos estadísticos avanzados y Machine Learning para detectar breakouts en el Nasdaq y operar criptomonedas (BTC, ETH, SOL).

## 🚀 Características Principales
* **Escáner de Breakouts:** Filtra activos del Nasdaq con Volumen Relativo (RVOL) > 1.5.
* **Análisis GARCH:** Modelado de volatilidad para ajustar el Stop Loss dinámicamente.
* **Análisis ARIMA:** Predicción de tendencia a corto plazo.
* **Machine Learning:** Clasificador Random Forest para calcular la probabilidad de éxito de cada señal.
* **Ejecución Directa:** Gestión de órdenes tipo *Bracket* (TP/SL) vía Alpaca API.

## 🛠️ Configuración en Render
Para que este bot funcione 24/7, debes configurar las siguientes **Environment Variables** en el dashboard de Render:

| Variable | Descripción |
| :--- | :--- |
| `ALPACA_API_KEY` | Tu llave de Alpaca (Paper o Live) |
| `ALPACA_SECRET_KEY` | Tu secreto de Alpaca |
| `TELEGRAM_BOT_TOKEN` | Token de tu bot de Telegram |
| `TELEGRAM_CHAT_ID` | Tu ID de chat para alertas |

## 📦 Instalación Local
1. Clona el repositorio.
2. Instala dependencias: `pip install -r requirements.txt`.
3. Crea un archivo `.env` con tus credenciales.
4. Ejecuta: `python bot_quant_alpaca.py`.

## ⚠️ Descargo de Responsabilidad
Este bot es para fines educativos y de prueba. El trading implica riesgos significativos de pérdida de capital.
