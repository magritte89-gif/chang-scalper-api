from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # 우리 HTML 앱에서 이 서버를 부를 수 있게 허용


def build_symbol(raw: str) -> str:
    """
    사용자가 입력한 심볼을 보고
    - 숫자만 있으면 KRX(.KS)
    - 이미 .KS/.KQ 있으면 그대로
    - 나머지는 해외 종목 그대로 사용
    """
    t = raw.strip().upper()
    if not t:
        return ""

    if t.endswith(".KS") or t.endswith(".KQ"):
        return t

    if t.isdigit():
        return t + ".KS"

    return t  # 예: MSFT, AAPL 등


def calc_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


@app.route("/analyze")
def analyze():
    raw_symbol = request.args.get("symbol", "").strip()
    if not raw_symbol:
        return jsonify({"error": "no_symbol", "message": "심볼을 입력해 주세요."}), 400

    symbol = build_symbol(raw_symbol)
    if not symbol:
        return jsonify({"error": "invalid_symbol", "message": "심볼 형식이 잘못되었습니다."}), 400

    try:
        # 최근 3개월 일봉 데이터
        data = yf.download(symbol, period="3mo", interval="1d", auto_adjust=False, progress=False)
    except Exception as e:
        return jsonify({"error": "download_failed", "message": str(e)}), 500

    if data is None or data.empty:
        return jsonify({"error": "empty_data", "message": f"{symbol} 데이터가 없습니다."}), 404

    data = data.dropna()
    if len(data) < 20:
        return jsonify({"error": "insufficient_data", "message": "20일선 계산에 필요한 데이터가 부족합니다."}), 400

    closes = data["Close"]
    volumes = data["Volume"]

    today_close = float(closes.iloc[-1])
    ma5 = float(closes.iloc[-5:].mean())
    ma20 = float(closes.iloc[-20:].mean())
    vol_today = float(volumes.iloc[-1])
    vol_prev = float(volumes.iloc[-2])

    # RSI (14)
    rsi = calc_rsi(closes)

    # 단타 A 세트 기준 스코어
    score = 0
    reasons = []

    # 1) 20일선 위
    if today_close > ma20:
        score += 1
        reasons.append("20일선 위 (안전)")
    else:
        reasons.append("20일선 아래 → 위험")

    # 2) 5일선 > 20일선
    if ma5 > ma20:
        score += 1
        reasons.append("5일선이 20일선 상향 돌파")

    # 3) 거래량 증가 (전일 대비 +50%)
    if vol_today > vol_prev * 1.5:
        score += 1
        reasons.append("거래량 증가 (전일 대비 +50% 이상)")
    else:
        reasons.append("거래량 평범 또는 감소")

    # 4) RSI 건강 구간
    if 45 <= rsi <= 60:
        score += 1
        reasons.append("RSI 건강 구간 (45~60)")
    elif rsi > 70:
        reasons.append("RSI 과열 (70 이상)")
    elif rsi < 30:
        reasons.append("RSI 과매도 (30 이하)")

    # 시그널 텍스트
    if score >= 3:
        signal = "BUY_STRONG"
        signal_kor = "🟢 매수 유력"
    elif score == 2:
        signal = "WATCH"
        signal_kor = "🟡 관망"
    else:
        signal = "AVOID"
        signal_kor = "🔴 매수주의"

    result = {
        "symbol_input": raw_symbol,
        "symbol_used": symbol,
        "today_close": today_close,
        "ma5": ma5,
        "ma20": ma20,
        "volume_today": vol_today,
        "volume_prev": vol_prev,
        "rsi": rsi,
        "score": score,
        "signal": signal,
        "signal_kor": signal_kor,
        "reasons": reasons,
    }

    return jsonify(result)


@app.route("/")
def health():
    return "Chang scalper API is running."


if __name__ == "__main__":
    # 로컬 테스트용 (Render에서는 무시됨)
    app.run(host="0.0.0.0", port=5000, debug=True)
