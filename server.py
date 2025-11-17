import os
from datetime import datetime, timedelta
import traceback

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# -----------------------------
# 공통 에러 응답
# -----------------------------
def error_response(message, status_code=500, **extra):
    payload = {"error": True, "message": message}
    payload.update(extra)
    return jsonify(payload), status_code


# -----------------------------
# 유틸 함수들
# -----------------------------
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """단순 RSI 계산"""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    ma_up = up.rolling(window=period, min_periods=period).mean()
    ma_down = down.rolling(window=period, min_periods=period).mean()

    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def parse_capital(raw: str):
    """'1,000,000' 같은 입력을 int로 변환. 실패 시 None"""
    if not raw:
        return None

    try:
        cleaned = raw.replace(",", "").strip()
        if not cleaned:
            return None
        return int(float(cleaned))
    except Exception:
        return None


def make_position_plan(today_close: float, capital: int):
    """
    자본(capital)의 10%를 한 종목에 사용한다고 가정하고
    40/30/30 3회 분할 매수 전략 계산.
    """
    if capital is None or capital <= 0 or today_close <= 0:
        return {
            "capital_input": capital,
            "position_budget": 0,
            "shares_total": 0,
            "pos1_shares": 0,
            "pos2_shares": 0,
            "pos3_shares": 0,
            "pos1_amount": 0,
            "pos2_amount": 0,
            "pos3_amount": 0,
        }

    position_budget = int(capital * 0.10)  # 10%
    shares_total = position_budget // today_close

    if shares_total <= 0:
        return {
            "capital_input": capital,
            "position_budget": position_budget,
            "shares_total": 0,
            "pos1_shares": 0,
            "pos2_shares": 0,
            "pos3_shares": 0,
            "pos1_amount": 0,
            "pos2_amount": 0,
            "pos3_amount": 0,
        }

    pos1_shares = int(round(shares_total * 0.4))
    pos2_shares = int(round(shares_total * 0.3))
    pos3_shares = int(shares_total - pos1_shares - pos2_shares)

    pos1_amount = int(pos1_shares * today_close)
    pos2_amount = int(pos2_shares * today_close)
    pos3_amount = int(pos3_shares * today_close)

    return {
        "capital_input": capital,
        "position_budget": position_budget,
        "shares_total": int(shares_total),
        "pos1_shares": int(pos1_shares),
        "pos2_shares": int(pos2_shares),
        "pos3_shares": int(pos3_shares),
        "pos1_amount": int(pos1_amount),
        "pos2_amount": int(pos2_amount),
        "pos3_amount": int(pos3_amount),
    }


def build_signal(today_close, ma20, rsi, volume_today, volume_prev):
    """간단한 룰 베이스로 시그널 / 이유 문장 생성"""
    reasons = []
    signal = "관망 구간"

    if today_close > ma20 and 30 < rsi < 65:
        signal = "우상향 눌림목 매수 관심"
        reasons.append("종가가 20일선 위에 위치 (중기 추세 우상향)")
        reasons.append("RSI가 30~65 구간으로 과열/과매도 아님")
    elif rsi >= 70:
        signal = "단기 과열 – 분할 청산/관망"
        reasons.append("RSI가 70 이상으로 단기 과열 구간")
    elif today_close < ma20 and rsi < 40:
        signal = "하락 추세 – 무리한 진입 자제"
        reasons.append("종가가 20일선 아래에 위치 (중기 하락 추세 가능)")
        reasons.append("RSI가 40 미만으로 약세 구간")

    if volume_prev and volume_today > volume_prev * 1.5:
        reasons.append("금일 거래량이 전일 대비 1.5배 이상 증가 (수급 주목)")
    elif volume_prev and volume_today < volume_prev * 0.7:
        reasons.append("금일 거래량이 전일 대비 감소 (관망 심리 확대 가능)")

    if not reasons:
        reasons.append("특별히 강한 시그널은 없으며, 기본 관망/분할 대응 구간입니다.")

    return signal, reasons


def build_strategy_text(signal, today_close, stop_loss_price, tp1_price, tp2_price):
    """화면에 보여 줄 '오늘의 시나리오' 문장"""
    base = f"""[오늘의 기본 시나리오]

1) 진입 관점
- 현재가: 약 {round(today_close)}원 기준
- 시그널: {signal}

2) 손절/익절 기준 (예시)
- 손절: 약 {round(stop_loss_price)}원 (현재가 기준 -3% 부근)
- 1차 익절: 약 {round(tp1_price)}원 (현재가 기준 +5% 부근)
- 2차 익절: 약 {round(tp2_price)}원 (현재가 기준 +7% 부근)

3) 운용 팁
- 시초가 갭상/갭하가 큰 경우, 처음 5~10분 캔들 방향을 보고 분할 진입 비율을 조정합니다.
- 장 중 거래량이 전일 대비 급증하면, 익절 구간을 한 단계씩 위로 조정하는 것을 고려합니다.
- 반대로 거래량이 죽으면서 횡보하면, 손절 라인을 조금 더 타이트하게(예: -2%) 당기고 관망 비중을 높입니다.

※ 위 내용은 공부/연습용 예시 시나리오이며, 실제 매매 판단과 책임은 전적으로 본인에게 있습니다.
"""
    return base


# -----------------------------
# Flask 라우트
# -----------------------------
@app.route("/")
def health():
    return jsonify({"status": "ok", "message": "chang-scalper-api running"})


@app.route("/analyze")
def analyze():
    try:
        symbol_input = request.args.get("symbol", "").strip()
        capital_raw = request.args.get("capital", "").strip()

        if not symbol_input:
            return error_response("symbol 파라미터(종목 코드)가 필요합니다.", 400)

        # 한국 종목 코드이면 .KS 붙여서 yfinance 조회
        symbol_used = symbol_input
        if symbol_input.isdigit() and len(symbol_input) == 6:
            symbol_used = symbol_input + ".KS"

        # ---- 1) 데이터 다운로드 ----
        try:
            end = datetime.today()
            start = end - timedelta(days=160)
            df = yf.download(symbol_used, start=start, end=end)
        except Exception as e:
            return error_response(
                f"데이터 다운로드 실패: {str(e)}", 500, symbol_used=symbol_used
            )

        if df is None or df.empty or len(df) < 40:
            return error_response(
                "가격 데이터가 부족합니다. 심볼/종목 코드를 다시 확인해 주세요.",
                400,
                symbol_used=symbol_used,
            )

        # ---- 2) 지표 계산 ----
        df = df.rename(columns={"Close": "close", "Volume": "volume"})
        if "close" not in df.columns or "volume" not in df.columns:
            return error_response(
                "필수 컬럼(close/volume)이 없습니다. 다른 종목으로 시도해 보세요.",
                500,
                symbol_used=symbol_used,
                columns=list(df.columns),
            )

        df["ma5"] = df["close"].rolling(window=5).mean()
        df["ma20"] = df["close"].rolling(window=20).mean()
        df["rsi"] = calc_rsi(df["close"])

        df = df.dropna()
        if len(df) < 5:
            return error_response(
                "유효한 지표 계산을 위한 데이터가 부족합니다.",
                400,
                symbol_used=symbol_used,
            )

        last = df.iloc[-1]
        prev = df.iloc[-2]

        today_close = float(last["close"])
        ma5 = float(last["ma5"])
        ma20 = float(last["ma20"])
        rsi = float(last["rsi"])

        volume_today = int(last["volume"])
        volume_prev = int(prev["volume"]) if not pd.isna(prev["volume"]) else 0

        stop_loss_price = today_close * 0.97  # -3%
        tp1_price = today_close * 1.05        # +5%
        tp2_price = today_close * 1.07        # +7%

        # ---- 3) 시그널/전략 ----
        signal_kor, reasons = build_signal(
            today_close=today_close,
            ma20=ma20,
            rsi=rsi,
            volume_today=volume_today,
            volume_prev=volume_prev,
        )

        strategy_text = build_strategy_text(
            signal=signal_kor,
            today_close=today_close,
            stop_loss_price=stop_loss_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
        )

        # ---- 4) 자본/포지션 ----
        capital = parse_capital(capital_raw)
        position_info = make_position_plan(today_close=today_close, capital=capital)

        # ---- 5) 최종 payload (딕셔너리만 사용, DataFrame X) ----
        payload = {
            "error": False,
            "symbol_input": symbol_input,
            "symbol_used": symbol_used,
            "today_close": round(today_close),
            "ma5": round(ma5),
            "ma20": round(ma20),
            "rsi": round(rsi, 1) if rsi == rsi else None,
            "volume_today": int(volume_today),
            "volume_prev": int(volume_prev),
            "stop_loss_price": round(stop_loss_price),
            "tp1_price": round(tp1_price),
            "tp2_price": round(tp2_price),
            "signal_kor": signal_kor,
            "reasons": reasons,
            "strategy_text": strategy_text,
            **position_info,
        }

        return jsonify(payload)

    except Exception as e:
        # 예기치 못한 모든 에러도 JSON으로 반환
        tb = traceback.format_exc()
        print("UNEXPECTED ERROR in /analyze\n", tb, flush=True)
        symbol_used = locals().get("symbol_used", None)
        return error_response(
            f"서버 내부 오류: {e}", 500, symbol_used=symbol_used, traceback=tb
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
