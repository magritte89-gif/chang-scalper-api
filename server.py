from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import re

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


def parse_capital(raw: str):
    """
    사용자가 입력한 자본 문자열을 숫자로 변환
    예: "10,000,000" -> 10000000
    """
    if not raw:
        return None
    # 숫자와 점만 남기고 제거
    cleaned = re.sub(r"[^0-9.]", "", raw)
    if not cleaned:
        return None
    try:
        value = float(cleaned)
        if value <= 0:
            return None
        return value
    except ValueError:
        return None


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
    raw_capital = request.args.get("capital", "").strip()

    if not raw_symbol:
        return jsonify({"error": "no_symbol", "message": "심볼을 입력해 주세요."}), 400

    symbol = build_symbol(raw_symbol)
    if not symbol:
        return jsonify({"error": "invalid_symbol", "message": "심볼 형식이 잘못되었습니다."}), 400

    capital_value = parse_capital(raw_capital)
    # None 이면 "자본 입력 안 함"으로 처리

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

    # --- 손절 / 익절 가격 계산 ---
    stop_loss_price = round(today_close * 0.97)   # -3%
    tp1_price = round(today_close * 1.05)         # +5%
    tp2_price = round(today_close * 1.07)         # +7%

    # --- 자본 기반 포지션 사이즈 계산 (표준형: 전체 자본의 10% 사용, 40/30/30 분할) ---
    position_budget = None
    shares_total = None
    pos1_amount = pos2_amount = pos3_amount = None
    pos1_shares = pos2_shares = pos3_shares = None

    if capital_value is not None and today_close > 0:
        # 오늘 이 종목에 쓸 최대 금액: 자본의 10%
        risk_fraction = 0.10
        position_budget = capital_value * risk_fraction

        # 총 매수 가능 주수
        shares_total = int(position_budget // today_close)

        # 분할 비중 (표준형 B: 40 / 30 / 30)
        pos1_shares = int(shares_total * 0.4)
        pos2_shares = int(shares_total * 0.3)
        pos3_shares = shares_total - pos1_shares - pos2_shares  # 나머지

        pos1_amount = pos1_shares * today_close
        pos2_amount = pos2_shares * today_close
        pos3_amount = pos3_shares * today_close

    # --- Step-by-Step 전략 텍스트 생성 ---
    strategy_lines = []

    # STEP 1. 오늘 이 종목을 볼 가치가 있는지
    strategy_lines.append("STEP 1. 오늘 이 종목을 볼 가치가 있을까?")
    if score >= 3:
        strategy_lines.append(" → 단타 A-세트 기준으로 '오늘 진입 후보'에 해당합니다.")
    elif score == 2:
        strategy_lines.append(" → 패턴은 나쁘지 않지만 애매한 구간입니다. '관망 또는 소액 진입'이 적합합니다.")
    else:
        strategy_lines.append(" → 추세/거래량/RSI 조건이 충분히 맞지 않아 오늘은 관망이 더 안전합니다.")

    # STEP 2. 오늘의 추세 요약
    strategy_lines.append("")
    strategy_lines.append("STEP 2. 오늘의 추세 요약")
    trend_desc = []
    if today_close > ma20:
        trend_desc.append("· 가격이 20일선 위에 있어 중기 추세는 양호합니다.")
    else:
        trend_desc.append("· 가격이 20일선 아래에 있어 중기 추세가 약한 편입니다.")

    if ma5 > ma20:
        trend_desc.append("· 5일선이 20일선 위에 있어 단기 추세도 우상향입니다.")
    else:
        trend_desc.append("· 5일선이 20일선 아래에 있어 단기 추세는 아직 약합니다.")

    if vol_today > vol_prev * 1.5:
        trend_desc.append("· 거래량이 전일 대비 크게 증가해 수급이 유입되는 모습입니다.")
    else:
        trend_desc.append("· 거래량이 전일 대비 크지 않아 강한 수급은 아닙니다.")

    if 45 <= rsi <= 60:
        trend_desc.append("· RSI는 45~60 구간으로, 과열도 과매도도 아닌 '건강한 구간'입니다.")
    elif rsi > 70:
        trend_desc.append("· RSI가 과열(70 이상)에 가까워 단기 급등 후 조정 가능성을 염두에 둬야 합니다.")
    elif rsi < 30:
        trend_desc.append("· RSI가 과매도(30 이하)에 가까워 단기 반등 가능성은 있으나 추세 확인이 필요합니다.")

    strategy_lines.extend(trend_desc)

    # STEP 3. 오늘의 추천 행동
    strategy_lines.append("")
    strategy_lines.append("STEP 3. 오늘의 추천 행동")
    if score >= 3:
        strategy_lines.append(" → '진입 가능' 구간입니다. 다만 반드시 분할 매수와 손절 기준을 함께 사용해야 합니다.")
    elif score == 2:
        strategy_lines.append(" → '부분 진입 또는 관망'이 적절합니다. 무리한 비중 확대는 피하는 편이 안전합니다.")
    else:
        strategy_lines.append(" → 오늘은 신규 매수보다는 관망을 추천합니다.")

    # STEP 4~5. 매수 타점 & 투자 금액
    strategy_lines.append("")
    strategy_lines.append("STEP 4. 매수 타점 (예시)")
    strategy_lines.append(" · 1차 매수: 현재가 ~ 5일선 근처 가격대에서 분할 진입을 고려합니다.")
    strategy_lines.append(" · 2차 매수: 1차 매수 후 단기 눌림(-1% 내외)이 나올 경우 추가 진입을 검토합니다.")
    strategy_lines.append(" · 3차 매수: 추세가 유지되는 선에서 추가 상승 또는 재조정 시 확인 후 진입합니다.")

    strategy_lines.append("")
    strategy_lines.append("STEP 5. 오늘 이 종목에 쓸 수 있는 최대 금액 (예시 기준)")
    if capital_value is not None and position_budget is not None and shares_total is not None and shares_total > 0:
        strategy_lines.append(f" · 입력 자본: 약 {capital_value:,.0f}원")
        strategy_lines.append(f" · 이 종목에 사용할 최대 금액 (자본의 10% 가정): 약 {position_budget:,.0f}원")
        strategy_lines.append(f" · 현재가 기준 총 매수 가능 수량: 약 {shares_total:,}주")
        strategy_lines.append(" · 표준형 분할 매수 (40% / 30% / 30%) 기준:")
        strategy_lines.append(f"    - 1차: {pos1_shares:,}주 (약 {pos1_amount:,.0f}원)")
        strategy_lines.append(f"    - 2차: {pos2_shares:,}주 (약 {pos2_amount:,.0f}원)")
        strategy_lines.append(f"    - 3차: {pos3_shares:,}주 (약 {pos3_amount:,.0f}원)")
    else:
        strategy_lines.append(" · 자본 정보를 입력하지 않아 구체적인 금액/수량 계산은 생략되었습니다.")
        strategy_lines.append(" · 원한다면 화면의 '투자 가능한 총 자본(원)' 입력란에 자본을 입력하고 다시 조회해 주세요.")

    # STEP 6~8. 손절/익절 및 보유 중 관리
    strategy_lines.append("")
    strategy_lines.append("STEP 6. 손절 기준 (예시)")
    strategy_lines.append(f" · 손절가: 현재가 대비 약 -3% 구간 (대략 {stop_loss_price:,.0f}원 부근)")
    strategy_lines.append(" · 손절가는 매수 전에 미리 정해 두고, 도달 시 추가 고민 없이 정리하는 것이 좋습니다.")

    strategy_lines.append("")
    strategy_lines.append("STEP 7. 익절 기준 (예시)")
    strategy_lines.append(f" · 1차 익절: 현재가 대비 +5% (대략 {tp1_price:,.0f}원 부근)")
    strategy_lines.append(f" · 2차 익절: 현재가 대비 +7% (대략 {tp2_price:,.0f}원 부근)")
    strategy_lines.append(" · 수익이 났을 때 일부라도 확정해 두는 습관이 심리적으로 안정에 도움이 됩니다.")

    strategy_lines.append("")
    strategy_lines.append("STEP 8. 보유 중 체크 포인트")
    strategy_lines.append(" · RSI가 70 이상으로 과열 구간에 진입하면, 일부 익절 또는 비중 축소를 고려합니다.")
    strategy_lines.append(" · 5일선을 이탈하고 거래량이 증가하며 하락하는 경우, 방어적인 대응이 필요합니다.")
    strategy_lines.append(" · 20일선까지 깨지는 경우 중기 추세가 훼손될 수 있으므로, 대부분 정리를 검토합니다.")

    # STEP 9~10. 청산 & 복기
    strategy_lines.append("")
    strategy_lines.append("STEP 9. 청산 시나리오")
    strategy_lines.append(" · 목표 수익(예: +5~7%) 구간에 도달했다면, 욕심을 과도하게 내지 말고 계획대로 청산합니다.")
    strategy_lines.append(" · 손절 구간에 도달했다면, '다시 오를 것'이라는 희망보다 원래 세운 원칙을 우선합니다.")

    strategy_lines.append("")
    strategy_lines.append("STEP 10. 복기")
    strategy_lines.append(" · 매매가 끝난 후, 진입/청산 위치와 오늘의 전략을 다시 비교해 보면서 한 줄 정도의 복기를 남겨 보세요.")
    strategy_lines.append(" · 시스템은 의사결정을 돕는 도구일 뿐, 최종 판단과 책임은 항상 본인에게 있음을 기억하는 것이 중요합니다.")

    strategy_text = "\n".join(strategy_lines)

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
        "strategy_text": strategy_text,
        "stop_loss_price": stop_loss_price,
        "tp1_price": tp1_price,
        "tp2_price": tp2_price,
        "capital_input": capital_value,
        "position_budget": position_budget,
        "shares_total": shares_total,
        "pos1_shares": pos1_shares,
        "pos2_shares": pos2_shares,
        "pos3_shares": pos3_shares,
        "pos1_amount": pos1_amount,
        "pos2_amount": pos2_amount,
        "pos3_amount": pos3_amount,
    }

    return jsonify(result)


@app.route("/")
def health():
    return "Chang scalper API is running."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
