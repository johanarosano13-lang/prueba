"""
Bot de alertas swing (Telegram). Solo avisa, no opera.
  Semanal: precio entrando en zona de Order Block grande
  Diario : divergencia grande del RSI
  Horario: cruce de medias (EMA rápida/lenta)
Hace una pasada por ejecución (GitHub Actions lo corre cada hora).
  python ema_alert_bot.py         -> revisa todo y avisa
  python ema_alert_bot.py --test  -> manda un mensaje de prueba a Telegram
"""
import os, sys, json
import numpy as np
import pandas as pd
import requests

# ---------------- ACTIVOS ----------------
WATCHLIST = [
    # ("crypto", símbolo en Kraken)  /  ("yf", ticker de Yahoo Finance)
    ("crypto", "BTC/USD"),
    ("crypto", "ETH/USD"),
    ("yf", "TSLA"),
    ("yf", "NVDA"),
    ("yf", "COIN"),
    ("yf", "MSTR"),
    ("yf", "EURUSD=X"),
    ("yf", "USDJPY=X"),
]

# ---------------- PARÁMETROS ----------------
# SEMANAL: order block
OB_LOOKBACK = 90         # velas semanales hacia atrás
OB_IMPULSE_ATR = 3.0     # "grande": el impulso posterior mueve >= 3 ATR
OB_IMPULSE_CANDLES = 5   # ...dentro de 5 velas
OB_ZONE = "body"         # "body" = cuerpo de la vela (open-close) / "range" = rango completo con mechas
OB_ONLY_FRESH = True     # solo OB que el precio NO volvió a tocar desde que se formó
# DIARIO: divergencia RSI
RSI_LEN = 14
PIVOT_N = 3              # un pivote se confirma con 3 velas a cada lado (3 días de demora)
DIV_LOOKBACK = 120
DIV_MIN_GAP = 10         # velas mínimas entre los dos pivotes
DIV_MIN_RSI_DIFF = 5     # diferencia mínima de RSI entre pivotes
DIV_MIN_PRICE_ATR = 0.5  # el precio debe superar el pivote anterior por >= 0.5 ATR
# HORARIO: cruce de medias
FAST, SLOW = 5, 10
GATE_1H = True           # True: avisa el cruce solo si coincide con OB semanal o divergencia diaria
GATE_DIV_HOURS = 72      # cuánto vale una divergencia diaria como contexto

STATE_FILE = "alert_state.json"
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
# --------------------------------------------

NAME = {"bull": "ALCISTA", "bear": "BAJISTA"}
DELTA = {"1h": pd.Timedelta(hours=1), "1d": pd.Timedelta(days=1), "1w": pd.Timedelta(days=7)}
YF_PERIOD = {"1h": "60d", "1d": "5y"}
SEND_FAILED = False
_kraken = None


def send(msg):
    global SEND_FAILED
    try:
        r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                          json={"chat_id": CHAT_ID, "text": msg}, timeout=15)
        if r.ok:
            return True
        print(f"ERROR Telegram {r.status_code}: {r.text}")
    except Exception as e:
        print("ERROR Telegram:", str(e).replace(TOKEN or "", "***"))
    SEND_FAILED = True
    return False


def fetch(kind, sym, tf):
    """Velas cerradas. tf: '1h' o '1d'."""
    global _kraken
    if kind == "crypto":
        if _kraken is None:
            import ccxt
            _kraken = ccxt.kraken()
        raw = _kraken.fetch_ohlcv(sym, tf, limit=720)
        df = pd.DataFrame(raw, columns=["t", "open", "high", "low", "close", "vol"])
        df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    else:
        import yfinance as yf
        df = yf.download(sym, period=YF_PERIOD[tf], interval=tf, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    return df[df.index + DELTA[tf] <= pd.Timestamp.now(tz="UTC")]


def to_weekly(d):
    """Velas semanales (lunes a domingo) a partir de las diarias."""
    w = d.resample("W-MON", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return w[w.index + DELTA["1w"] <= pd.Timestamp.now(tz="UTC")]


def atr(df, n=14):
    c = df["close"]
    tr = pd.concat([df.high - df.low, (df.high - c.shift()).abs(),
                    (df.low - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def rsi(c, n=RSI_LEN):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def pivots(x, n, kind):
    out = []
    for i in range(n, len(x) - n):
        w = x[i - n:i + n + 1]
        if (kind == "low" and x[i] == w.min()) or (kind == "high" and x[i] == w.max()):
            if not out or i - out[-1] > n:  # evita pivotes duplicados por empates
                out.append(i)
    return out


# ---------- SEMANAL: order blocks ----------
def find_obs(df):
    """OB alcista: última vela bajista antes de un impulso alcista grande (espejo para bajista).
    Se descarta si un cierre posterior lo atraviesa o (OB_ONLY_FRESH) si el precio ya volvió a tocarlo.
    Devuelve (dirección, zona_baja, zona_alta, semana, impulso_en_ATR)."""
    d = df.iloc[-OB_LOOKBACK:]
    a = atr(d).values
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    n = len(d)
    obs = []
    for i in range(14, n - 2):
        if np.isnan(a[i]):
            continue
        end = min(i + 1 + OB_IMPULSE_CANDLES, n)
        if c[i] < o[i] and c[i + 1] > o[i + 1]:
            dirn, move = "bull", h[i + 1:end].max() - c[i]
        elif c[i] > o[i] and c[i + 1] < o[i + 1]:
            dirn, move = "bear", c[i] - l[i + 1:end].min()
        else:
            continue
        if move < OB_IMPULSE_ATR * a[i]:
            continue
        lo, hi = (min(o[i], c[i]), max(o[i], c[i])) if OB_ZONE == "body" else (l[i], h[i])
        if dirn == "bull":
            if (c[i + 1:] < l[i]).any():
                continue
            if OB_ONLY_FRESH and (l[i + 2:] <= hi).any():
                continue
        else:
            if (c[i + 1:] > h[i]).any():
                continue
            if OB_ONLY_FRESH and (h[i + 2:] >= lo).any():
                continue
        obs.append((dirn, lo, hi, d.index[i], move / a[i]))
    return obs


def check_ob(sym, px, obs, st, ctx):
    sent = st["ob_sent"].setdefault(sym, [])
    ctx["ob"] = None
    status = f"fuera de zona ({len(obs)} OB vigentes)"
    for dirn, lo, hi, ts, mult in obs:
        if not (lo <= px <= hi):
            continue
        ctx["ob"] = dirn
        oid = f"{dirn}|{ts:%Y-%m-%d}"
        status = f"dentro de OB {NAME[dirn]}"
        if oid in sent:
            status += " (ya avisado)"
            continue
        zona = "cuerpo de la vela" if OB_ZONE == "body" else "rango completo"
        ok = send(f"{sym} · SEMANAL\nPrecio dentro de OB {NAME[dirn]} grande (sin tocar desde su formación)\n"
                  f"Zona ({zona}): {lo:.5g} – {hi:.5g}\nVela del OB: semana del {ts:%Y-%m-%d}\n"
                  f"Impulso posterior: {mult:.1f} ATR\nPrecio: {px:.5g}")
        if ok:
            sent.append(oid)
            status += " (alerta enviada)"
        else:
            status += " (envío FALLÓ)"
    return status


# ---------- DIARIO: divergencia RSI ----------
def find_divergence(df):
    r = rsi(df.close).iloc[-DIV_LOOKBACK:].values
    d = df.iloc[-DIV_LOOKBACK:]
    a = atr(df).iloc[-1]
    n = len(d)
    if n < 40 or np.isnan(a):
        return None
    recent = lambda i: i >= n - 1 - PIVOT_N - 2  # pivote recién confirmado
    lo, hi = d.low.values, d.high.values
    pl = pivots(lo, PIVOT_N, "low")
    if len(pl) >= 2:
        i1, i2 = pl[-2], pl[-1]
        if (recent(i2) and i2 - i1 >= DIV_MIN_GAP and lo[i1] - lo[i2] >= DIV_MIN_PRICE_ATR * a
                and r[i2] - r[i1] >= DIV_MIN_RSI_DIFF and r[i1] < 40):
            return "bull", d.index[i2], r[i1], r[i2], lo[i1], lo[i2]
    ph = pivots(hi, PIVOT_N, "high")
    if len(ph) >= 2:
        i1, i2 = ph[-2], ph[-1]
        if (recent(i2) and i2 - i1 >= DIV_MIN_GAP and hi[i2] - hi[i1] >= DIV_MIN_PRICE_ATR * a
                and r[i1] - r[i2] >= DIV_MIN_RSI_DIFF and r[i1] > 60):
            return "bear", d.index[i2], r[i1], r[i2], hi[i1], hi[i2]
    return None


def check_div(sym, d1, st, ctx):
    res = find_divergence(d1)
    if not res:
        return "sin divergencia"
    dirn, ts, r1, r2, p1, p2 = res
    ctx["div"] = {"dir": dirn, "seen": pd.Timestamp.now(tz="UTC").isoformat()}
    key = f"{dirn}|{ts}"
    if st["last"].get(f"{sym}|1d") == key:
        return f"{NAME[dirn]} (ya avisada)"
    ok = send(f"{sym} · DIARIO\nDivergencia {NAME[dirn]} grande en RSI\n"
              f"Precio: {p1:.5g} → {p2:.5g}\nRSI: {r1:.1f} → {r2:.1f}\nPivote: {ts:%Y-%m-%d}")
    if ok:
        st["last"][f"{sym}|1d"] = key
    return f"{NAME[dirn]} (alerta {'enviada' if ok else 'FALLÓ'})"


# ---------- HORARIO: cruce de medias ----------
def check_cross(sym, h1, st, ctx):
    c = h1.close
    gap = c.ewm(span=FAST, adjust=False).mean() - c.ewm(span=SLOW, adjust=False).mean()
    g0, g1 = gap.iloc[-2], gap.iloc[-1]
    if g0 <= 0 < g1:
        dirn = "bull"
    elif g0 >= 0 > g1:
        dirn = "bear"
    else:
        return "sin cruce"
    ts = h1.index[-1]
    key = f"{dirn}|{ts}"
    if st["last"].get(f"{sym}|1h") == key:
        return f"cruce {NAME[dirn]} (ya avisado)"
    why = []
    if ctx.get("ob") == dirn:
        why.append("precio en OB semanal " + NAME[dirn].lower())
    div = ctx.get("div")
    if div and div["dir"] == dirn and \
            pd.Timestamp.now(tz="UTC") - pd.Timestamp(div["seen"]) <= pd.Timedelta(hours=GATE_DIV_HOURS):
        why.append("divergencia RSI diaria " + NAME[dirn].lower())
    if GATE_1H and not why:
        st["last"][f"{sym}|1h"] = key
        return f"cruce {NAME[dirn]} sin confluencia (no se avisa)"
    ok = send(f"{sym} · HORARIO\nCruce {NAME[dirn]} EMA{FAST}/EMA{SLOW}\nPrecio: {c.iloc[-1]:.5g}\n"
              f"Contexto: {', '.join(why) if why else 'sin confluencia'}\nVela: {ts:%Y-%m-%d %H:%M} UTC")
    if ok:
        st["last"][f"{sym}|1h"] = key
    return f"cruce {NAME[dirn]} (alerta {'enviada' if ok else 'FALLÓ'})"


def run_symbol(kind, sym, st):
    ctx = st["ctx"].setdefault(sym, {})
    h1 = fetch(kind, sym, "1h")
    d1 = fetch(kind, sym, "1d")
    px = float(h1.close.iloc[-1])
    obs = find_obs(to_weekly(d1))
    for dirn, lo, hi, ts, mult in obs:
        print(f"   {sym} OB {NAME[dirn]} semana {ts:%Y-%m-%d} zona {lo:.5g}–{hi:.5g} ({mult:.1f} ATR)")
    s_ob = check_ob(sym, px, obs, st, ctx)  # fija el contexto semanal
    s_div = check_div(sym, d1, st, ctx)                         # fija el contexto diario
    s_x = check_cross(sym, h1, st, ctx)
    print(f"{sym}: precio {px:.5g} | semanal OB: {s_ob} | diario RSI: {s_div} | horario cruce: {s_x}")


def load_state():
    try:
        st = json.load(open(STATE_FILE))
    except Exception:
        st = {}
    for k in ("last", "ob_sent", "ctx"):
        if not isinstance(st.get(k), dict):
            st[k] = {}
    return st


def main():
    if not TOKEN or not CHAT_ID:
        print("ERROR: faltan los secretos TELEGRAM_TOKEN y/o TELEGRAM_CHAT_ID "
              "(Settings > Secrets and variables > Actions)")
        sys.exit(1)
    if "--test" in sys.argv:
        ok = send("✅ Bot de alertas funcionando. Si ves este mensaje, Telegram está bien configurado.")
        print("Mensaje de prueba enviado." if ok else "FALLÓ el mensaje de prueba.")
        sys.exit(0 if ok else 1)
    st = load_state()
    for kind, sym in WATCHLIST:
        try:
            run_symbol(kind, sym, st)
        except Exception as e:
            print(f"Error con {sym}: {e}")
    json.dump(st, open(STATE_FILE, "w"))
    sys.exit(1 if SEND_FAILED else 0)


if __name__ == "__main__":
    main()
