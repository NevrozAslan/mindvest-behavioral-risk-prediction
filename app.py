# app.py
# ---------------------------------------------------------
# MindVest (TR) — Panik Satış Risk Kapısı (PoC)
# 3 katman: Risk Profili + Mental Durum + Davranış
# Amaç: Gençleri yatırıma yönlendirmek, ani kararları/panik satışı azaltmak,
# bankada AUM/bağlılık artışı sağlamak. (Yatırım tavsiyesi vermez.)
# ---------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score


# =============================
# Ürün metni (jüri dili)
# =============================

APP_NAME = "MindVest"
DATA_PATH = "mindvest_demo.csv"

DISCLAIMER = (
    "Bu demo bir **Proof of Concept (PoC)**’tir ve **yatırım tavsiyesi vermez**. "
    "Uygulama; **Risk Profili + Mental Durum + Davranışsal Senaryolar** üzerinden "
    "**panik satış riskini** tahmin eder ve eğitim amaçlı yönlendirmeler üretir."
)

VALUE_PROPOSITION = (
    "Hedefimiz özellikle **genç kullanıcıları** yatırıma daha sağlıklı şekilde yönlendirmek: "
    "ani kararları azaltmak, panik satışın önüne geçmek ve uzun vadeli yatırım disiplinini güçlendirmek.\n\n"
    "**Banka faydası:** müşteri bağlılığı ↑, panik satış ↓, yatırımda süreklilik ↑, AUM ↑."
)

TAGLINE = "Piyasayı değil, **insan davranışını** tahmin ediyoruz."


# =============================
# Senaryo seçenekleri (WOW)
# =============================

KAYIP_MAP = {
    "Beklerim / planı gözden geçiririm": 0,
    "Hemen satarım (zararı keserim)": 1,
    "Daha alırım (ortalamayı düşürürüm)": 2,
}

KAZANC_MAP = {
    "Kârın bir kısmını alırım": 0,
    "Aynen devam ederim": 1,
    "Daha fazla eklerim (FOMO)": 2,
}

SOSYAL_MAP = {
    "Resmi kaynak arar, doğrularım": 0,
    "Hemen satarım": 1,
    "Arkadaşlara/çevreye sorarım": 2,
}

SERINLEME_MAP = {
    "Evet, uygularım": 0,
    "Bazen": 1,
    "Hayır": 2,
}

RISK_TOL_MAP = {"Muhafazakâr": 2, "Dengeli": 3, "Agresif": 4}
ELDE_TUTMA_MAP = {"0–7 gün": 7, "8–30 gün": 21, "1–6 ay": 120, "6+ ay": 240}


# =============================
# Türkçe kolonlar
# =============================

REQUIRED_COLS = [
    "stres_puani",
    "kaygi_puani",
    "uyku_kalitesi",
    "risk_toleransi",
    "kayip_senaryosu_tepki",
    "kazanc_senaryosu_tepki",
    "sosyal_tetikleyici_tepki",
    "serinleme_kurali",
    "karar_hizi",
    "pismanlik_egilimi",
    "finansal_okuryazarlik",
    "onceki_kayip_deneyimi",
    "elde_tutma_gunu",
    "panik_satis",
]

FEATURE_COLS = [
    "stres_puani",
    "kaygi_puani",
    "uyku_kalitesi",
    "risk_toleransi",
    "kayip_senaryosu_tepki",
    "kazanc_senaryosu_tepki",
    "sosyal_tetikleyici_tepki",
    "serinleme_kurali",
    "karar_hizi",
    "pismanlik_egilimi",
    "finansal_okuryazarlik",
    "onceki_kayip_deneyimi",
    "elde_tutma_gunu",
]


# =============================
# Data + Model
# =============================

@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    # Excel (TR) çoğu zaman ';' ile kaydediyor. Önce ',' dene, olmazsa ';' dene.
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        # Eğer tek kolon geldiyse ve kolon adında ';' görüyorsak yanlış ayrıştırılmış demektir
        if df.shape[1] == 1 and ";" in df.columns[0]:
            df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV kolonları eksik: {missing}\nMevcut kolonlar: {list(df.columns)}")

    for c in REQUIRED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLS).copy()
    df["panik_satis"] = df["panik_satis"].astype(int)
    return df



@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame) -> Tuple[Pipeline, Optional[float], pd.Series]:
    X = df[FEATURE_COLS].copy()
    y = df["panik_satis"].astype(int)

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)

    auc = None
    if y_test.nunique() > 1:
        proba = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))

    coef = pd.Series(model.named_steps["clf"].coef_[0], index=FEATURE_COLS).abs().sort_values(ascending=False)
    return model, auc, coef


# =============================
# Skorlar: 3 katman (Jüri anlatımı)
# =============================

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def risk_profili_skoru(f: Dict[str, int]) -> float:
    # daha yüksek = daha agresif/riske açık profil
    # basit PoC skoru (0-1)
    rt = (f["risk_toleransi"] - 1) / 4          # 1..5 → normalize
    fk = (f["finansal_okuryazarlik"] - 1) / 4   # 1..5
    hp = clamp(f["elde_tutma_gunu"] / 240)      # uzun vade → daha planlı
    pl = 0.15 if f["onceki_kayip_deneyimi"] == 1 else 0.0  # kayıp tecrübesi bazılarını temkinli yapar (nötr + küçük)
    score = 0.35*rt + 0.35*fk + 0.25*hp + 0.05*pl
    return clamp(score)


def mental_durum_skoru(f: Dict[str, int]) -> float:
    # daha yüksek = mental yük daha yüksek (riskli an)
    stress = f["stres_puani"] / 10
    anxiety = f["kaygi_puani"] / 10
    sleep_bad = (10 - f["uyku_kalitesi"]) / 10
    score = 0.4*stress + 0.4*anxiety + 0.2*sleep_bad
    return clamp(score)


def davranis_skoru(f: Dict[str, int]) -> float:
    # daha yüksek = davranışsal tetikleyicilere daha açık (panic riski artar)
    loss_sell = 1.0 if f["kayip_senaryosu_tepki"] == 1 else 0.4 if f["kayip_senaryosu_tepki"] == 2 else 0.1
    gain_fomo = 1.0 if f["kazanc_senaryosu_tepki"] == 2 else 0.3 if f["kazanc_senaryosu_tepki"] == 1 else 0.2
    social_panic = 1.0 if f["sosyal_tetikleyici_tepki"] == 1 else 0.4 if f["sosyal_tetikleyici_tepki"] == 2 else 0.1
    cooldown_bad = 1.0 if f["serinleme_kurali"] == 2 else 0.5 if f["serinleme_kurali"] == 1 else 0.1
    speed = f["karar_hizi"] / 5
    regret = f["pismanlik_egilimi"] / 5
    score = 0.25*loss_sell + 0.15*gain_fomo + 0.2*social_panic + 0.15*cooldown_bad + 0.15*speed + 0.10*regret
    return clamp(score)


def seviye(x: float) -> str:
    if x >= 0.70:
        return "Yüksek"
    if x >= 0.40:
        return "Orta"
    return "Düşük"


# =============================
# Karar motoru (Gate + yönlendirme)
# =============================

@dataclass
class Karar:
    panik_olasiligi: float
    gate: str
    yatirimci_tipi: str
    ozet: str
    yonlendirme: List[str]


def yatirimci_tipi(f: Dict[str, int]) -> str:
    impulsive = (f["kayip_senaryosu_tepki"] == 1) and (f["karar_hizi"] >= 4) and (f["serinleme_kurali"] == 2)
    rational = (f["kayip_senaryosu_tepki"] == 0) and (f["sosyal_tetikleyici_tepki"] == 0) and (f["karar_hizi"] <= 3)
    if impulsive:
        return "Dürtüsel"
    if rational:
        return "Rasyonel"
    return "Duygusal"


def karar_ver(model: Pipeline, f: Dict[str, int]) -> Karar:
    x = pd.DataFrame([f])[FEATURE_COLS]
    p = float(model.predict_proba(x)[0, 1])

    rp = risk_profili_skoru(f)
    md = mental_durum_skoru(f)
    dv = davranis_skoru(f)

    # güçlü blok tetikleyici
    hard_block = (dv >= 0.75) and (md >= 0.60) and (p >= 0.55)

    if hard_block or p >= 0.70:
        gate = "BLOCK"
        ozet = "Panik satış riski yüksek: şu anda işlem yerine kısa bir soğuma ve bilgi desteği daha güvenli."
        yon = [
            "15 dakika serinleme (cooldown) uygula, sonra tekrar değerlendir.",
            "Tek işlem yerine kademeli yaklaşımı öğren (parça parça alım/satım mantığı).",
            "“Planım neydi?” sorusuna 1 cümle yaz: hedef süre, risk limiti, çıkış kriteri."
        ]
    elif p >= 0.40:
        gate = "CAUTION"
        ozet = "Orta risk: işlem yapacaksan küçük tutar + plan + limit ile ilerlemek daha iyi."
        yon = [
            "Küçük tutar + limit emir prensibi (ani karar azaltır).",
            "Sosyal medya haberlerini resmi kaynakla doğrulamadan işlem yapma.",
            "Kâr/zarar senaryoları için önceden kurallar belirle (disiplin)."
        ]
    else:
        gate = "ALLOW"
        ozet = "Düşük risk: planlı hareket edersen panik olasılığı düşük görünüyor."
        yon = [
            "Disiplin: hedef süre + risk limiti + çıkış kriteri belirle.",
            "Finansal okuryazarlığı artır: temel kavramlar (volatilite, çeşitlendirme).",
            "Aşırı özgüvenle risk artırma; düzenli gözden geçirme yap."
        ]

    # risk toleransı çok düşükse temkin
    if gate == "ALLOW" and f["risk_toleransi"] <= 2:
        gate = "CAUTION"
        ozet = "Risk toleransın düşük: temkin modu daha uygun."
        yon = [
            "Düşük volatilite + uzun vade prensiplerini öğren.",
            "Küçük tutar ile başla, çeşitlendirme mantığını uygula.",
            "Ani haber akışında ‘bekle-doğrula’ kuralı."
        ]

    return Karar(
        panik_olasiligi=p,
        gate=gate,
        yatirimci_tipi=yatirimci_tipi(f),
        ozet=ozet,
        yonlendirme=yon,
    )


# =============================
# UI yardımcıları
# =============================

def metric_row(items: List[Tuple[str, str]]):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


# =============================
# App
# =============================

st.set_page_config(page_title=f"{APP_NAME} — TR PoC", page_icon="🧠", layout="centered")

st.markdown(f"# 🧠 {APP_NAME}")
st.markdown(f"**{TAGLINE}**")
st.info(DISCLAIMER)
st.write(VALUE_PROPOSITION)

with st.sidebar:
    st.markdown("### Akış")
    step = st.radio("Adım seç", ["1) Mental Durum", "2) Davranış Senaryoları", "3) Risk Profili", "4) Sonuç"], index=0)

# Data + model
try:
    df = load_dataset(DATA_PATH)
except Exception as e:
    st.error("CSV okunamadı. Dosya adını/kolonları kontrol et.")
    st.code(str(e))
    st.stop()

model, auc, coef = train_model(df)
st.caption(f"PoC ROC-AUC (iç doğrulama): {auc:.2f}" if auc is not None else "PoC: ROC-AUC hesaplanamadı.")

# Session defaults
if "f" not in st.session_state:
    st.session_state.f = {
        "stres_puani": 5,
        "kaygi_puani": 5,
        "uyku_kalitesi": 7,
        "risk_toleransi": 3,
        "kayip_senaryosu_tepki": 0,
        "kazanc_senaryosu_tepki": 1,
        "sosyal_tetikleyici_tepki": 0,
        "serinleme_kurali": 0,
        "karar_hizi": 3,
        "pismanlik_egilimi": 3,
        "finansal_okuryazarlik": 3,
        "onceki_kayip_deneyimi": 0,
        "elde_tutma_gunu": 120,
    }


# 1) Mental Durum
if step == "1) Mental Durum":
    st.markdown("## 1) Mental Durum Ölçümü")
    st.caption("Uyku + stres + kaygı; ani karar ve panik satışı artırabilen kritik sinyallerdir.")

    sleep = st.radio("Son 2 gecenin uykusu nasıldı?", ["İyi (≈7+ saat)", "Orta (≈6–7)", "Zayıf (≈5–6)", "Çok kötü (≈<5)"], index=1)
    sleep_map = {"İyi (≈7+ saat)": 9, "Orta (≈6–7)": 7, "Zayıf (≈5–6)": 5, "Çok kötü (≈<5)": 3}
    st.session_state.f["uyku_kalitesi"] = sleep_map[sleep]

    stress = st.radio("Şu an stres düzeyin?", ["Düşük", "Orta", "Yüksek", "Çok yüksek"], index=1)
    st.session_state.f["stres_puani"] = {"Düşük": 3, "Orta": 5, "Yüksek": 7, "Çok yüksek": 9}[stress]

    anxiety = st.radio("Şu an kaygı/gerginlik düzeyin?", ["Düşük", "Orta", "Yüksek", "Çok yüksek"], index=1)
    st.session_state.f["kaygi_puani"] = {"Düşük": 3, "Orta": 5, "Yüksek": 7, "Çok yüksek": 9}[anxiety]

    md = mental_durum_skoru(st.session_state.f)
    metric_row([("Mental Durum Skoru", f"{md:.2f}"), ("Seviye", seviye(md))])
    st.success("Kaydedildi. Sol menüden **2) Davranış Senaryoları** adımına geç.")

# 2) Davranış
if step == "2) Davranış Senaryoları":
    st.markdown("## 2) Davranışsal Senaryolar")
    st.caption("Farklı tetikleyiciler: kayıp şoku, kazanç/FOMO, sosyal medya paniği, serinleme disiplini.")

    loss = st.radio("S1 — Kayıp şoku: 10.000 TL yatırımın 5 günde %12 düştü. Ne yaparsın?",
                    list(KAYIP_MAP.keys()), index=0)
    st.session_state.f["kayip_senaryosu_tepki"] = KAYIP_MAP[loss]

    gain = st.radio("S2 — Kazanç/FOMO: Yatırımın 2 haftada %18 yükseldi. Ne yaparsın?",
                    list(KAZANC_MAP.keys()), index=1)
    st.session_state.f["kazanc_senaryosu_tepki"] = KAZANC_MAP[gain]

    social = st.radio("S3 — Sosyal tetikleyici: Twitter’da “şirket batıyor” trend oldu. Resmi açıklama yok. Ne yaparsın?",
                      list(SOSYAL_MAP.keys()), index=0)
    st.session_state.f["sosyal_tetikleyici_tepki"] = SOSYAL_MAP[social]

    cooldown = st.radio("S4 — Serinleme kuralı: Büyük karar öncesi 15 dk bekleme kuralını uygular mısın?",
                        list(SERINLEME_MAP.keys()), index=0)
    st.session_state.f["serinleme_kurali"] = SERINLEME_MAP[cooldown]

    dv = davranis_skoru(st.session_state.f)
    metric_row([("Davranış Skoru", f"{dv:.2f}"), ("Seviye", seviye(dv))])
    st.success("Kaydedildi. Sol menüden **3) Risk Profili** adımına geç.")

# 3) Risk Profili
if step == "3) Risk Profili":
    st.markdown("## 3) Risk Profili Analizi")
    st.caption("Bu bölüm kullanıcının uzun vadeli yatırımcı karakterini (risk profili) çıkarır.")

    rt = st.radio("Genel risk toleransın?", list(RISK_TOL_MAP.keys()), index=1)
    st.session_state.f["risk_toleransi"] = RISK_TOL_MAP[rt]

    speed = st.radio("Karar hızın nasıl? (5=çok hızlı/ani)", ["1", "2", "3", "4", "5"], index=2)
    st.session_state.f["karar_hizi"] = int(speed)

    regret = st.radio("Pişmanlık eğilimin? (5=çok hızlı pişman olurum)", ["1", "2", "3", "4", "5"], index=2)
    st.session_state.f["pismanlik_egilimi"] = int(regret)

    fk = st.radio("Finansal okuryazarlık düzeyin? (5=yüksek)", ["1", "2", "3", "4", "5"], index=2)
    st.session_state.f["finansal_okuryazarlik"] = int(fk)

    prev = st.radio("Daha önce ciddi kayıp yaşadın mı?", ["Hayır", "Evet"], index=0)
    st.session_state.f["onceki_kayip_deneyimi"] = 1 if prev == "Evet" else 0

    hp = st.radio("Ortalama elde tutma süren?", list(ELDE_TUTMA_MAP.keys()), index=2)
    st.session_state.f["elde_tutma_gunu"] = ELDE_TUTMA_MAP[hp]

    rp = risk_profili_skoru(st.session_state.f)
    metric_row([("Risk Profili Skoru", f"{rp:.2f}"), ("Seviye", seviye(rp))])
    st.success("Kaydedildi. Sol menüden **4) Sonuç** adımına geç.")

# 4) Sonuç
if step == "4) Sonuç":
    st.markdown("## 4) Sonuç — Yönlendirme")
    f = st.session_state.f

    rp = risk_profili_skoru(f)
    md = mental_durum_skoru(f)
    dv = davranis_skoru(f)

    karar = karar_ver(model, f)

    metric_row([
        ("Panik Satış Olasılığı (P)", f"{karar.panik_olasiligi:.2f}"),
        ("Gate", karar.gate),
        ("Yatırımcı Tipi", karar.yatirimci_tipi),
    ])

    metric_row([
        ("Risk Profili", f"{rp:.2f} ({seviye(rp)})"),
        ("Mental Durum", f"{md:.2f} ({seviye(md)})"),
        ("Davranış", f"{dv:.2f} ({seviye(dv)})"),
    ])

    if karar.gate == "BLOCK":
        st.error("🔴 İşlem Engeli — yüksek panik riski")
    elif karar.gate == "CAUTION":
        st.warning("🟡 Temkin Modu — orta panik riski")
    else:
        st.success("🟢 Uygun — düşük panik riski")

    st.write(karar.ozet)

    st.markdown("### Eğitim/Yönlendirme (tavsiye değil)")
    for item in karar.yonlendirme:
        st.markdown(f"- {item}")

    with st.expander("Model detayları (opsiyonel)"):
        st.write("Global önem sırası (|coef|):")
        st.dataframe(coef.rename("importance(|coef|)"))
