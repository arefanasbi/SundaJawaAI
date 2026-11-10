"""
app.py — SundaJawaAI
Aplikasi identifikasi bahasa daerah: Sunda Kasar, Sunda Lemes, Jawa, Aceh, Palembang.
Input: teks langsung | audio (Whisper) | gambar (Tesseract OCR).
"""
import logging
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.classifier import load_model, predict_language
from utils.audio import inspect_audio_safe, load_whisper_model, transcribe_audio
from utils.ocr import extract_text_from_image, is_tesseract_available

# ---------------------------------------------------------------------------
# Setup logging ke terminal (bukan ke UI)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
PAGES = [
    "Beranda",
    "Analisis Teks",
    "Pengolahan Audio",
    "Analisis Foto",
    "Pelatihan Model",
    "Evaluasi Model",
    "Tentang Aplikasi",
]

# ---------------------------------------------------------------------------
# Konfigurasi halaman
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sunda-Jawa AI",
    page_icon="🌺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — pertahankan desain pink/magenta
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root { --plum:#641044; --berry:#a51c68; --pink:#e94b9a; --lav:#9b70ec; --ink:#28123e; }
html, body, [class*="css"] { font-family:'Plus Jakarta Sans',sans-serif; }
.stApp { background:linear-gradient(135deg,#fff7fc 0%,#fceafa 48%,#f8f1ff 100%); }
.block-container { max-width:1500px; padding-top:1.4rem; padding-bottom:3rem; }
section[data-testid="stSidebar"] {
 background:linear-gradient(180deg,#641044 0%,#8e185d 55%,#53103f 100%);
 border-right:1px solid rgba(255,255,255,.16);
}
section[data-testid="stSidebar"] * { color:#fff8fd !important; }
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
 padding:.62rem .75rem; border-radius:13px; font-weight:600;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:rgba(255,255,255,.12); }
section[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap:.22rem; }
.brand-mark { text-align:center; padding:.8rem .2rem 1.1rem; }
.brand-mark .symbol { font-size:2.7rem; filter:drop-shadow(0 5px 8px #3b092b); }
.brand-mark h2 { color:white; font-size:1.3rem; margin:.15rem 0; font-weight:800; }
.brand-mark p { color:#f6d8e9; font-size:.78rem; margin:0; }
.hero {
 position:relative; overflow:hidden; padding:2.15rem 2.25rem; border-radius:26px;
 background:linear-gradient(110deg,#fff0f8 0%,#f9d7ec 48%,#e8d7ff 100%);
 border:1px solid rgba(255,255,255,.9);
 box-shadow:0 18px 42px rgba(126,37,99,.12); margin-bottom:1.35rem;
}
.hero:after { content:"✿"; position:absolute; right:4%; top:-35%; font-size:18rem; color:rgba(194,54,137,.09); transform:rotate(-15deg); pointer-events:none; }
.hero .eyebrow { color:#a32a73; letter-spacing:.15em; font-size:.72rem; font-weight:800; text-transform:uppercase; }
.hero h1 { color:#541044; font-size:clamp(2.15rem,4vw,3.6rem); font-weight:800; letter-spacing:-.045em; margin:.35rem 0 .25rem; line-height:1.1; }
.hero p { color:#44234f; font-size:1rem; max-width:720px; margin:.4rem 0 1rem; }
.hero .sub { color:#765b7d; font-size:.84rem; }
.pill { display:inline-block; border-radius:30px; padding:.35rem .75rem; background:#fff; color:#9b216a; font-size:.75rem; font-weight:700; box-shadow:0 3px 12px #a43b781c; }
.section-heading { color:#4e1748; font-size:1.22rem; font-weight:800; margin:.35rem 0 .85rem; }
.feature-card {
 background:rgba(255,255,255,.78); border:1px solid #f0cfe4; border-radius:20px;
 padding:1.05rem 1.1rem; min-height:116px; box-shadow:0 8px 22px rgba(128,34,98,.07);
}
.feature-icon { display:inline-flex; width:44px; height:44px; align-items:center; justify-content:center; border-radius:14px; background:linear-gradient(135deg,#f8d1e8,#eadbff); font-size:1.35rem; margin-bottom:.55rem; }
.feature-card h3 { color:#411342; font-size:1rem; margin:0 0 .2rem; font-weight:800; }
.feature-card p { color:#806985; font-size:.8rem; margin:0; }
.panel {
 background:rgba(255,255,255,.86); border:1px solid #f0d7e9; border-radius:21px;
 padding:1.15rem 1.25rem; box-shadow:0 10px 28px rgba(128,34,98,.065);
}
.result-card {
 background:linear-gradient(145deg,#ffe0f1,#f4e6ff); border:1px solid #f2c9e5;
 border-radius:18px; padding:1.1rem; text-align:center;
}
.result-card h3 { color:#65124d; margin:.2rem 0; font-size:1.15rem; font-weight:800; }
.result-score { color:#721452; font-size:2rem; font-weight:800; }
.muted { color:#806985; font-size:.84rem; }
div.stButton > button {
 border:0; border-radius:13px; color:white; font-weight:800;
 background:linear-gradient(100deg,#b51e70,#e34c9d); padding:.58rem 1rem;
 box-shadow:0 5px 14px rgba(183,31,112,.18);
}
div.stButton > button:hover { color:white; border:0; filter:brightness(1.06); }
div[data-testid="stTextArea"] textarea { border-radius:15px; border:1px solid #e9cde1; background:#fff; }
div[data-testid="stFileUploader"] { background:#fffafd; border:1px dashed #dba8ca; border-radius:15px; padding:.5rem; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #f1dbea; padding:13px; border-radius:15px; }
.stProgress > div > div { background:linear-gradient(90deg,#c12c83,#a477ec); }
hr { border-color:#efd8e9; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cache: model klasifikasi
# ---------------------------------------------------------------------------
@st.cache_resource
def get_classifier():
    return load_model()


# ---------------------------------------------------------------------------
# Cache: model Whisper (tidak reload setiap kali tombol ditekan)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_whisper_model(model_size: str):
    return load_whisper_model(model_size)


# ---------------------------------------------------------------------------
# Navigasi
# ---------------------------------------------------------------------------
def go_to(page: str):
    st.session_state["nav_page"] = page


if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Beranda"


# ---------------------------------------------------------------------------
# Panel hasil prediksi (dipakai oleh teks, audio, foto)
# ---------------------------------------------------------------------------
def prediction_panel(text: str):
    """
    Jalankan klasifikasi dan tampilkan hasil.
    Dipakai bersama oleh semua alur input.
    """
    if not text or not text.strip():
        st.warning("Tidak ada teks untuk dianalisis.")
        return

    try:
        model = get_classifier()
        label, confidence, probs = predict_language(model, text)
    except FileNotFoundError:
        st.error("Model belum tersedia. Jalankan `python train_model.py` dari folder SundaJawaAI.")
        return
    except Exception as exc:
        logger.exception("Prediksi gagal")
        st.error("Prediksi gagal. Lihat terminal untuk detail teknis.")
        return

    st.markdown('<div class="section-heading">Hasil identifikasi bahasa</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.1], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Status identifikasi**")
        if label == "Tidak yakin":
            st.warning(f"Model belum cukup yakin · skor tertinggi {confidence:.1%}")
        else:
            st.success(f"Bahasa terdeteksi: **{label}**")
        st.markdown(
            f'<div class="result-card">'
            f'<div class="muted">Prediksi model</div>'
            f'<h3>{label}</h3>'
            f'<div class="result-score">{confidence:.1%}</div>'
            f'<span class="pill">Skor keyakinan tertinggi</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Persentase adalah probabilitas relatif yang diperkirakan model untuk setiap kategori, "
            "bukan ukuran pasti komposisi bahasa. Akurasi bergantung pada kualitas dan keragaman dataset."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Perbandingan skor per kategori bahasa**")
        for language, score in sorted(probs.items(), key=lambda x: -x[1]):
            st.write(f"**{language}** · {score:.1%}")
            st.progress(float(score))
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand-mark">
      <div class="symbol">🌺</div>
      <h2>Sunda-Jawa AI</h2>
      <p>REGIONAL LANGUAGE DETECTION</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("**NAVIGASI UTAMA**")
    page = st.radio("Pilih halaman", PAGES, key="nav_page", label_visibility="collapsed")
    st.divider()
    st.caption("NATURAL LANGUAGE PROCESSING")
    st.caption("Sunda · Jawa · Aceh · Palembang · Teks · Audio · Gambar")


# ===========================================================================
# HALAMAN: BERANDA
# ===========================================================================
if page == "Beranda":
    st.markdown("""
    <div class="hero">
      <span class="pill">● MODEL AI · PROTOTIPE</span>
      <h1>Sunda-Jawa <span style="color:#d12d87">AI</span></h1>
      <p><b>Sistem Identifikasi Bahasa Daerah Berbasis Artificial Intelligence</b></p>
      <p>Deteksi Sunda Kasar, Sunda Lemes, Jawa, Aceh, dan Palembang melalui teks, suara, maupun gambar.</p>
      <div class="sub">Satu aplikasi untuk membantu mengenali lima kategori bahasa daerah.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Jelajahi fitur</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    cards = [
        (c1, "⌨️", "Analisis Teks",   "Ketik kalimat untuk mendeteksi bahasa.",        "Analisis Teks"),
        (c2, "🎙️", "Pengolahan Audio","Berbicara langsung atau unggah audio.",          "Pengolahan Audio"),
        (c3, "🖼️", "Analisis Gambar", "Unggah foto yang berisi tulisan.",               "Analisis Foto"),
    ]
    for col, icon, title, desc, target in cards:
        with col:
            st.markdown(
                f'<div class="feature-card"><div class="feature-icon">{icon}</div>'
                f'<h3>{title}</h3><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )
            st.button(
                f"Buka {title}  →",
                key=f"open_{target}",
                use_container_width=True,
                on_click=go_to,
                args=(target,),
            )

    st.write("")
    m1, m2, m3 = st.columns(3)
    for col, value, label, desc in [
        (m1, "5 Kategori",   "BAHASA DIDUKUNG",    "Sunda Kasar · Sunda Lemes · Jawa · Aceh · Palembang"),
        (m2, "3 Metode",     "JENIS INPUT",         "Teks · Audio · Gambar"),
        (m3, "TF-IDF + LR",  "MESIN KLASIFIKASI",  "Machine Learning"),
    ]:
        with col:
            st.markdown(
                f'<div class="panel"><div class="muted">{label}</div>'
                f'<div style="font-size:1.5rem;font-weight:800;color:#641044">{value}</div>'
                f'<div class="muted">{desc}</div></div>',
                unsafe_allow_html=True,
            )
    st.caption("Catatan: dataset saat ini merupakan dataset awal demonstrasi dan perlu diperluas serta divalidasi untuk evaluasi akademik.")


# ===========================================================================
# HALAMAN: ANALISIS TEKS
# ===========================================================================
elif page == "Analisis Teks":
    st.markdown(
        '<div class="hero"><span class="pill">TEXT CLASSIFICATION</span>'
        '<h1>⌨️ Analisis Teks</h1>'
        '<p>Ketik atau tempel kalimat untuk mengidentifikasi bahasa daerah.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    text = st.text_area(
        "Masukkan kalimat",
        height=170,
        max_chars=500,
        placeholder=(
            "Contoh Sunda Lemes: Abdi badé angkat.\n"
            "Contoh Sunda Kasar: Kuring rék indit.\n"
            "Contoh Aceh: Peu haba droeneuh?\n"
            "Contoh Palembang: Apo kabar kau?\n"
            "Contoh Jawa: Aku arep lunga."
        ),
    )
    st.caption(f"{len(text)} / 500 karakter")

    if st.button("✨ Analisis sekarang", type="primary"):
        if not text.strip():
            st.warning("Masukkan teks terlebih dahulu.")
        else:
            with st.status("Menganalisis teks...", expanded=True) as status:
                st.write("⏳ Memproses teks...")
                time.sleep(0.3)
                st.write("🔍 Menganalisis karakteristik bahasa...")
                time.sleep(0.3)
                st.write("📊 Menghitung probabilitas setiap kategori...")
                time.sleep(0.2)
                try:
                    model = get_classifier()
                    label, confidence, probs = predict_language(model, text)
                    st.write("✅ Analisis selesai.")
                    status.update(label="Analisis selesai!", state="complete", expanded=False)
                except FileNotFoundError:
                    status.update(label="Gagal", state="error", expanded=False)
                    st.error("Model belum tersedia. Jalankan `python train_model.py` dari folder SundaJawaAI.")
                    st.stop()
                except Exception as exc:
                    logger.exception("Klasifikasi teks gagal")
                    status.update(label="Gagal", state="error", expanded=False)
                    st.error("Analisis gagal. Lihat terminal untuk detail teknis.")
                    st.stop()

            # Tampilkan hasil
            st.markdown('<div class="section-heading">Hasil identifikasi bahasa</div>', unsafe_allow_html=True)
            left, right = st.columns([1, 1.1], gap="large")
            with left:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown("**Status identifikasi**")
                if label == "Tidak yakin":
                    st.warning(f"Model belum cukup yakin · skor tertinggi {confidence:.1%}")
                else:
                    st.success(f"Bahasa terdeteksi: **{label}**")
                st.markdown(
                    f'<div class="result-card">'
                    f'<div class="muted">Prediksi model</div>'
                    f'<h3>{label}</h3>'
                    f'<div class="result-score">{confidence:.1%}</div>'
                    f'<span class="pill">Skor keyakinan tertinggi</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Persentase adalah probabilitas relatif yang diperkirakan model. "
                    "Akurasi bergantung pada kualitas dataset."
                )
                st.markdown("</div>", unsafe_allow_html=True)
            with right:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown("**Perbandingan skor per kategori bahasa**")
                for language, score in sorted(probs.items(), key=lambda x: -x[1]):
                    st.write(f"**{language}** · {score:.1%}")
                    st.progress(float(score))
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# HALAMAN: PENGOLAHAN AUDIO
# ===========================================================================
elif page == "Pengolahan Audio":
    st.markdown(
        '<div class="hero"><span class="pill">SPEECH TO TEXT</span>'
        '<h1>🎙️ Pengolahan Audio</h1>'
        '<p>Rekam suara melalui mikrofon atau unggah rekaman yang sudah tersimpan.</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    mode = st.radio(
        "Sumber audio",
        ["Rekam langsung dari mikrofon", "Unggah file audio"],
        horizontal=True,
    )
    audio_data = None
    if mode == "Rekam langsung dari mikrofon":
        st.info(
            "Izinkan akses mikrofon pada browser. "
            "Rekam selama beberapa detik, lalu tekan Stop. "
            "Putar ulang rekaman sebelum dianalisis."
        )
        if hasattr(st, "audio_input"):
            audio_data = st.audio_input("Rekam suara", key="live_audio")
        else:
            st.error("Fitur rekam langsung membutuhkan Streamlit 1.39 atau lebih baru.")
    else:
        audio_data = st.file_uploader(
            "Pilih file audio",
            type=["wav", "mp3", "m4a", "ogg", "webm"],
        )
    st.markdown("</div>", unsafe_allow_html=True)

    whisper_size = st.select_slider(
        "Model Whisper",
        options=["tiny", "base", "small"],
        value="base",
        help="tiny lebih cepat dan ringan; small membutuhkan lebih banyak memori.",
    )

    if audio_data is not None:
        raw = audio_data.getvalue()
        filename = getattr(audio_data, "name", "") or ""

        # Tampilkan info file dan pemutar (tidak ada JSON diagnostik)
        col_info, col_player = st.columns([1, 1])
        with col_info:
            st.markdown(
                f'<div class="panel">'
                f'<div class="muted">File audio diterima</div>'
                f'<div style="font-weight:700;color:#641044">{filename or "audio (rekaman langsung)"}</div>'
                f'<div class="muted">{len(raw):,} byte</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_player:
            st.audio(raw, format="audio/wav")

        if len(raw) < 1500:
            st.error(
                "Data rekaman terlalu kecil. "
                "Coba rekam ulang dan pastikan browser mengizinkan mikrofon."
            )
        else:
            if st.button("🎧 Transkripsikan dan deteksi bahasa", type="primary"):
                transcript = None
                error_msg = None

                with st.status("Memproses audio...", expanded=True) as status:
                    # Langkah 1: Periksa file
                    st.write("🔍 Memeriksa file audio...")
                    audio_info = inspect_audio_safe(raw, filename)
                    logger.info("Audio info: %s", audio_info)

                    # Validasi silent
                    if audio_info.get("silent") is True:
                        status.update(label="Audio hening", state="error", expanded=False)
                        st.error(
                            "Audio terdeteksi hening atau volumenya sangat rendah. "
                            "Periksa mikrofon, input perangkat, dan izin browser, lalu rekam ulang."
                        )
                        st.stop()

                    # Langkah 2: Muat model
                    st.write("🤖 Memuat model Whisper...")
                    try:
                        whisper_model = get_whisper_model(whisper_size)
                    except RuntimeError as exc:
                        status.update(label="Gagal memuat model", state="error", expanded=False)
                        st.error(str(exc))
                        st.caption(
                            "Pastikan openai-whisper sudah terpasang: "
                            "`pip install -r requirements-audio.txt`"
                        )
                        st.stop()

                    # Langkah 3: Transkripsi
                    st.write("🎵 Mentranskripsikan audio...")
                    try:
                        transcript = transcribe_audio(raw, whisper_model, filename)
                    except Exception as exc:
                        logger.exception("Transkripsi audio gagal")
                        status.update(label="Transkripsi gagal", state="error", expanded=False)
                        st.error(
                            "Gagal memproses audio. "
                            "Pastikan FFmpeg terinstal dan dapat dipanggil dari terminal."
                        )
                        st.caption(f"Detail teknis (terminal): {type(exc).__name__}")
                        st.stop()

                    # Langkah 4: Analisis bahasa
                    st.write("🌐 Menganalisis karakteristik bahasa...")
                    time.sleep(0.2)
                    st.write("✅ Proses selesai.")
                    status.update(label="Transkripsi selesai!", state="complete", expanded=False)

                # Tampilkan transkripsi
                if transcript is not None:
                    if not transcript.strip():
                        st.warning(
                            "Whisper belum menemukan ucapan. "
                            "Coba bicara lebih dekat ke mikrofon dan rekam di tempat yang lebih tenang."
                        )
                    else:
                        st.markdown('<div class="panel">', unsafe_allow_html=True)
                        st.markdown("**Hasil transkripsi**")
                        st.text_area("Teks hasil pengenalan suara", transcript, height=120)
                        st.markdown("</div>", unsafe_allow_html=True)
                        prediction_panel(transcript)


# ===========================================================================
# HALAMAN: ANALISIS FOTO
# ===========================================================================
elif page == "Analisis Foto":
    st.markdown(
        '<div class="hero"><span class="pill">IMAGE TO TEXT · OCR</span>'
        '<h1>🖼️ Analisis Gambar</h1>'
        '<p>Unggah foto berisi tulisan, ekstrak teksnya, lalu deteksi bahasa.</p></div>',
        unsafe_allow_html=True,
    )

    # Peringatan dini jika Tesseract tidak ditemukan
    if not is_tesseract_available():
        st.warning(
            "⚠️ **Tesseract OCR tidak terdeteksi** di sistem ini.\n\n"
            "Unduh dan instal dari: https://github.com/UB-Mannheim/tesseract/wiki\n\n"
            "Saat instalasi, centang bahasa **Indonesian** dan **English**. "
            "Setelah instalasi selesai, restart aplikasi ini."
        )

    image_file = st.file_uploader("Unggah gambar", type=["png", "jpg", "jpeg"])
    if image_file:
        st.image(image_file, caption="Pratinjau gambar", use_container_width=True)

        if st.button("🔎 Baca gambar dan deteksi", type="primary"):
            extracted = None
            error_msg = None

            with st.status("Memproses gambar...", expanded=True) as status:
                st.write("🖼️ Membaca gambar...")
                time.sleep(0.2)

                st.write("⚙️ Melakukan preprocessing...")
                time.sleep(0.3)

                st.write("🔡 Mengekstrak teks menggunakan OCR...")
                try:
                    extracted = extract_text_from_image(image_file)
                except RuntimeError as exc:
                    logger.warning("OCR gagal: %s", exc)
                    status.update(label="OCR gagal", state="error", expanded=False)
                    # Pesan ramah, tanpa traceback
                    err_lines = str(exc).split("\n")
                    st.error(err_lines[0])
                    for line in err_lines[1:]:
                        if line.strip():
                            st.caption(line)
                    st.stop()
                except Exception as exc:
                    logger.exception("OCR error tidak terduga")
                    status.update(label="OCR gagal", state="error", expanded=False)
                    st.error("Gagal memproses gambar. Lihat terminal untuk detail teknis.")
                    st.stop()

                st.write("🌐 Menganalisis karakteristik bahasa...")
                time.sleep(0.2)
                st.write("✅ Selesai.")
                status.update(label="OCR selesai!", state="complete", expanded=False)

            if extracted is not None:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown("**Teks hasil OCR**")
                st.text_area("Hasil ekstraksi teks dari gambar", extracted, height=130)
                st.markdown("</div>", unsafe_allow_html=True)

                if extracted.strip():
                    prediction_panel(extracted)
                else:
                    st.warning(
                        "Tidak ada teks yang terbaca. "
                        "Gunakan gambar yang lebih tajam, terang, dan tidak miring."
                    )


# ===========================================================================
# HALAMAN: PELATIHAN MODEL
# ===========================================================================
elif page == "Pelatihan Model":
    st.markdown(
        '<div class="hero"><span class="pill">MODEL TRAINING</span>'
        '<h1>🧠 Pelatihan Model</h1>'
        '<p>Latih ulang model klasifikasi dari dataset yang tersedia.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="panel">Dataset akan digunakan untuk membangun ulang model TF-IDF dan Logistic Regression.</div>',
        unsafe_allow_html=True,
    )
    if st.button("Mulai pelatihan model", type="primary"):
        import subprocess
        import sys

        with st.spinner("Melatih model..."):
            result = subprocess.run(
                [sys.executable, str(ROOT / "train_model.py")],
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
        if result.returncode == 0:
            st.success("Pelatihan selesai.")
            st.code(result.stdout[-5000:] or "Model berhasil disimpan.")
            get_classifier.clear()
        else:
            st.error("Pelatihan gagal.")
            st.code(result.stderr[-5000:])


# ===========================================================================
# HALAMAN: EVALUASI MODEL
# ===========================================================================
elif page == "Evaluasi Model":
    st.markdown(
        '<div class="hero"><span class="pill">MODEL EVALUATION</span>'
        '<h1>📊 Evaluasi Model</h1>'
        '<p>Lihat distribusi data yang tersedia untuk masing-masing bahasa.</p></div>',
        unsafe_allow_html=True,
    )
    df = pd.read_csv(ROOT / "dataset" / "data_bahasa.csv")
    a, b = st.columns(2)
    a.metric("Jumlah kalimat", len(df))
    b.metric("Jumlah kelas", df["label"].nunique())
    st.bar_chart(df["label"].value_counts())
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.warning(
        "Distribusi dataset bukan pengganti evaluasi ilmiah. "
        "Gunakan data uji terpisah dan data yang telah divalidasi penutur."
    )


# ===========================================================================
# HALAMAN: TENTANG APLIKASI
# ===========================================================================
elif page == "Tentang Aplikasi":
    st.markdown(
        '<div class="hero"><span class="pill">ABOUT THE PROJECT</span>'
        '<h1>🌺 Tentang Aplikasi</h1>'
        '<p>Prototipe identifikasi lima kategori bahasa daerah berbasis Natural Language Processing.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="panel">
    <h3>Tujuan aplikasi</h3>
    Mengidentifikasi teks sebagai Sunda Kasar, Sunda Lemes, Jawa, Aceh, atau Palembang
    melalui masukan teks, audio, dan gambar.
    <h3>Teknologi</h3>
    <ul>
      <li><b>TF-IDF + Logistic Regression</b> untuk klasifikasi teks.</li>
      <li><b>Whisper</b> untuk mengubah ucapan menjadi teks.</li>
      <li><b>Tesseract OCR</b> untuk membaca tulisan pada gambar.</li>
      <li><b>Streamlit</b> untuk antarmuka aplikasi.</li>
    </ul>
    <h3>Alur sistem</h3>
    Audio dan gambar diubah menjadi teks terlebih dahulu.
    Hasil teks kemudian dianalisis oleh model klasifikasi bahasa.
    </div>
    """, unsafe_allow_html=True)
