"""
utils/audio.py
Modul pemrosesan audio untuk SundaJawaAI.

- inspect_audio_safe  : membaca metadata tanpa pernah raise exception.
- load_whisper_model  : wrapper untuk Whisper (di-cache oleh app.py).
- transcribe_audio    : transkripsi audio ke teks.
"""
import os as _os
import shutil as _shutil

# ---------------------------------------------------------------------------
# Patch PATH untuk FFmpeg di Windows (jalankan sekali saat modul diimpor)
# Berguna jika terminal dibuka SEBELUM PATH diperbarui.
# ---------------------------------------------------------------------------
_FFMPEG_CANDIDATES = [
    r"C:\ffmpeg\ffmpeg-2026-11-09-git-16a345b022-full_build\bin",
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
    r"C:\tools\ffmpeg\bin",
]

def _patch_ffmpeg_path():
    if _shutil.which("ffmpeg"):
        return  # Sudah ada di PATH
    for candidate in _FFMPEG_CANDIDATES:
        if _os.path.isfile(_os.path.join(candidate, "ffmpeg.exe")):
            _os.environ["PATH"] = candidate + ";" + _os.environ.get("PATH", "")
            break

_patch_ffmpeg_path()
import io
import logging
import tempfile
import wave
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inspeksi metadata audio
# ---------------------------------------------------------------------------

def _inspect_wav(audio_bytes: bytes) -> dict:
    """Baca metadata WAV PCM via stdlib wave."""
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        rate = wav.getframerate()
        frame_count = wav.getnframes()
        duration = frame_count / float(rate or 1)
        frames = wav.readframes(frame_count)

        rms = None
        silent = False
        if sample_width == 2 and frames:
            try:
                import audioop
                rms = audioop.rms(frames, sample_width)
                silent = rms < 35
            except Exception:
                silent = len(frames) == 0

        return {
            "format": "WAV",
            "duration": duration,
            "channels": channels,
            "sample_rate": rate,
            "rms": rms,
            "silent": silent,
            "sample_width": sample_width,
            "frame_count": frame_count,
        }


def _inspect_with_pydub(audio_bytes: bytes, suffix: str) -> dict:
    """Gunakan pydub untuk membaca metadata format non-WAV."""
    from pydub import AudioSegment  # type: ignore

    seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=suffix.lstrip("."))
    duration = len(seg) / 1000.0
    rms = seg.rms
    silent = rms < 50
    return {
        "format": suffix.upper().lstrip("."),
        "duration": duration,
        "channels": seg.channels,
        "sample_rate": seg.frame_rate,
        "rms": rms,
        "silent": silent,
        "sample_width": seg.sample_width,
        "frame_count": int(seg.frame_count()),
    }


def inspect_audio_safe(audio_bytes: bytes, filename: str = "") -> dict:
    """
    Baca metadata audio tanpa pernah raise exception.
    Mendukung WAV, MP3, M4A, OGG, WEBM, dsb.
    Kembalikan dict dengan kunci: format, duration, channels, sample_rate,
    rms, silent, sample_width, frame_count.
    """
    suffix = Path(filename).suffix.lower() if filename else ".wav"
    if not suffix:
        suffix = ".wav"

    # --- Coba WAV dulu (paling umum dari st.audio_input) ---
    try:
        return _inspect_wav(audio_bytes)
    except Exception as e_wav:
        logger.debug("WAV parse gagal (%s), coba pydub...", e_wav)

    # --- Coba pydub untuk format lain ---
    try:
        return _inspect_with_pydub(audio_bytes, suffix)
    except ImportError:
        logger.debug("pydub tidak terinstal; metadata non-WAV tidak tersedia.")
    except Exception as e_pydub:
        logger.debug("pydub parse gagal: %s", e_pydub)

    # --- Fallback minimal ---
    return {
        "format": suffix.upper().lstrip(".") or "UNKNOWN",
        "duration": None,
        "channels": None,
        "sample_rate": None,
        "rms": None,
        "silent": len(audio_bytes) < 1500,
        "sample_width": None,
        "frame_count": None,
    }


# ---------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------

def load_whisper_model(model_size: str = "base"):
    """
    Muat model Whisper.
    Fungsi ini dipanggil dari app.py yang mem-cache hasilnya dengan
    @st.cache_resource sehingga model tidak dimuat ulang setiap sesi.
    """
    try:
        import whisper  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Paket Whisper belum terpasang. Jalankan: pip install -r requirements-audio.txt"
        ) from exc

    logger.info("Memuat model Whisper '%s'...", model_size)
    model = whisper.load_model(model_size)
    logger.info("Model Whisper '%s' berhasil dimuat.", model_size)
    return model


def transcribe_audio(audio_bytes: bytes, model, filename: str = "") -> str:
    """
    Transkripsi audio ke teks menggunakan model Whisper yang sudah dimuat.

    Parameters
    ----------
    audio_bytes : bytes
        Isi file audio.
    model : whisper.Whisper
        Model Whisper yang sudah dimuat (dari load_whisper_model).
    filename : str
        Nama file asli (digunakan untuk menentukan ekstensi).

    Returns
    -------
    str
        Hasil transkripsi (stripped).
    """
    if not audio_bytes or len(audio_bytes) < 1500:
        raise RuntimeError(
            "Data audio kosong atau terlalu pendek. Rekam ulang atau pilih file audio lain."
        )

    # Tentukan suffix yang tepat
    suffix = Path(filename).suffix.lower() if filename else ""
    if not suffix or suffix not in {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"}:
        # st.audio_input selalu mengembalikan WAV
        suffix = ".wav"

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        result = model.transcribe(
            temp_path,
            fp16=False,          # CPU-safe
            language=None,       # auto-detect
            verbose=False,
        )
        text = result.get("text", "").strip()
        logger.info("Transkripsi selesai: %d karakter", len(text))
        return text

    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass
