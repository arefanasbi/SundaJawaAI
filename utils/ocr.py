"""
utils/ocr.py
Modul OCR untuk SundaJawaAI.

- find_tesseract()         : mencari path Tesseract di Windows secara otomatis.
- is_tesseract_available() : cek ketersediaan tanpa raise exception.
- preprocess_image()       : grayscale + contrast + thresholding.
- extract_text_from_image(): OCR gambar dengan preprocessing.
"""
import logging
import os
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lokasi Tesseract umum di Windows
# ---------------------------------------------------------------------------
_TESSERACT_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",   # ← lokasi terdeteksi
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\{user}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    r"C:\tools\Tesseract-OCR\tesseract.exe",
]


def find_tesseract() -> str | None:
    """
    Cari executable Tesseract.
    Urutan: PATH → variabel env TESSERACT_CMD → lokasi umum Windows.
    Kembalikan path string jika ditemukan, None jika tidak.
    """
    # 1. Cek PATH sistem
    if shutil.which("tesseract"):
        return shutil.which("tesseract")

    # 2. Variabel lingkungan khusus
    env_path = os.environ.get("TESSERACT_CMD", "")
    if env_path and Path(env_path).is_file():
        return env_path

    # 3. Lokasi umum Windows
    for candidate in _TESSERACT_CANDIDATES:
        expanded = candidate.format(user=os.environ.get("USERNAME", ""))
        if Path(expanded).is_file():
            return expanded

    return None


def _configure_tesseract() -> bool:
    """
    Konfigurasi pytesseract.tesseract_cmd.
    Kembalikan True jika berhasil dikonfigurasi, False jika tidak ditemukan.
    """
    try:
        import pytesseract  # type: ignore

        path = find_tesseract()
        if path:
            pytesseract.pytesseract.tesseract_cmd = path
            logger.info("Tesseract ditemukan: %s", path)
            return True

        logger.warning("Tesseract tidak ditemukan di PATH maupun lokasi umum Windows.")
        return False
    except ImportError:
        logger.error("pytesseract tidak terinstal.")
        return False


def is_tesseract_available() -> bool:
    """
    Cek apakah Tesseract tersedia dan dapat dijalankan.
    Aman dipanggil kapan saja tanpa raise exception.
    """
    try:
        import pytesseract

        _configure_tesseract()
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Preprocessing gambar
# ---------------------------------------------------------------------------

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocessing untuk meningkatkan akurasi OCR:
    1. Konversi ke RGB (hindari mode RGBA/P yang bermasalah)
    2. Grayscale
    3. Upscale jika terlalu kecil
    4. Peningkatan kontras
    5. Sharpening
    6. Thresholding (binarisasi)
    """
    # Pastikan RGB
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    # Grayscale
    gray = image.convert("L")

    # Upscale jika resolusi rendah (OCR lebih akurat pada gambar besar)
    w, h = gray.size
    min_dim = 800
    if max(w, h) < min_dim:
        scale = min_dim / max(w, h)
        gray = gray.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Tingkatkan kontras
    gray = ImageEnhance.Contrast(gray).enhance(2.0)

    # Sharpen
    gray = gray.filter(ImageFilter.SHARPEN)

    # Binarisasi (threshold manual untuk reproduksibilitas)
    try:
        # Gunakan OpenCV jika tersedia (hasil lebih baik)
        import cv2  # type: ignore
        import numpy as np

        arr = np.array(gray)
        _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gray = Image.fromarray(thresh)
    except ImportError:
        # Fallback: PIL point-threshold sederhana
        gray = gray.point(lambda p: 255 if p > 140 else 0)

    return gray


# ---------------------------------------------------------------------------
# Ekstraksi teks
# ---------------------------------------------------------------------------

def extract_text_from_image(uploaded_file) -> str:
    """
    Ekstrak teks dari gambar menggunakan Tesseract OCR.

    Parameters
    ----------
    uploaded_file : file-like atau path
        Objek file dari Streamlit atau path ke gambar.

    Returns
    -------
    str
        Teks hasil OCR (stripped).

    Raises
    ------
    RuntimeError
        Jika Tesseract tidak ditemukan atau gambar tidak dapat dibuka.
    """
    import pytesseract  # type: ignore

    # Konfigurasi path Tesseract
    if not _configure_tesseract():
        raise RuntimeError(
            "Tesseract OCR tidak ditemukan.\n\n"
            "Unduh dan instal dari: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "Pastikan memilih bahasa 'Indonesian' dan 'English' saat instalasi.\n"
            "Setelah instal, restart aplikasi.\n\n"
            "Atau set variabel lingkungan TESSERACT_CMD ke path tesseract.exe Anda."
        )

    # Buka gambar
    try:
        image = Image.open(uploaded_file)
    except Exception as exc:
        raise RuntimeError(f"File gambar tidak dapat dibuka: {exc}") from exc

    # Preprocessing
    try:
        processed = preprocess_image(image)
    except Exception as exc:
        logger.warning("Preprocessing gagal, gunakan gambar asli: %s", exc)
        processed = image.convert("RGB")

    # OCR — coba bahasa ind+eng, fallback ke eng saja
    config = "--oem 3 --psm 6"
    text = ""
    for lang in ("ind+eng", "eng"):
        try:
            text = pytesseract.image_to_string(processed, lang=lang, config=config)
            if text.strip():
                break
        except pytesseract.TesseractError as e:
            logger.warning("OCR dengan lang='%s' gagal: %s", lang, e)
            if lang == "eng":
                raise RuntimeError(
                    f"OCR gagal: {e}\n"
                    "Pastikan Tesseract terinstal dengan data bahasa 'ind' dan 'eng'."
                ) from e

    return text.strip()
