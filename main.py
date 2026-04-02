from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import os
import time
import logging
import json
import cv2
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import asyncio
import onnxruntime as ort
import gc
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# SABİTLER
ONNX_MODEL_PATH  = "./saved_model/Chroma.onnx"
CLASS_NAMES_PATH = "./saved_model/class_names.json"
IMAGE_SIZE       = (224, 224)
MAX_FILE_SIZE_MB    = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000"
).split(",")


# RENK TANIMLAMALARI
COLOR_NAMES = {
    "kirmizi_alt":   ([0,   100, 100], [10,  255, 255]),
    "kirmizi_ust":   ([170, 100, 100], [180, 255, 255]),
    "turuncu":       ([10,  100, 100], [25,  255, 255]),
    "sari":          ([25,  100, 100], [35,  255, 255]),
    "yesil":         ([35,  50,  50],  [85,  255, 255]),
    "camgobegi":     ([85,  50,  50],  [100, 255, 255]),
    "mavi":          ([100, 50,  50],  [130, 255, 255]),
    "mor":           ([130, 50,  50],  [160, 255, 255]),
    "pembe":         ([155, 40,  100], [170, 255, 255]),
    "beyaz":         ([0,   0,   200], [180, 30,  255]),
    "gri":           ([0,   0,   80],  [180, 30,  200]),
    "siyah":         ([0,   0,   0],   [180, 255, 80]),
}


# RESPONSE ŞEMALARI
class DigerOlasilik(BaseModel):
    stil: str
    guven_yuzdesi: float

class StilSonucu(BaseModel):
    tahmin: str
    guven: float
    diger_olasiliklar: list[DigerOlasilik]

class DominantRenk(BaseModel):
    rgb: list[int]
    yuzde: float
    isim: str
    notr: bool

class GenelIstatistikler(BaseModel):
    ort_doygunluk: float
    ort_parlaklik: float
    renk_cesitliligi: float
    notr_oran: float

class UyumAnalizi(BaseModel):
    tur: str
    aciklama: str
    skor: float

class RenkAnalizi(BaseModel):
    dominant_colors: list[DominantRenk]
    genel_istatistikler: GenelIstatistikler
    uyum_analizi: UyumAnalizi
    stil_tahmini: str

class AnalizSonucu(BaseModel):
    status: str
    dosya_boyutu_mb: float
    cikarim_suresi_ms: float
    stil: StilSonucu
    renk_analizi: RenkAnalizi


# GLOBAL DEĞİŞKENLER
model       = None
class_names = []
executor    = ThreadPoolExecutor(max_workers=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names

    get_rembg_session()

    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)
    else:
        logger.error(f"'{CLASS_NAMES_PATH}' bulunamadi.")

    if os.path.exists(ONNX_MODEL_PATH):
        try:
            model = ort.InferenceSession(
                ONNX_MODEL_PATH,
                providers=["CPUExecutionProvider"]
            )
            model_output_size = model.get_outputs()[0].shape[-1]
            if model_output_size != len(class_names):
                logger.warning(
                    f"Uyari: Model {model_output_size} sinif uretiyor "
                    f"ama class_names.json'da {len(class_names)} eleman var!"
                )
            logger.info("ONNX modeli basariyla yuklendi.")
        except Exception as e:
            logger.error(f"Model yuklenirken hata: {e}")
    else:
        logger.error(f"'{ONNX_MODEL_PATH}' bulunamadi. Sunucu modelsiz basladi.")

    yield
    executor.shutdown(wait=False)
    logger.info("Sunucu kapatildi.")


app = FastAPI(title="Chroma_AI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


# RENK ANALİZİ

def _notr_mu(hsv: np.ndarray) -> bool:
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if s < 30:
        return True
    if v < 25:
        return True
    if v > 230 and s < 40:
        return True
    return False


def get_color_name(rgb):
    pixel = np.uint8([[rgb]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0][0]

    if _notr_mu(hsv):
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        if v < 50:
            return "siyah"
        if v > 200 and s < 30:
            return "beyaz"
        return "gri"

    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    kirmizi_alt = COLOR_NAMES["kirmizi_alt"]
    kirmizi_ust = COLOR_NAMES["kirmizi_ust"]
    if (
        (kirmizi_alt[0][0] <= h <= kirmizi_alt[1][0] and
         kirmizi_alt[0][1] <= s <= kirmizi_alt[1][1] and
         kirmizi_alt[0][2] <= v <= kirmizi_alt[1][2])
        or
        (kirmizi_ust[0][0] <= h <= kirmizi_ust[1][0] and
         kirmizi_ust[0][1] <= s <= kirmizi_ust[1][1] and
         kirmizi_ust[0][2] <= v <= kirmizi_ust[1][2])
    ):
        return "kirmizi"

    skip = {"kirmizi_alt", "kirmizi_ust"}
    for name, (lower, upper) in COLOR_NAMES.items():
        if name in skip:
            continue
        if lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]:
            return name

    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def _golge_filtrele(renkli_hsv: list) -> list:
    if len(renkli_hsv) <= 1:
        return renkli_hsv

    kaldir = set()
    for i in range(len(renkli_hsv)):
        for j in range(i + 1, len(renkli_hsv)):
            hi, si, vi = renkli_hsv[i]
            hj, sj, vj = renkli_hsv[j]
            hue_fark = abs(hi - hj)
            hue_fark = min(hue_fark, 360.0 - hue_fark)
            bri_fark = abs(vi - vj)
            if hue_fark < 25 and bri_fark > 55:
                if vi < vj:
                    kaldir.add(i)
                else:
                    kaldir.add(j)

    return [r for idx, r in enumerate(renkli_hsv) if idx not in kaldir]


def _renk_uyum_turu(dominant_colors_rgb: np.ndarray) -> dict:
    # HSV değerlerini çıkar
    tum_hsv = []
    for rgb in dominant_colors_rgb:
        pixel = np.uint8([[rgb]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0][0]
        tum_hsv.append((float(hsv[0]) * 2.0, float(hsv[1]), float(hsv[2]), _notr_mu(hsv)))

    renkli_hsv_ham = [(h, s, v) for h, s, v, notr in tum_hsv if not notr]
    notr_sayisi    = sum(1 for *_, notr in tum_hsv if notr)
    toplam         = len(tum_hsv)

    # Gölgeleri filtrele
    renkli_hsv = _golge_filtrele(renkli_hsv_ham)

    # Tür tespiti
    if len(renkli_hsv) < 2:
        tur = "Monokromatik / Nötr"
        ort_fark = hue_std = 0.0
    else:
        hue_vals = [h for h, s, v in renkli_hsv]
        farklar = []
        for i in range(len(hue_vals)):
            for j in range(i + 1, len(hue_vals)):
                fark = abs(hue_vals[i] - hue_vals[j])
                farklar.append(min(fark, 360.0 - fark))
        ort_fark = float(np.mean(farklar))
        max_fark = float(np.max(farklar))
        hue_std  = float(np.std(hue_vals))

        if hue_std < 20:
            tur = "Monokromatik"
        elif ort_fark < 40:
            tur = "Analog"
        elif 160.0 <= max_fark <= 200.0:
            tur = "Komplementer"
        elif 130.0 <= max_fark < 160.0:
            tur = "Split-Komplementer"
        elif 100.0 <= max_fark <= 140.0 and len(renkli_hsv) >= 3:
            tur = "Triadik"
        elif 60.0 <= ort_fark < 100.0:
            tur = "Yarı-Komplementer"
        else:
            tur = "Karma / Çok Renkli"

    # Doygunluk tutarlılığı
    if len(renkli_hsv) >= 2:
        sat_vals = [s for h, s, v in renkli_hsv]
        sat_std  = float(np.std(sat_vals))
        doygunluk_skoru = max(0.0, 100.0 - (sat_std / 80.0) * 100.0)
    else:
        doygunluk_skoru = 80.0

    #  Parlaklık hiyerarşisi
    bri_vals = [v for h, s, v, notr in tum_hsv]
    bri_fark = max(bri_vals) - min(bri_vals)
    if bri_fark < 30:
        parlaklik_skoru = 40.0
    elif bri_fark < 80:
        parlaklik_skoru = 70.0
    elif bri_fark < 160:
        parlaklik_skoru = 95.0
    else:
        parlaklik_skoru = 75.0

    #  Renk yoğunluğu
    renkli_sayisi = len(renkli_hsv)
    if renkli_sayisi <= 1:
        yogunluk_skoru = 92.0
    elif renkli_sayisi == 2:
        yogunluk_skoru = 88.0
    elif renkli_sayisi == 3:
        yogunluk_skoru = 80.0
    elif renkli_sayisi == 4:
        yogunluk_skoru = 65.0
    else:
        yogunluk_skoru = 45.0

    # Nötr tampon bonusu
    notr_oran = notr_sayisi / max(toplam, 1)
    if notr_oran >= 0.4:
        notr_bonus = 20.0
    elif notr_oran >= 0.2:
        notr_bonus = 12.0
    elif notr_oran > 0:
        notr_bonus = 6.0
    else:
        notr_bonus = 0.0

    # Genel skor
    ham_skor = (
        doygunluk_skoru * 0.30 +
        parlaklik_skoru * 0.25 +
        yogunluk_skoru  * 0.25 +
        notr_bonus
    )
    skor = round(max(5.0, min(100.0, ham_skor)), 1)

    aciklama = _uyum_aciklamasi(tur, skor, doygunluk_skoru, parlaklik_skoru, yogunluk_skoru, notr_bonus)
    return {"tur": tur, "aciklama": aciklama, "skor": skor}


def _uyum_aciklamasi(
    tur: str,
    skor: float,
    doygunluk_skoru: float,
    parlaklik_skoru: float,
    yogunluk_skoru: float,
    notr_bonus: float,
) -> str:

    zayif = min(
        [("doygunluk", doygunluk_skoru),
         ("parlaklik", parlaklik_skoru),
         ("yogunluk",  yogunluk_skoru)],
        key=lambda x: x[1]
    )[0]

    if zayif == "yogunluk" and tur != "Karma / Çok Renkli":
        zayif = min(
            [("doygunluk", doygunluk_skoru),
             ("parlaklik",  parlaklik_skoru)],
            key=lambda x: x[1]
        )[0]

    if skor >= 82:
        if notr_bonus >= 12:
            return "Nötr tonlar paleti mükemmel dengelemiş. Güçlü bir zemin üzerine oturmuş, bakışı yönlendiren bir kombin."
        if tur in ("Monokromatik", "Analog"):
            return "Tonlar arasındaki geçiş akıcı ve kasıtlı görünüyor. Göz dinleniyor, palet kendinden emin konuşuyor."
        if tur == "Komplementer":
            return "Zıt renkler ustaca dengelenmiş. Bu kontrast göz yormuyor, tam tersi paleti güçlendiriyor."
        if tur == "Split-Komplementer":
            return "Zıt rengin iki yanı dengeli tutulmuş. Kontrastlı ama sınırda değil, göz rahat geziniyor."
        if tur == "Triadik":
            return "Üç renk eşit ağırlıkta dağıtılmış. Bu cesaret isteyen bir seçim ve karşılığını almış."
        return "Palet dengeli ve okunaklı. Renkler birbirini bastırmıyor, her biri yerli yerinde duruyor."

    elif skor >= 65:
        if zayif == "doygunluk":
            return "Palet genel olarak iyi ama renkler benzer canlılıkta değil. Biri öne çıkarken diğerleri soluklaşıyor."
        if zayif == "parlaklik":
            return "Renk seçimi makul ama açık-koyu dengesi henüz oturmamış. Biraz daha net bir hiyerarşi kombini güçlendirir."
        if zayif == "yogunluk":
            return "Çok sayıda renk dikkat dağıtmaya başlıyor. İki veya üç renge odaklanmak paleti çok daha güçlü kılar."
        return "İyi bir başlangıç noktası ama palet biraz daha sadeleştirilebilir. Küçük değişiklikler büyük fark yaratır."

    elif skor >= 45:
        if zayif == "doygunluk":
            return "Renkler arasındaki canlılık farkı kombini dengesiz kılıyor. Hepsi aynı yoğunlukta olsaydı çok daha iyi görünürdü."
        if zayif == "parlaklik":
            return "Parlaklık farkları ya çok az ya da çok fazla. Kasıtlı bir kontrast kurmadan bu ara değerler paleti muğlak bırakıyor."
        if zayif == "yogunluk":
            return "Bu kadar farklı renk aynı anda pek çalışmıyor. Baskın iki renk seç, diğerlerini aksan olarak kullan."
        return "Palet bir yere varmaya çalışıyor ama henüz varmamış. Biraz daha odaklı bir seçim işe yarar."

    else:
        if zayif == "doygunluk":
            return "Renkler çok farklı canlılık seviyelerinde ve bu kombinasyonu dağınık gösteriyor. Palet bir bütün olarak okunmuyor."
        if zayif == "parlaklik":
            return "Parlaklık dengesi kurulamamış. Renkler birbirini ne tamamlıyor ne de kasıtlı bir kontrast oluşturuyor."
        if zayif == "yogunluk":
            return "Çok fazla renk aynı anda konuşuyor ve hiçbiri duyulmuyor. Bu kombinasyondan iki rengi çıkarmak her şeyi değiştirir."
        return "Palet şu haliyle birbirine bağlı bir bütün oluşturmuyor. Daha az renkle daha güçlü bir sonuç mümkün."




def _sezon_tahmini(avg_saturation: float, avg_brightness: float, hue_std: float) -> str:
    sat_norm = avg_saturation / 255.0
    bri_norm = avg_brightness / 255.0

    if sat_norm < 0.15 and bri_norm > 0.6:
        return "Kış tonları"
    if sat_norm < 0.25 and bri_norm < 0.45:
        return "Sonbahar tonları"
    if sat_norm > 0.55 and bri_norm > 0.55:
        return "Yaz tonları"
    if sat_norm > 0.35 and bri_norm > 0.5 and hue_std > 25:
        return "İlkbahar tonları"
    if bri_norm > 0.65 and sat_norm < 0.4:
        return "Narin tonlar"
    return "Çok renkli tonlar"


_rembg_session = None

def get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        _rembg_session = new_session("u2net")
    return _rembg_session


def _rembg_mask(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    from rembg import remove
    session = get_rembg_session()

    result_bytes = remove(image_bytes, session=session)
    result       = np.frombuffer(result_bytes, dtype=np.uint8)
    result_img   = cv2.imdecode(result, cv2.IMREAD_UNCHANGED)

    img_rgb = cv2.cvtColor(result_img[:, :, :3], cv2.COLOR_BGR2RGB)
    alpha   = result_img[:, :, 3]

    img_rgb = cv2.resize(img_rgb, (200, 200))
    alpha   = cv2.resize(alpha,   (200, 200))
    mask    = (alpha > 30).astype(np.uint8) * 255

    return img_rgb, mask


def rgb_to_hsv_numpy(pixels_rgb: np.ndarray) -> np.ndarray:
    pixels_norm = pixels_rgb / 255.0
    r, g, b = pixels_norm[:, 0], pixels_norm[:, 1], pixels_norm[:, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin + 1e-8

    h = np.zeros_like(r)
    mask_r = cmax == r
    mask_g = cmax == g
    mask_b = cmax == b
    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 360)
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120)
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240)
    h = h / 2.0

    s = np.where(cmax == 0, 0, (diff / cmax) * 255)
    v = cmax * 255

    return np.stack([h, s, v], axis=1)


def analyze_colors(image_bytes: bytes, n_colors: int = 5) -> dict:
    if not image_bytes:
        return {"dominant_colors": [], "genel_istatistikler": {}}

    try:
        img_rgb, mask = _rembg_mask(image_bytes)
    except Exception as e:
        logger.error(f"[rembg HATA] {type(e).__name__}: {e}")
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"dominant_colors": [], "genel_istatistikler": {}}
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (200, 200))
        mask    = np.ones((200, 200), dtype=np.uint8) * 255

    pixels = img_rgb[mask > 0].astype(np.float32)
    if len(pixels) < 100:
        pixels = img_rgb.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(
        pixels, n_colors, None, criteria, 15, cv2.KMEANS_PP_CENTERS
    )

    unique, counts  = np.unique(labels, return_counts=True)
    percentages     = counts / counts.sum()
    sorted_idx      = np.argsort(-percentages)
    dominant_colors = centers[sorted_idx].astype(int)
    percentages     = percentages[sorted_idx]

    hsv_pixels     = rgb_to_hsv_numpy(pixels)
    avg_saturation = float(np.mean(hsv_pixels[:, 1]))
    avg_brightness = float(np.mean(hsv_pixels[:, 2]))
    std_hue        = float(np.std(hsv_pixels[:, 0]))

    renkli_dominant = [
        dominant_colors[i] for i in range(min(5, len(dominant_colors)))
        if not _notr_mu(
            cv2.cvtColor(np.uint8([[dominant_colors[i]]]), cv2.COLOR_RGB2HSV)[0][0]
        )
    ]

    uyum  = _renk_uyum_turu(
        np.array(renkli_dominant) if renkli_dominant else dominant_colors[:3]
    )
    sezon = _sezon_tahmini(avg_saturation, avg_brightness, std_hue)

    s_vals    = hsv_pixels[:, 1]
    v_vals    = hsv_pixels[:, 2]
    notr_mask = (s_vals < 30) | (v_vals < 25) | ((v_vals > 230) & (s_vals < 40))
    notr_oran = round(float(np.mean(notr_mask)) * 100, 1)

    return {
        "dominant_colors": [
            {
                "rgb":   dominant_colors[i].tolist(),
                "yuzde": round(float(percentages[i]) * 100, 1),
                "isim":  get_color_name(dominant_colors[i]),
                "notr":  _notr_mu(
                    cv2.cvtColor(
                        np.uint8([[dominant_colors[i]]]), cv2.COLOR_RGB2HSV
                    )[0][0]
                ),
            }
            for i in range(n_colors)
        ],
        "genel_istatistikler": {
            "ort_doygunluk":    round(avg_saturation / 255 * 100, 1),
            "ort_parlaklik":    round(avg_brightness  / 255 * 100, 1),
            "renk_cesitliligi": round(std_hue, 1),
            "notr_oran":        notr_oran,
        },
        "uyum_analizi": {
            "tur":      uyum["tur"],
            "aciklama": uyum["aciklama"],
            "skor":     uyum["skor"],
        },
        "stil_tahmini": sezon,
    }


# INFERENCE

def rgba_to_rgb_white_bg(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.paste(image, mask=image.split()[3])
    return background.convert("RGB")

def run_inference(img_array: np.ndarray) -> np.ndarray:
    input_name = model.get_inputs()[0].name
    return model.run(None, {input_name: img_array})[0]

def prepare_for_model(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode == "RGBA":
        image = rgba_to_rgb_white_bg(image_bytes)
    else:
        image = image.convert("RGB")
    image     = image.resize(IMAGE_SIZE)
    img_array = np.array(image, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)

def run_color_analysis(image_bytes: bytes) -> dict:
    return analyze_colors(image_bytes)


# ENDPOINT
@app.post("/api/analyze-image", response_model=AnalizSonucu)
async def analyze_image(file: UploadFile = File(...)):

    if model is None:
        raise HTTPException(status_code=503, detail="Model su an kullanilamiyor.")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Desteklenmeyen dosya tipi. JPEG, PNG, WEBP kabul edilir.")

    icerik = await file.read()
    if len(icerik) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"Maksimum dosya boyutu: {MAX_FILE_SIZE_MB}MB")

    try:
        loop = asyncio.get_running_loop()
        
        baslangic = time.perf_counter()

        # 1. AŞAMA: Sadece model için tensör hazırlığı ve çıkarım
        img_tensor = await loop.run_in_executor(executor, prepare_for_model, icerik)
        tahminler = await loop.run_in_executor(executor, run_inference, img_tensor)
        
        tahmin_dizisi   = tahminler[0]
        top_k           = min(3, len(tahmin_dizisi))
        en_iyi_indexler = np.argsort(tahmin_dizisi)[-top_k:][::-1]

        stil_listesi = [
            {
                "stil": class_names[i] if i < len(class_names) else f"Sinif_{i}",
                "guven_yuzdesi": round(float(tahmin_dizisi[i]) * 100, 1),
            }
            for i in en_iyi_indexler
        ]

        del img_tensor 
        del tahminler
        gc.collect() 

        renk_sonucu = await loop.run_in_executor(executor, run_color_analysis, icerik)

        bitis = time.perf_counter()
        cikarim_suresi_ms = round((bitis - baslangic) * 1000, 2)

        logger.info(
            f"Analiz tamamlandi | stil={stil_listesi[0]['stil']} "
            f"guven={stil_listesi[0]['guven_yuzdesi']}% "
            f"sure={cikarim_suresi_ms}ms"
        )

        return AnalizSonucu(
            status="success",
            dosya_boyutu_mb=round(len(icerik) / (1024 * 1024), 2),
            cikarim_suresi_ms=cikarim_suresi_ms,
            stil=StilSonucu(
                tahmin=stil_listesi[0]["stil"],
                guven=stil_listesi[0]["guven_yuzdesi"],
                diger_olasiliklar=[
                    DigerOlasilik(**s) for s in stil_listesi[1:]
                ],
            ),
            renk_analizi=RenkAnalizi(
                dominant_colors=[
                    DominantRenk(**r) for r in renk_sonucu["dominant_colors"]
                ],
                genel_istatistikler=GenelIstatistikler(
                    **renk_sonucu["genel_istatistikler"]
                ),
                uyum_analizi=UyumAnalizi(
                    **renk_sonucu["uyum_analizi"]
                ),
                stil_tahmini=renk_sonucu["stil_tahmini"],
            ),
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Beklenmeyen hata: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Goruntu analizi sirasinda bir hata olustu.")
