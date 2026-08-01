#----------------------------------------------------------------------------
# 1. GEREKLİ KÜTÜPHANELERİN YÜKLENMESİ (IMPORT LIBRARIES)
#----------------------------------------------------------------------------
import mediapipe as mp
import cv2
import numpy
import time

#----------------------------------------------------------------------------
# 2. YARDIMCI FONKSİYONLARIN TANIMLANMASI
#----------------------------------------------------------------------------

#OpenCv ile kamera başlatma
cap = cv2.VideoCapture(0)   # Varsayılan dahili kamerayı index0'a bağlama


#mediapipe face mesh modülü:
mp_face_mesh = mp.solutions.face_mesh

# Face Mesh modelini başlatma
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces = 1,          # Aynı anda 1 yüz takip edilecek
    static_image_mode= False,
    refine_landmarks= True,     # Dudak ve göz detay noktalarını daha hassas hale getirildi
    min_detection_confidence= 0.5,  # Tespit için güven skoru
    min_tracking_confidence= 0.5    # Takip için güven skoru
)

# Yüz meshinden kullanılacak one'li landmark indexlerini al
# NOT: Buradaki sayılar rastgele yazılmamış olup mediapipe kütüphanesinin belirlenen index numaralarıdır
LEFT_EYE = [159, 145]           # Üst ve alt göz kapağı
LEFT_BROW = [65, 158]           # Kaş ile göz arası
MOUTH = [13,14]                 # üst vee alt dudak
MOUTH_LEFT_RIGHT = [69,291]     # Dudağın solu ve sağı
MOUTH_TOP_BOTTOM = [13,14]      # Dudak üst ve altı
BROW_LEFT_RIGHT = [70,300]


# Ham metrikleri (henüz duygu kararı vermeden) hesaplayan fonksiyon.
def get_metrics(landmarks, image_widht, image_height):

    def get_point(index):
        """
        index değerini geri döndüren yardımcı fonksiyon
        """
        _landmarks = landmarks[index]
        return numpy.array([int(_landmarks.x * image_widht), int(_landmarks.y * image_height)])

    # --- Referans ölçüler (yüzün genel boyutu) ---
    face_left = get_point(234)
    face_right = get_point(454)
    face_width = numpy.linalg.norm(face_left - face_right)

    # Yüzün yüksekliği (alın-çene). Dikey ölçümleri genişliğe değil yüksekliğe göre
    # normalize etmek daha tutarlıdır, çünkü kameraya yakınlaşıp uzaklaşma oranı bozmaz.
    face_top = get_point(10)
    face_bottom = get_point(152)
    face_height = numpy.linalg.norm(face_top - face_bottom)

    # --- Kaş ve göz ölçümleri ---
    brow_point = get_point(65)
    eye_top = get_point(159)
    eye_bottom = get_point(145)

    brow_lift = numpy.linalg.norm(brow_point - eye_top)   # kaş-göz mesafesi (şaşkınlık/korku)
    eye_open = numpy.linalg.norm(eye_top - eye_bottom)     # göz kapağı açıklığı (uykululuk)

    # --- Ağız ölçümleri ---
    mouth_left = get_point(61)
    mouth_right = get_point(291)
    mouth_top = get_point(13)
    mouth_bottom = get_point(14)

    mouth_width = numpy.linalg.norm(mouth_left - mouth_right)   # ağız genişliği (mutluluk)
    mouth_open = numpy.linalg.norm(mouth_top - mouth_bottom)     # ağız dikey açıklığı (şaşkınlık/korku)

    # Dudak köşelerinin, dudağın orta noktasına göre ne kadar "düştüğü" (üzgün ifade için)
    mouth_center_y = (mouth_top[1] + mouth_bottom[1]) / 2
    mouth_corner_y = (mouth_left[1] + mouth_right[1]) / 2
    corner_drop = mouth_corner_y - mouth_center_y   # pozitifse köşeler ortadan daha aşağıda

    # Normalize etme
    return {
        "brow":   brow_lift   / face_width,
        "eye":    eye_open    / face_height,
        "mwidth": mouth_width / face_width,
        "mopen":  mouth_open  / face_height,
        "drop":   corner_drop / face_height,
    }


# Kalibrasyon sırasında ölçülen "nötr yüz" değerlerine göre (baseline) esnek eşikler.
def detect_emotion(metrics, baseline):

    eye_ratio    = metrics["eye"]   / baseline["eye"]     # nötre göre göz ne kadar açık/kapalı
    brow_ratio   = metrics["brow"]  / baseline["brow"]    # nötre göre kaş ne kadar kalkmış


    # baseline mopen sıfıra çok yakın çıkabiliyor (nötr yüzde ağız neredeyse hiç açık değil),
    # o durumda gerçek değeri sabit küçük bir sayıya bölüyoruz ki oran patlamasın/anlamsızlaşmasın
    mopen_base  = baseline["mopen"]
    if mopen_base > 1e-6:
        mopen_ratio = metrics["mopen"] / mopen_base
    else:
        mopen_ratio = metrics["mopen"] / 0.01

    mopen_abs    = metrics["mopen"]                       # ağzın gerçekten açık olup olmadığını da kontrol etme
    drop_delta   = metrics["drop"]  - baseline["drop"]     # nötre göre köşelerin ne kadar aşağı indiği
    mwidth_ratio = metrics["mwidth"] / baseline["mwidth"]  # nötre göre ağız ne kadar genişlemiş

    # Gözler nötre göre belirgin küçülmüşse -> uykulu
    if eye_ratio < 0.75:
        return "Uykulu"

    # Kaş belirgin kalkmış ve ağız gerçekten açılmışsa -> şaşkın
    elif brow_ratio > 1.20 and mopen_ratio > 1.5 and mopen_abs > 0.03:
        return "Saskin"

    # Kaş belirgin kalkmış ve gözler nötre göre iri açılmışsa  -> korkmuş
    elif brow_ratio > 1.20 and eye_ratio > 1.15:
        return "Korkmus"

    # 4) Dudak köşeleri nötre göre belirgin aşağı düşmüşse ve ağız gerilmemişse -> üzgün
    elif drop_delta > 0.008 and mwidth_ratio < 1.05:
        return "Uzgun"

    # 5) Ağız nötre göre belirgin genişlemişse -> mutlu
    elif mwidth_ratio > 1.15:
        return "Mutlu"

    else:
        return "Notr"


#----------------------------------------------------------------------------
# 3. WEB CAM İLE DUYGU TANIMA
#----------------------------------------------------------------------------

mp_draw = mp.solutions.drawing_utils    # Yüz üzerindeki noktaların çizilmesi
draw_spec = mp_draw.DrawingSpec(thickness = 1,circle_radius = 1)    # Çizilecek nokta ve çizgilerin kalınlığı ve yarıçapını ayarlama

CALIBRATION_SECONDS = 3.0   # Kalibrasyon için nötr yüzde sabit durulacak süre
calibration_samples = []    # Kalibrasyon sırasında toplanan metrik listeleri
calibration_start = None    # Kalibrasyonun başladığı zaman damgası
baseline = None             # Kalibrasyon bitince buraya nötr yüz ortalaması yazılacak


while True:
    ret, frame = cap.read()     #ret: Okuma başarılı mı? (True/False), frame: Görüntü matrisi

    if not ret:
        break

    # Görüntüyü RGB'ye çevirme medipipe için
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = face_mesh.process((rgb_frame))

    # Ekran boyutu
    h,w,_ = frame.shape

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            metrics = get_metrics(face_landmarks.landmark, w, h)


            # Henüz baseline yoksa: kullanıcıya nötr yüz yapmasını söyleyip birkaç saniye boyunca metrikleri topluyoruz ve ortalamasını baseline olarak kaydediyoruz.
            if baseline is None:
                if calibration_start is None:
                    calibration_start = time.time()

                # İşlemin kaç saniyede yapıldığı
                calibration_samples.append(metrics)
                elapsed = time.time() - calibration_start
                remaining = max(0.0, CALIBRATION_SECONDS - elapsed)

                cv2.putText(frame, "Lutfen notr yuz yapin ve sabit durun...",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                cv2.putText(frame, f"Kalibrasyon: {remaining:.1f} sn",
                            (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

                if elapsed >= CALIBRATION_SECONDS:
                    # Toplanan örneklerin her bir metrik için ortalamasını al
                    keys = calibration_samples[0].keys()
                    baseline = {
                        k: float(numpy.mean([s[k] for s in calibration_samples]))
                        for k in keys
                    }
                    print("Kalibrasyon tamamlandi. Baseline (notr yuz) degerleri:", baseline)

            # --- CANLI TESPİT AŞAMASI ---
            else:
                emotion = detect_emotion(metrics, baseline)

                # Duygu adını ekrana yazdırma
                cv2.putText(frame, f"Duygu: {emotion}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Ham oranları da ekranda göstererek eşikleri gözlemleyip ince ayar
                # yapabilmeyi kolaylaştırıyoruz (istersen bu satırı silebilirsin).
                debug_text = (f"eye:{metrics['eye']:.3f} brow:{metrics['brow']:.3f} "
                              f"mopen:{metrics['mopen']:.3f} mwidth:{metrics['mwidth']:.3f} "
                              f"drop:{metrics['drop']:.3f}")
                cv2.putText(frame, debug_text, (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

            # Yüz Mesh Noktalarını Çizdirme
            mp.solutions.drawing_utils.draw_landmarks(
                image = frame,
                landmark_list = face_landmarks,
                connections = mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = None,
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1,
                                                                               circle_radius=1)
            )



    cv2.imshow("Canlı mimik ve duygu takibi",frame)

    # Klavyeden 'r' tuşuna basılırsa kalibrasyon sıfırlanır, tekrar nötr yüz istenir
    key = cv2.waitKey(60) & 0xFF
    if key == ord("r"):
        baseline = None
        calibration_samples = []
        calibration_start = None

    # Klavyeden 'q' tuşuna basıldığında döngüden çıkılır
    if key == ord("q"):
        break



# Kamera kaynağını serbest bırakıyoruz ve açılan OpenCV pencerelerini kapatma
cap.release()
cv2.destroyAllWindows()