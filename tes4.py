import cv2
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie
import requests

# Fungsi untuk memuat animasi Lottie
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Muat animasi Lottie bertema medis
animation_medical_left = load_lottieurl("https://lottie.host/552b60ef-801a-47e9-a5a9-15a5904a3242/czf6w9u6Y0.json")
def preprocess_image(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return image_array, morph

def extract_features(segmented_image):
    contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            eccentricity = minor_axis / major_axis if major_axis > 0 else 0
        else:
            eccentricity = 0
        features.append((area, perimeter, eccentricity))
    return features

# Fungsi untuk klasifikasi menggunakan fuzzy inference system (FIS)
def fuzzy_classification(features):
    # Definisi variabel fuzzy dan membership functions tetap sama
    area = ctrl.Antecedent(np.arange(0, 50000, 1), 'area')
    perimeter = ctrl.Antecedent(np.arange(0, 1000, 1), 'perimeter')
    eccentricity = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'eccentricity')
    cell_type = ctrl.Consequent(np.arange(0, 5, 1), 'cell_type')

    # Membership functions tetap sama
    area['small'] = fuzz.trimf(area.universe, [0, 0, 1000])
    area['medium'] = fuzz.trimf(area.universe, [500, 2000, 5000])
    area['large'] = fuzz.trimf(area.universe, [3000, 10000, 50000])

    perimeter['short'] = fuzz.trimf(perimeter.universe, [0, 0, 100])
    perimeter['medium'] = fuzz.trimf(perimeter.universe, [50, 200, 400])
    perimeter['long'] = fuzz.trimf(perimeter.universe, [300, 600, 1000])

    eccentricity['low'] = fuzz.trimf(eccentricity.universe, [0, 0, 0.5])
    eccentricity['medium'] = fuzz.trimf(eccentricity.universe, [0.3, 0.5, 0.7])
    eccentricity['high'] = fuzz.trimf(eccentricity.universe, [0.6, 1, 1])

    cell_type['lymphocyte'] = fuzz.trimf(cell_type.universe, [0, 0, 1])
    cell_type['monocyte'] = fuzz.trimf(cell_type.universe, [0.5, 1, 2])
    cell_type['neutrophil'] = fuzz.trimf(cell_type.universe, [1.5, 2, 3])
    cell_type['eosinophil'] = fuzz.trimf(cell_type.universe, [2.5, 3, 4])
    cell_type['basophil'] = fuzz.trimf(cell_type.universe, [3.5, 4, 4])

    # Aturan-aturan yang sudah ada
    rule1 = ctrl.Rule(area['small'] & eccentricity['low'], cell_type['lymphocyte'])
    rule2 = ctrl.Rule(area['large'] & eccentricity['medium'], cell_type['monocyte'])
    rule3 = ctrl.Rule(perimeter['medium'] & eccentricity['high'], cell_type['neutrophil'])
    rule4 = ctrl.Rule(area['medium'] & eccentricity['medium'], cell_type['eosinophil'])
    rule5 = ctrl.Rule(perimeter['long'] & eccentricity['low'], cell_type['basophil'])
    rule6 = ctrl.Rule(area['medium'] & eccentricity['high'], cell_type['neutrophil'])
    rule7 = ctrl.Rule(area['medium'] & eccentricity['low'], cell_type['lymphocyte'])
    rule8 = ctrl.Rule(perimeter['short'] & eccentricity['low'], cell_type['lymphocyte'])
    rule9 = ctrl.Rule(area['large'] & perimeter['long'], cell_type['monocyte'])
    rule10 = ctrl.Rule(perimeter['medium'] & eccentricity['medium'], cell_type['monocyte'])
    rule11 = ctrl.Rule(perimeter['long'] & eccentricity['high'], cell_type['neutrophil'])
    rule12 = ctrl.Rule(area['medium'] & perimeter['long'], cell_type['neutrophil'])
    rule13 = ctrl.Rule(area['medium'] & perimeter['medium'] & eccentricity['medium'], cell_type['eosinophil'])
    rule14 = ctrl.Rule(perimeter['medium'] & eccentricity['high'], cell_type['eosinophil'])
    rule15 = ctrl.Rule(area['medium'] & perimeter['medium'] & eccentricity['medium'], cell_type['basophil'])
    rule16 = ctrl.Rule(area['medium'] & eccentricity['medium'], cell_type['basophil'])

    # Aturan-aturan baru
    # Kombinasi tambahan untuk Limfosit
    rule17 = ctrl.Rule(area['small'] & perimeter['short'] & eccentricity['low'], cell_type['lymphocyte'])
    rule18 = ctrl.Rule(area['small'] & perimeter['medium'] & eccentricity['low'], cell_type['lymphocyte'])

    # Kombinasi tambahan untuk Monosit
    rule19 = ctrl.Rule(area['large'] & perimeter['medium'] & eccentricity['medium'], cell_type['monocyte'])
    rule20 = ctrl.Rule(area['large'] & perimeter['long'] & eccentricity['high'], cell_type['monocyte'])

    # Kombinasi tambahan untuk Neutrofil
    rule21 = ctrl.Rule(area['medium'] & perimeter['long'] & eccentricity['high'], cell_type['neutrophil'])
    rule22 = ctrl.Rule(area['large'] & perimeter['long'] & eccentricity['high'], cell_type['neutrophil'])

    # Kombinasi tambahan untuk Eosinofil
    rule23 = ctrl.Rule(area['medium'] & perimeter['short'] & eccentricity['medium'], cell_type['eosinophil'])
    rule24 = ctrl.Rule(area['medium'] & perimeter['long'] & eccentricity['medium'], cell_type['eosinophil'])

    # Kombinasi tambahan untuk Basofil
    rule25 = ctrl.Rule(area['medium'] & perimeter['short'] & eccentricity['low'], cell_type['basophil'])
    rule26 = ctrl.Rule(area['large'] & perimeter['medium'] & eccentricity['low'], cell_type['basophil'])

    # Menggabungkan semua aturan
    cell_ctrl = ctrl.ControlSystem([
        rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, 
        rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16,
        rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24,
        rule25, rule26
    ])
    
    cell_sim = ctrl.ControlSystemSimulation(cell_ctrl)

    # Sisa kode untuk processing dan klasifikasi tetap sama
    results = []
    scores_by_category = {'Lymphocyte': [], 'Monocyte': [], 'Neutrophil': [], 'Eosinophil': [], 'Basophil': []}

    for feature in features:
        area_value, perimeter_value, eccentricity_value = feature

        if not (0 <= area_value <= 50000 and 0 <= perimeter_value <= 1000 and 0 <= eccentricity_value <= 1.1):
            continue

        try:
            cell_sim.input['area'] = area_value
            cell_sim.input['perimeter'] = perimeter_value
            cell_sim.input['eccentricity'] = eccentricity_value

            cell_sim.compute()
            score = cell_sim.output['cell_type']

            if score < 0.5:
                category = 'Lymphocyte'
            elif score < 1.5:
                category = 'Monocyte'
            elif score < 2.5:
                category = 'Neutrophil'
            elif score < 3.5:
                category = 'Eosinophil'
            else:
                category = 'Basophil'

            scores_by_category[category].append(score)
            results.append((feature, f"{category} (Skor: {score:.2f})"))
        except KeyError:
            continue

    avg_scores = {category: np.mean(scores) if scores else 0 for category, scores in scores_by_category.items()}
    dominant_category = max(avg_scores, key=avg_scores.get)

    return results, avg_scores, dominant_category

# UI Streamlit dengan markdown di tengah
st.markdown("<h1 style='text-align: center;'>🩸 Klasifikasi Sel Darah Putih</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>🧪 Dengan Segmentasi Citra dan FIS berbasis Mamdani</h5>", unsafe_allow_html=True)

# Layout dengan tiga kolom
col1, col2, col3 = st.columns([1, 4, 1])


# Animasi di sisi kiri
with col1:
    st_lottie(animation_medical_left, height=200, key="medical_left")

# Menu upload gambar di tengah
with col2:
    uploaded_file = st.file_uploader("📁 Unggah Gambar", type=['jpg', 'jpeg', 'png'])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    original_image, segmented_image = preprocess_image(image_np)
    features = extract_features(segmented_image)
    results, avg_scores, dominant_category = fuzzy_classification(features)

    st.subheader("🖼️ Gambar yang Diunggah")
    st.image(image, caption="Gambar Asli", use_column_width=True)

    st.subheader("🔍 Hasil Segmentasi")
    segmented_image_pil = Image.fromarray(segmented_image)
    st.image(segmented_image_pil, caption="Gambar Tersegmentasi", use_column_width=True)

    st.subheader("📊 Hasil Klasifikasi")
    for feature, category in results:
        st.write(f"Fitur: Area={feature[0]:.2f}, Perimeter={feature[1]:.2f}, Eccentricity={feature[2]:.2f}")
        st.write(f"Kategori Prediksi: {category}")

    st.subheader("📈 Rata-rata Skor per Kategori")
    for category, avg_score in avg_scores.items():
        if avg_score > 0:
            st.write(f"{category}: {avg_score:.2f}")

    st.subheader("🏆 Kategori Dominan")
    st.write(f"Gambar ini kemungkinan besar termasuk kategori **{dominant_category}**.")
