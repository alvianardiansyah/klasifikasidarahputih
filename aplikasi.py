import cv2
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie
import requests
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import io

# Set page config
st.set_page_config(
    page_title="Klasifikasi Sel Darah Putih",
    page_icon="🩸",
    layout="wide"
)

def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Apply dark theme
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    .css-1d391kg {
        background-color: #1E2127;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
        background-color: #0E1117;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E2127;
        color: white;
        padding: 10px 20px;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E3137;
    }
    .css-1outpf7 {
        background-color: #1E2127;
        border-radius: 4px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .stMarkdown {
        color: white;
    }
    .upload-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

def preprocess_image(image_array):
    # Get image dimensions
    height, width = image_array.shape[:2]
    
    # Calculate center ROI coordinates
    roi_size = min(width, height) // 2
    center_x = width // 2
    center_y = height // 2
    
    # Calculate ROI coordinates
    x = center_x - roi_size // 2
    y = center_y - roi_size // 2
    
    # Create fixed ROI mask
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (x, y), (x + roi_size, y + roi_size), 255, -1)
    
    # Convert to HSV for color segmentation
    hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
    
    # Define range for purple/blue color
    lower_purple = np.array([120, 30, 30])
    upper_purple = np.array([180, 255, 255])
    
    # Create WBC mask
    wbc_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # Apply ROI mask
    wbc_mask = cv2.bitwise_and(wbc_mask, wbc_mask, mask=mask)
    
    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    wbc_mask = cv2.morphologyEx(wbc_mask, cv2.MORPH_CLOSE, kernel)
    wbc_mask = cv2.morphologyEx(wbc_mask, cv2.MORPH_OPEN, kernel)
    
    # Draw bounding box on original image
    image_with_box = image_array.copy()
    cv2.rectangle(image_with_box, (x, y), (x + roi_size, y + roi_size), (0, 255, 0), 2)
    
    return image_with_box, wbc_mask, (x, y, roi_size, roi_size)

def get_segmentation_steps(image_array):
    # Convert to HSV
    hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
    
    # Enhance contrast
    contrast_enhanced = cv2.convertScaleAbs(image_array, alpha=1.2, beta=0)
    
    # Create initial mask
    lower_purple = np.array([120, 30, 30])
    upper_purple = np.array([180, 255, 255])
    initial_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # Create final mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    return {
        'original': image_array,
        'contrast': contrast_enhanced,
        'initial_mask': initial_mask,
        'final_mask': final_mask
    }

def extract_features(segmented_image, roi_coords):
    if roi_coords is None:
        return []
    
    x, y, w, h = roi_coords
    roi = segmented_image[y:y+h, x:x+w]
    
    # Find contours in ROI
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

def fuzzy_classification(features):
    # Define fuzzy variables
    area = ctrl.Antecedent(np.arange(0, 50000, 1), 'area')
    perimeter = ctrl.Antecedent(np.arange(0, 1000, 1), 'perimeter')
    eccentricity = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'eccentricity')
    cell_type = ctrl.Consequent(np.arange(0, 5, 1), 'cell_type')

    # Membership functions
    area['small'] = fuzz.trimf(area.universe, [0, 1000, 2000])
    area['medium'] = fuzz.trimf(area.universe, [1500, 3000, 6000])
    area['large'] = fuzz.trimf(area.universe, [5000, 15000, 50000])

    perimeter['short'] = fuzz.trimf(perimeter.universe, [0, 50, 150])
    perimeter['medium'] = fuzz.trimf(perimeter.universe, [100, 250, 450])
    perimeter['long'] = fuzz.trimf(perimeter.universe, [400, 700, 1000])

    eccentricity['low'] = fuzz.trimf(eccentricity.universe, [0, 0.2, 0.4])
    eccentricity['medium'] = fuzz.trimf(eccentricity.universe, [0.3, 0.5, 0.7])
    eccentricity['high'] = fuzz.trimf(eccentricity.universe, [0.6, 0.8, 1])

    cell_type['lymphocyte'] = fuzz.trimf(cell_type.universe, [0, 0, 1])
    cell_type['monocyte'] = fuzz.trimf(cell_type.universe, [0.5, 1.5, 2.5])
    cell_type['neutrophil'] = fuzz.trimf(cell_type.universe, [1.5, 2.5, 3.5])
    cell_type['eosinophil'] = fuzz.trimf(cell_type.universe, [2.5, 3.5, 4])
    cell_type['basophil'] = fuzz.trimf(cell_type.universe, [3, 4, 4.5])

    # Define rules
    rules = [
        ctrl.Rule(area['small'] & eccentricity['low'], cell_type['lymphocyte']),
        ctrl.Rule(area['large'] & eccentricity['medium'], cell_type['monocyte']),
        ctrl.Rule(perimeter['medium'] & eccentricity['high'], cell_type['neutrophil']),
        ctrl.Rule(area['medium'] & eccentricity['medium'], cell_type['eosinophil']),
        ctrl.Rule(perimeter['long'] & eccentricity['low'], cell_type['basophil']),
        ctrl.Rule(area['medium'] & eccentricity['high'], cell_type['neutrophil']),
        ctrl.Rule(area['medium'] & eccentricity['low'], cell_type['lymphocyte']),
        ctrl.Rule(perimeter['short'] & eccentricity['low'], cell_type['lymphocyte']),
        ctrl.Rule(area['large'] & perimeter['long'], cell_type['monocyte']),
        ctrl.Rule(perimeter['medium'] & eccentricity['medium'], cell_type['monocyte']),
        ctrl.Rule(perimeter['long'] & eccentricity['high'], cell_type['neutrophil']),
        ctrl.Rule(area['medium'] & perimeter['long'], cell_type['neutrophil']),
        ctrl.Rule(area['medium'] & perimeter['medium'] & eccentricity['medium'], cell_type['eosinophil']),
        ctrl.Rule(perimeter['medium'] & eccentricity['high'], cell_type['eosinophil']),
        ctrl.Rule(area['medium'] & perimeter['medium'] & eccentricity['medium'], cell_type['basophil']),
        ctrl.Rule(area['medium'] & eccentricity['medium'], cell_type['basophil']),
        ctrl.Rule(area['small'] & perimeter['short'] & eccentricity['low'], cell_type['lymphocyte']),
        ctrl.Rule(area['small'] & perimeter['medium'] & eccentricity['low'], cell_type['lymphocyte']),
        ctrl.Rule(area['large'] & perimeter['medium'] & eccentricity['medium'], cell_type['monocyte']),
        ctrl.Rule(area['large'] & perimeter['long'] & eccentricity['high'], cell_type['monocyte']),
        ctrl.Rule(area['medium'] & perimeter['long'] & eccentricity['high'], cell_type['neutrophil']),
        ctrl.Rule(area['large'] & perimeter['long'] & eccentricity['high'], cell_type['neutrophil']),
        ctrl.Rule(area['medium'] & perimeter['short'] & eccentricity['medium'], cell_type['eosinophil']),
        ctrl.Rule(area['medium'] & perimeter['long'] & eccentricity['medium'], cell_type['eosinophil']),
        ctrl.Rule(area['medium'] & perimeter['short'] & eccentricity['low'], cell_type['basophil']),
        ctrl.Rule(area['large'] & perimeter['medium'] & eccentricity['low'], cell_type['basophil']),
        # Improved basophil rules based on typical characteristics
        ctrl.Rule(area['small'] & perimeter['short'] & eccentricity['low'], cell_type['basophil']),
        ctrl.Rule(area['small'] & perimeter['medium'] & eccentricity['low'], cell_type['basophil']),
        ctrl.Rule(area['medium'] & perimeter['short'] & eccentricity['low'], cell_type['basophil']),
        ctrl.Rule(area['small'] & perimeter['short'] & eccentricity['medium'], cell_type['basophil']),
        ctrl.Rule((area['small'] | area['medium']) & perimeter['short'] & eccentricity['low'], cell_type['basophil']),
        ctrl.Rule(area['small'] & (perimeter['short'] | perimeter['medium']) & eccentricity['low'], cell_type['basophil']),
    ]

    # Create control system
    cell_ctrl = ctrl.ControlSystem(rules)
    cell_sim = ctrl.ControlSystemSimulation(cell_ctrl)

    if not features:
        return None, "Tidak dapat mendeteksi sel darah putih"

    try:
        feature = features[0]
        area_value, perimeter_value, eccentricity_value = feature

        cell_sim.input['area'] = area_value
        cell_sim.input['perimeter'] = perimeter_value
        cell_sim.input['eccentricity'] = eccentricity_value

        cell_sim.compute()
        score = cell_sim.output['cell_type']

        # Classification based on score
        if score < 1.0:
            prediction = 'Lymphocyte'
        elif score < 1.5:
            prediction = 'Monocyte'
        elif score < 2.5:
            prediction = 'Neutrophil'
        elif score < 3.5:
            prediction = 'Eosinophil'
        else:
            prediction = 'Basophil'

        return feature, f"{prediction} (Skor: {score:.2f})"

    except Exception as e:
        return None, f"Error dalam klasifikasi: {str(e)}"

def get_classification_interpretation(cell_type):
    interpretations = {
        'Lymphocyte': "Sel dengan ukuran kecil dan bentuk bulat",
        'Monocyte': "Sel berukuran besar dengan bentuk tidak beraturan",
        'Neutrophil': "Sel dengan inti tersegmentasi",
        'Eosinophil': "Sel dengan granula merah-oranye",
        'Basophil': "Sel dengan granula biru-ungu yang menutupi inti"
    }
    return interpretations.get(cell_type, "Interpretasi tidak tersedia")

def export_results(features, classification):
    df = pd.DataFrame({
        'Fitur': ['Area', 'Perimeter', 'Eccentricity', 'Klasifikasi'],
        'Nilai': [features[0], features[1], features[2], classification]
    })
    return df.to_csv(index=False).encode('utf-8')

def create_feature_visualizations(features):
    if not features or len(features[0]) != 3:
        return None
    
    feature = features[0]
    
    # Create normalized data for visualization
    max_area = 50000  # Maximum expected area
    max_perimeter = 1000  # Maximum expected perimeter
    normalized_data = {
        'Area': feature[0] / max_area,
        'Perimeter': feature[1] / max_perimeter,
        'Eccentricity': feature[2]  # Already between 0 and 1
    }
    
    # Create feature comparison chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(normalized_data.keys(), normalized_data.values())
    ax.set_title('Normalized Feature Values', pad=20)
    ax.set_ylim(0, 1)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # Customize appearance
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#1E2127')
    ax.set_facecolor('#2E3137')
    
    return fig

def main():
    # Main Streamlit UI
    st.markdown("<h1 style='text-align: center;'>🩸 Klasifikasi Sel Darah Putih</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>🧪 Dengan Segmentasi Citra dan FIS Mamdani</h5>", unsafe_allow_html=True)
    
    # Create two columns for the upload section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Load and display Lottie animation
        animation = load_lottieurl("https://lottie.host/552b60ef-801a-47e9-a5a9-15a5904a3242/czf6w9u6Y0.json")
        if animation:
            st_lottie(animation, height=200, key="medical_animation")
    
    with col2:
        st.markdown("### Unggah Gambar Sel Darah Putih")
        uploaded_file = st.file_uploader("Drag and drop file here", type=['jpg', 'jpeg', 'png'])
        st.markdown("<small>Limit 200MB per file • JPG, JPEG, PNG</small>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Process Image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        image_with_box, segmented_image, roi_coords = preprocess_image(image_np)
        features = extract_features(segmented_image, roi_coords)
        
        # Segmentation Section
        st.header("🔍 Tahapan Segmentasi")
        steps = get_segmentation_steps(image_np)
        
        col1, col2, col3, col4 = st.columns(4)
        columns = [col1, col2, col3, col4]
        titles = ["Gambar Asli", "Peningkatan Kontras", "Mask Awal", "Mask Akhir"]
        images = [steps['original'], steps['contrast'], steps['initial_mask'], steps['final_mask']]
        
        for col, title, img in zip(columns, titles, images):
            with col:
                st.markdown(f"**{title}**")
                st.image(img)
        
        # Detection Result and Final Segmentation Side by Side
        st.header("📸 Hasil Akhir")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Hasil Deteksi ROI**")
            st.image(image_with_box, use_column_width=True)
        
        with col2:
            st.markdown("**Hasil Segmentasi Akhir**")
            st.image(segmented_image, use_column_width=True)
        
        # Classification Section
        if features:
            feature, classification_result = fuzzy_classification(features)
            
            if feature is not None:
                st.header("📊 Hasil Klasifikasi")
                
                # Display classification result
                cell_type = classification_result.split()[0]
                st.success(f"### {classification_result}")
                st.info(f"**Karakteristik {cell_type}:** {get_classification_interpretation(cell_type)}")
                
                # Feature Details Tabs
                tab_fitur, tab_vis = st.tabs(["📐 Fitur Bentuk", "📈 Visualisasi"])
                
                with tab_fitur:
                    col1, col2, col3 = st.columns(3)
                    metrics = [
                        ("Area", feature[0], "px²"),
                        ("Perimeter", feature[1], "px"),
                        ("Eccentricity", feature[2], "")
                    ]
                    
                    for col, (label, value, unit) in zip([col1, col2, col3], metrics):
                        with col:
                            st.metric(
                                label, 
                                f"{value:.2f}{unit}" if not np.isnan(value) else "N/A"
                            )
                    
                    st.markdown("""
                    **Interpretasi Fitur Bentuk:**
                    * Area: Ukuran total sel dalam piksel persegi
                    * Perimeter: Panjang tepi sel dalam piksel
                    * Eccentricity: Ukuran kebulatan sel (0 = lingkaran sempurna, 1 = garis)
                    """)
                
                with tab_vis:
                    st.markdown("### Visualisasi Fitur")
                    # Create bar chart for shape features
                    shape_features = {
                        'Area': feature[0] / max(feature[0], 1),
                        'Perimeter': feature[1] / max(feature[1], 1),
                        'Eccentricity': feature[2]
                    }
                    
                    fig = plt.figure(figsize=(10, 6))
                    plt.style.use('dark_background')
                    plt.bar(shape_features.keys(), shape_features.values())
                    plt.title('Normalized Shape Features')
                    plt.ylim(0, 1)
                    st.pyplot(fig)
                
                # Export Section
                st.subheader("💾 Ekspor Hasil")
                csv = export_results(feature, classification_result)
                st.download_button(
                    "📥 Unduh hasil analisis (CSV)",
                    data=csv,
                    file_name="hasil_klasifikasi_wbc.csv",
                    mime="text/csv"
                )
                
                # Methodology
                with st.expander("ℹ️ Metodologi"):
                    st.markdown("""
                    **Proses Klasifikasi:**
                    1. Preprocessing citra untuk isolasi sel darah putih
                    2. Ekstraksi fitur geometri (area, perimeter, eccentricity)
                    3. Klasifikasi menggunakan Fuzzy Inference System
                    4. Validasi hasil berdasarkan karakteristik sel
                    """)

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🧬 Sistem Klasifikasi Sel Darah Putih menggunakan Fuzzy Logic dan Image Processing</p>
    <p>Created with ❤️ by Kelompok 3</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
