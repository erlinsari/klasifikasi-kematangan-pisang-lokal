import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import os
import h5py

# Konfigurasi Halaman
st.set_page_config(
    page_title="Sistem Deteksi Kematangan Pisang",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS: Clean, White, Symmetrical, Centered Navbar
# Custom CSS: Clean, Symmetrical, Pastel Yellow Theme
css_kustom = """
<style>
    @import url('https://googleapis.com');

    html, body, [class*="css"]  {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #FEFBEA; /* Warna Background Kuning Pastel */
        color: #2F3E46;
    }
    
    .stApp {
        background-color: #FEFBEA; /* Warna Background Utama Aplikasi Kuning Pastel */
    }

    /* TABS NAVBAR STYLING - CENTERED WITH LEAF GREEN ACCENT */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        gap: 30px;
        border-bottom: 2px solid #EAE5C9;
        padding-bottom: 10px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0px;
        padding: 10px 15px;
        color: #7A7555;
        font-weight: 600;
        font-size: 1.1rem;
        border: none !important;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        color: #2E7D32 !important; /* Teks Tab aktif warna Hijau Daun */
        border-bottom: 3px solid #E6A100 !important; /* Garis bawah tab aktif warna Kuning Gelap */
        background-color: transparent !important;
    }

    /* Card styling - Putih bersih agar kontras di atas Latar Belakang Kuning Pastel */
    .metric-card {
        background-color: #FFFFFF;
        color: #2F3E46 !important;
        border: 1px solid #EAE5C9;
        border-radius: 16px;
        padding: 32px 24px;
        box-shadow: 0 4px 20px rgba(230, 161, 0, 0.05);
        text-align: center;
        transition: transform 0.2s ease-in-out;
    }
    
    .info-card {
        background-color: #FFFFFF;
        color: #2F3E46 !important;
        border: 1px solid #EAE5C9;
        border-left: 4px solid #2E7D32; /* Aksen garis kiri Hijau */
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 2px 12px rgba(230, 161, 0, 0.03);
    }

    h1, h2, h3, h4 {
        color: #1A3038 !important;
        font-weight: 700 !important;
    }

    /* Gradient Button: Green Leaf Theme */
    .stButton > button {
        background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%);
        color: white;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #1B5E20 0%, #0F3813 100%);
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.3);
        color: white;
    }
    
    /* Soft aesthetic status colors */
    .status-mentah {
        color: #1B5E20;
        background-color: #E8F5E9;
        padding: 6px 20px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #C8E6C9;
    }
    
    .status-matang {
        color: #E65100;
        background-color: #FFF3E0;
        padding: 6px 20px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #FFE0B2;
    }

    .status-terlalu {
        color: #B71C1C;
        background-color: #FFEBEE;
        padding: 6px 20px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #FFCDD2;
    }
    
    hr {
        border-color: #EAE5C9;
        margin: 2rem 0;
    }
    
    p, li {
        font-size: 1.05rem;
        line-height: 1.7;
        color: #4A5759;
    }
</style>
"""
st.markdown(css_kustom, unsafe_allow_html=True)

# Konstanta
UKURAN_INPUT = (224, 224)
CLASS_NAMES = ['matang', 'mentah', 'terlalu_matang']


# --- FUNGSI PEMBUATAN ARSITEKTUR MODEL ---
def inject_dense_weights(model, h5_path, expected_in):
    kernel_1 = None
    bias_1 = None
    kernel_2 = None
    bias_2 = None
    
    with h5py.File(h5_path, 'r') as f:
        g = f['model_weights'] if 'model_weights' in f else f
        
        def search_group(group):
            nonlocal kernel_1, bias_1, kernel_2, bias_2
            for k in group.keys():
                item = group[k]
                if isinstance(item, h5py.Group):
                    search_group(item)
                elif isinstance(item, h5py.Dataset):
                    if item.shape == (expected_in, 128):
                        kernel_1 = item[:]
                        for sub_k in group.keys():
                            if group[sub_k].shape == (128,):
                                bias_1 = group[sub_k][:]
                                break
                    elif item.shape == (128, 3):
                        kernel_2 = item[:]
                        for sub_k in group.keys():
                            if group[sub_k].shape == (3,):
                                bias_2 = group[sub_k][:]
                                break
        
        search_group(g)
        
    if kernel_1 is not None and bias_1 is not None:
        model.layers[-3].set_weights([kernel_1, bias_1])
    if kernel_2 is not None and bias_2 is not None:
        model.layers[-1].set_weights([kernel_2, bias_2])

def get_augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.1)
    ], name="Augmentasi_On_The_Fly")

@st.cache_resource
def load_mobilenet_v3():
    try:
        data_augmentation = get_augmentation_layer()
        base_model = tf.keras.applications.MobileNetV3Large(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = data_augmentation(inputs)
        x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d_1')(x)
        x = tf.keras.layers.Dense(128, activation='relu', name='dense_2')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name="MobileNetV3_Large")
        inject_dense_weights(model, "models/best_mobilenet_pisang.h5", 960)
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_convnext_tiny():
    try:
        data_augmentation = get_augmentation_layer()
        base_model = tf.keras.applications.ConvNeXtTiny(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = data_augmentation(inputs)
        x = tf.keras.applications.convnext.preprocess_input(x)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d')(x)
        x = tf.keras.layers.Dense(128, activation='relu', name='dense')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name="ConvNeXt_Tiny")
        inject_dense_weights(model, "models/best_convnext_pisang.h5", 768)
        return model, None
    except Exception as e:
        return None, str(e)

def preprocess_image(image):
    img = image.resize(UKURAN_INPUT)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)
    return img_array

def format_hasil(class_name, confidence):
    label = class_name.replace('_', ' ').title()
    if class_name == 'mentah':
        return label, confidence, 'status-mentah'
    elif class_name == 'matang':
        return label, confidence, 'status-matang'
    else:
        return label, confidence, 'status-terlalu'

def tampilkan_gambar_visualisasi(nama_file, deskripsi):
    path_visual = f"Visualisasi/{nama_file}"
    if os.path.exists(path_visual):
        gambar = Image.open(path_visual)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(gambar, caption=deskripsi, use_container_width=True)
    else:
        st.info(f"Visualisasi {nama_file} belum tersedia.")

# HEADER
st.markdown("<h1 style='text-align: center; margin-top: 1rem;'>Sistem Inspeksi Kematangan Pisang</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #6B7280; margin-bottom: 2rem;'>Sistem Otomatisasi Penilaian Kualitas Berbasis Deep Learning</p>", unsafe_allow_html=True)

# TABS NAVIGATION
tab_prediksi, tab_dataset, tab_convnext, tab_mobilenet, tab_komparasi = st.tabs([
    "Inspeksi Visual", 
    "Metodologi & Dataset", 
    "Analisis ConvNeXt-Tiny", 
    "Analisis MobileNetV3",
    "Komparasi Model"
])

# --- TAB 1: PREDIKSI ---
with tab_prediksi:
    st.markdown("<br>", unsafe_allow_html=True)
    
    model_mob, err_mob = load_mobilenet_v3()
    model_conv, err_conv = load_convnext_tiny()

    if model_mob is None or model_conv is None:
        st.warning("Peringatan: File bobot model belum terpasang dengan benar pada peladen (server).")
    else:
        kolom_kiri, kolom_tengah, kolom_kanan = st.columns([1, 2, 1])

        with kolom_tengah:
            st.markdown("<div class='info-card'><b>Instruksi Operasional:</b> Unggah citra sampel pisang dengan pencahayaan netral dan resolusi yang memadai untuk memperoleh hasil analisis komparatif.</div>", unsafe_allow_html=True)
            berkas_unggah = st.file_uploader("Pilih file citra (Resolusi disarankan: > 500x500px)", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
            
            if berkas_unggah is not None:
                citra_asli = Image.open(berkas_unggah).convert('RGB')
                st.image(citra_asli, caption="Citra Sampel Terunggah", use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                with col_btn2:
                    tombol_prediksi = st.button("Lakukan Inspeksi Paralel", use_container_width=True)

        if berkas_unggah is not None and 'tombol_prediksi' in locals() and tombol_prediksi:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>Laporan Hasil Inspeksi Komparatif</h2>", unsafe_allow_html=True)
            
            with st.spinner("Mengeksekusi jaringan saraf tiruan secara paralel..."):
                img_tensor = preprocess_image(citra_asli)
                
                # ConvNeXt
                start_conv = time.time()
                pred_c = model_conv.predict(img_tensor, verbose=0)
                waktu_conv = time.time() - start_conv
                class_c = CLASS_NAMES[np.argmax(pred_c[0])]
                conf_c = np.max(pred_c[0]) * 100
                
                # MobileNet
                start_mob = time.time()
                pred_m = model_mob.predict(img_tensor, verbose=0)
                waktu_mob = time.time() - start_mob
                class_m = CLASS_NAMES[np.argmax(pred_m[0])]
                conf_m = np.max(pred_m[0]) * 100
                
                kelas_conv, conf_conv, style_conv = format_hasil(class_c, conf_c)
                kelas_mob, conf_mob, style_mob = format_hasil(class_m, conf_m)

                col_conv, col_mob = st.columns(2)
                
                with col_conv:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3 style='color: #374151; font-size: 1.2rem; margin-bottom: 1.5rem;'>Arsitektur ConvNeXt-Tiny</h3>
                        <div style='margin-bottom: 2rem;'>
                            <span class='{style_conv}'>{kelas_conv}</span>
                        </div>
                        <p style='margin: 0; font-size: 0.9rem; color: #6B7280;'>Tingkat Kepercayaan (Confidence)</p>
                        <h2 style='margin: 0 0 1.5rem 0; color: #111827;'>{conf_conv:.2f}%</h2>
                        <div style='background-color: #F3F4F6; padding: 10px; border-radius: 6px; display: inline-block;'>
                            <span style='font-size: 0.85rem; color: #4B5563;'>Waktu Inferensi: <b>{waktu_conv:.3f} detik</b></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_mob:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3 style='color: #374151; font-size: 1.2rem; margin-bottom: 1.5rem;'>Arsitektur MobileNetV3-Large</h3>
                        <div style='margin-bottom: 2rem;'>
                            <span class='{style_mob}'>{kelas_mob}</span>
                        </div>
                        <p style='margin: 0; font-size: 0.9rem; color: #6B7280;'>Tingkat Kepercayaan (Confidence)</p>
                        <h2 style='margin: 0 0 1.5rem 0; color: #111827;'>{conf_mob:.2f}%</h2>
                        <div style='background-color: #F3F4F6; padding: 10px; border-radius: 6px; display: inline-block;'>
                            <span style='font-size: 0.85rem; color: #4B5563;'>Waktu Inferensi: <b>{waktu_mob:.3f} detik</b></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<br><p style='text-align: center; font-size: 0.95rem; color: #6B7280;'>Laporan di atas merupakan evaluasi komparatif antara dua model Deep Learning mutakhir.</p>", unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)
    tampilkan_gambar_visualisasi("komparasi.png", "Perbandingan Metrik Kinerja Akhir (Accuracy, Precision, Recall, F1-Score)")
    st.markdown("</div>", unsafe_allow_html=True)
