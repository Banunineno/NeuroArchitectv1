import streamlit as st
import cv2
import torch
import numpy as np
import gc
import os
import io
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# --- KONFIGURASI JUDUL & DIREKTORI ---
SAVE_DIR = "results_realistic"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

st.set_page_config(page_title="NeuroArchitect: Sketch-to-Color Reality", layout="wide")
st.title("🏙️ NeuroArchitect: Sketch-to-Color Reality")
st.markdown("### Transformasi Sketsa Hitam Putih Menjadi Foto Arsitektur Realistik")

# --- SISTEM MEMORI & GALLERY ---
if 'gallery' not in st.session_state:
    st.session_state.gallery = []
if 'last_render' not in st.session_state:
    st.session_state.last_render = None

# --- LOAD MODEL (Realistic Vision v6.0) ---
@st.cache_resource
def load_models():
    # Model ControlNet tetap menggunakan Canny
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", 
        torch_dtype=torch.float16
    )
    
    # MENGGUNAKAN MODEL REALISTIC VISION V6.0 (Lebih Nyata)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V6.0_B1_noVAE", 
        controlnet=controlnet, 
        torch_dtype=torch.float16,
        safety_checker=None
    )
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Optimasi Khusus GPU 4GB & RAM 24GB
    pipe.enable_attention_slicing() 
    pipe.enable_model_cpu_offload() 
    
    return pipe

def get_canny_map(image):
    img = np.array(image.convert("RGBA").convert("RGB"))
    canny = cv2.Canny(img, 100, 200)
    canny = canny[:, :, None]
    canny = np.concatenate([canny, canny, canny], axis=2)
    return Image.fromarray(canny)

with st.spinner("Menghubungkan ke Model Realistic Vision v6.0... (Mohon tunggu jika ada download baru)"):
    pipe = load_models()

# --- SIDEBAR: PENGATURAN ---
st.sidebar.header("⚙️ Kontrol Realisme")
style = st.sidebar.selectbox("Gaya Bangunan", 
    ["Modern Glass Mansion", "Classic Red Brick Loft", "Japanese Timber House", "Tropical Stone Villa"])

steps = st.sidebar.slider("Detail Realisme (Steps)", 20, 50, 30)

# PROMPT KHUSUS REALISTIC VISION
prompt_map = {
    "Modern Glass Mansion": "high-end modern villa, full height glass windows with reflections, white concrete, swimming pool, luxury real estate photography, 8k uhd, photorealistic",
    "Classic Red Brick Loft": "authentic industrial loft, aged red brick walls, black iron window frames, urban street setting, detailed textures, professional architectural shot, 8k",
    "Japanese Timber House": "contemporary japanese architecture, light cedar wood, minimalist design, garden with gravel, soft natural daylight, high resolution photo",
    "Tropical Stone Villa": "luxury tropical architecture, volcanic stone walls, teak wood accents, palm tree shadows, sunset warm lighting, cinematic architectural photography"
}

# Negative Prompt agar TIDAK JADI KARTUN
neg_prompt = "cartoon, anime, sketch, drawing, painting, blurry, low quality, distorted, monochrome, dark, foggy, text, watermark"

# --- LAYOUT UTAMA ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Input Sketsa")
    file = st.file_uploader("Upload sketsa garis Anda", type=["jpg", "png"])
    if file:
        input_img = Image.open(file)
        st.image(input_img, caption="Sketsa Asli", width="stretch")
        canny_processed = get_canny_map(input_img)

with col2:
    st.subheader("✨ Hasil Foto Realistik")
    if file and st.button("🚀 MULAI RENDER"):
        with st.spinner("AI sedang menghidupkan sketsa Anda..."):
            try:
                result = pipe(
                    prompt=prompt_map[style],
                    negative_prompt=neg_prompt,
                    image=canny_processed,
                    num_inference_steps=steps,
                    controlnet_conditioning_scale=1.0,
                ).images[0]
                
                st.session_state.last_render = result
                st.session_state.gallery.append({"image": result, "style": style})
                
                # Simpan ke folder E:
                save_path = os.path.join(SAVE_DIR, f"render_{len(st.session_state.gallery)}.png")
                result.save(save_path)
                
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.last_render:
        st.image(st.session_state.last_render, caption="Hasil Render Terbaru", width="stretch")
        buf = io.BytesIO()
        st.session_state.last_render.save(buf, format="PNG")
        st.download_button("💾 Download Hasil", data=buf.getvalue(), file_name="render_realistik.png")

# --- BAGIAN GALLERY (BARU) ---
if st.session_state.gallery:
    st.divider()
    st.subheader("🖼️ Riwayat Render (Gallery)")
    cols = st.columns(4)
    for i, item in enumerate(reversed(st.session_state.gallery)):
        with cols[i % 4]:
            st.image(item["image"], caption=f"Style: {item['style']}", width="stretch")