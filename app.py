import os
import torch
import clip
import streamlit as st
from PIL import Image
import faiss

from retrieval import generate_caption, ImageRetrieval

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="CLIP Semantic Search",
    page_icon="🔍",
    layout="wide"
)

# =========================
# Header
# =========================
st.markdown(
    """
    <h1 style="text-align:center;">🔍 CLIP Semantic Search</h1>
    <p style="text-align:center; color:gray;">
    Bidirectional Text ↔ Image Retrieval using CLIP + Vector Database
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# =========================
# Device
# =========================
device = "cpu"

# =========================
# Load CLIP Model
# =========================
@st.cache_resource
def load_model():
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

clip_model, clip_preprocess = load_model()

# =========================
# Load Data
# =========================
@st.cache_resource
def load_data():
    captions_features = torch.load(
        "embeddings/captions_features_new.pt", map_location="cpu"
    ).float()

    processed_captions = torch.load(
        "embeddings/processed_captions_new.pt", map_location="cpu"
    )

    img_names = torch.load(
        "embeddings/img_names_new.pt", map_location="cpu"
    )

    faiss_index = faiss.read_index("embeddings/image_index.faiss")

    return faiss_index, captions_features, processed_captions, img_names


faiss_index, captions_features, processed_captions, img_names = load_data()

# =========================
# Retrieval Engine
# =========================
retrieval_engine = ImageRetrieval(
    clip_model=clip_model,
    faiss_index=faiss_index,
    device=device
)

# =========================
# Image Helper
# =========================
def load_and_resize_image(path, size=(256, 256)):
    img = Image.open(path).convert("RGB")
    img.thumbnail(size)
    return img

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ Controls")

    mode = st.radio(
        "Choose Mode",
        ["📝 Text → Image Search", "🖼 Image → Caption Search"]
    )

    top_k = st.slider(
        "Number of results",
        1, 5, 3
    )

    st.markdown("---")
    st.caption("CLIP • FAISS • Streamlit")

# =========================
# TEXT → IMAGE SEARCH
# =========================
if mode == "📝 Text → Image Search":
    st.subheader("📝 Text → Image Semantic Search")

    with st.container():
        query = st.text_input(
            "Describe what you are looking for",
            placeholder="e.g. a dog running in a park"
        )

        search_clicked = st.button("🔍 Search")

    if search_clicked and query:
        with st.spinner("Searching semantic space..."):
            indices = retrieval_engine.retrieve(query, k=top_k)

        st.success(f"Found {len(indices)} matching images")

        st.markdown("### 📸 Results")

        cols = st.columns(top_k)
        shown = 0
        for i, idx in enumerate(indices):
            if shown >= top_k:
                break
            img_path = os.path.join("data/images_subset", img_names[idx])
            if os.path.exists(img_path):
                img = load_and_resize_image(img_path)
                cols[i].image(img, use_container_width=True)
                shown+=1


# =========================
# IMAGE → CAPTION SEARCH
# =========================
elif mode == "🖼 Image → Caption Search":
    st.subheader("🖼 Image → Caption Retrieval")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Uploaded Image**")
            st.image(img, use_container_width=True)

        with col2:
            st.markdown("**Predicted Caption**")

            if st.button("📝 Generate Caption"):
                with st.spinner("Analyzing image..."):
                    caption = generate_caption(
                        img,
                        clip_model,
                        clip_preprocess
                    )

                st.success(caption)

# =========================
# Footer
# =========================
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:gray; font-size:13px;">
    Built using CLIP embeddings • FAISS vector search • Streamlit UI
    </p>
    """,
    unsafe_allow_html=True
)

