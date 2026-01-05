import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F

import streamlit as st
import tempfile
import numpy as np
import cv2
from PIL import Image
import time

from multimodal_fixed import (
    AudioXceptionClassifier,
    VideoXceptionModel,
    AdaptiveFusionModule,
    load_checkpoint_auto,
    extract_audio_from_video,
    preprocess_audio,
    setup_face_detection,
    detect_face_hybrid,
    video_transform,
    device
)

st.set_page_config(
    page_title="Deepfake Detector",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    * { box-sizing: border-box; }
    
    html, body, .stApp {
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }
    
    .stApp {
        background: linear-gradient(160deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%) !important;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1300px;
        overflow-x: hidden !important;
    }
    
    /* Prevent layout shifts */
    .main, .block-container, [data-testid="stVerticalBlock"] {
        overflow-x: hidden !important;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, label, div, .stMarkdown {
        color: #e4e4e7 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    /* HEADER - CLEAN GRADIENT */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 50px 40px;
        text-align: center;
        margin-bottom: 36px;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.25);
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: white !important;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-subtitle {
        font-size: 1rem;
        color: rgba(255,255,255,0.85) !important;
        margin-top: 10px;
        font-weight: 400;
    }
    
    /* SECTION TITLE */
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #a78bfa !important;
        margin-bottom: 20px;
    }
    
    /* FILE UPLOADER */
    .stFileUploader { margin-bottom: 20px; }
    .stFileUploader label { display: none !important; }
    .stFileUploader * { color: #e4e4e7 !important; }
    .stFileUploader small { color: #71717a !important; }
    
    div[data-testid="stFileUploaderFileName"] { color: #a78bfa !important; }
    div[data-testid="stFileUploaderFileSize"] { color: #71717a !important; }
    
    /* VIDEO */
    .stVideo {
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 20px;
    }
    
    .stVideo video {
        width: 100% !important;
        max-height: 350px !important;
        object-fit: contain !important;
        background: #0f0f1a;
        border-radius: 12px;
    }
    
    /* RADIO */
    .stRadio > div {
        background: rgba(39, 39, 42, 0.5);
        padding: 10px 16px;
        border-radius: 10px;
    }
    
    .stRadio label { color: #e4e4e7 !important; }
    .stRadio > label { color: #a78bfa !important; font-weight: 500 !important; }
    
    /* RESULT BOX */
    .result-box {
        background: rgba(39, 39, 42, 0.6);
        border-radius: 16px;
        padding: 40px 24px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .result-box-real {
        border: 2px solid #10b981;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.15);
    }
    
    .result-box-fake {
        border: 2px solid #ef4444;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.15);
    }
    
    .verdict-icon { font-size: 4rem; margin-bottom: 12px; }
    
    .verdict-real {
        color: #10b981 !important;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .verdict-fake {
        color: #ef4444 !important;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .confidence-label {
        color: #71717a !important;
        font-size: 1rem;
        margin-top: 8px;
    }
    
    /* MODALITY ANALYSIS */
    .modality-section {
        background: rgba(39, 39, 42, 0.4);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .modality-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e4e4e7 !important;
        margin-bottom: 10px;
    }
    
    .modality-bar-container {
        width: 100%;
    }
    
    .modality-bar-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
        font-size: 0.85rem;
    }
    
    .label-real {
        color: #10b981 !important;
        font-weight: 600;
    }
    
    .label-fake {
        color: #ef4444 !important;
        font-weight: 600;
    }
    
    .modality-bar {
        display: flex;
        height: 12px;
        border-radius: 6px;
        overflow: hidden;
        background: rgba(39, 39, 42, 0.6);
    }
    
    .bar-real {
        background: linear-gradient(90deg, #10b981, #34d399);
        height: 100%;
    }
    
    .bar-fake {
        background: linear-gradient(90deg, #ef4444, #f87171);
        height: 100%;
    }
    
    .analysis-info {
        display: flex;
        gap: 12px;
        margin-top: 12px;
    }
    
    .info-item {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 12px;
        background: rgba(39, 39, 42, 0.4);
        border-radius: 10px;
    }
    
    .info-icon {
        font-size: 1.1rem;
    }
    
    .info-text {
        color: #a1a1aa !important;
        font-size: 0.85rem;
    }
    
    /* WEIGHT BARS */
    .weight-section { margin: 20px 0; }
    .weight-item { margin-bottom: 14px; }
    
    .weight-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    
    .weight-name { font-size: 0.9rem; color: #e4e4e7 !important; }
    .weight-value { font-weight: 600; color: #a78bfa !important; }
    
    .progress-bg {
        background: rgba(39, 39, 42, 0.8);
        border-radius: 6px;
        height: 8px;
        overflow: hidden;
    }
    
    .progress-video {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 6px;
    }
    
    .progress-audio {
        height: 100%;
        background: linear-gradient(90deg, #a78bfa, #c084fc);
        border-radius: 6px;
    }
    
    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 14px 28px !important;
        font-size: 0.95rem !important;
        width: 100%;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* CONFIG */
    .stSelectbox label, .stSlider label {
        color: #a1a1aa !important;
        font-weight: 500 !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(39, 39, 42, 0.8) !important;
        border: 1px solid rgba(113, 113, 122, 0.3) !important;
        border-radius: 8px !important;
    }
    
    /* MESSAGES */
    .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        color: #10b981 !important;
        border-radius: 10px !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        color: #ef4444 !important;
        border-radius: 10px !important;
    }
    
    .stInfo {
        background: rgba(167, 139, 250, 0.1) !important;
        color: #a78bfa !important;
        border-radius: 10px !important;
    }
    
    /* EMPTY STATE */
    .empty-state {
        text-align: center;
        padding: 60px 30px;
    }
    
    .empty-icon {
        font-size: 4rem;
        margin-bottom: 16px;
        opacity: 0.5;
    }
    
    .empty-text {
        color: #71717a !important;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(113, 113, 122, 0.3), transparent);
        margin: 20px 0;
    }
    
    /* SAMPLE FRAME */
    .stImage {
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 16px;
    }
    
    .stImage img {
        border-radius: 12px;
    }
    
    .stImage img {
        border-radius: 16px;
        border: 2px solid rgba(99, 102, 241, 0.3);
    }
</style>
""", unsafe_allow_html=True)

if 'config' not in st.session_state:
    st.session_state.config = {
        'fusion_strategy': 'equal',
        'audio_temperature': 2.0,
        'decision_threshold': 0.5,
        'frame_skip': 5,
        'max_frames': 300
    }

if 'show_config' not in st.session_state:
    st.session_state.show_config = False

@st.cache_resource
def load_models_cached(audio_path, video_path):
    audio_model = AudioXceptionClassifier(num_classes=2).to(device)
    load_checkpoint_auto(audio_model, audio_path, is_video_model=False)
    audio_model.eval()
    
    video_model = VideoXceptionModel(num_classes=2).to(device)
    load_checkpoint_auto(video_model, video_path, is_video_model=True)
    video_model.eval()
    
    haar_cascade, dnn_net = setup_face_detection()
    return audio_model, video_model, haar_cascade, dnn_net


def process_video(video_path, video_model, audio_model, fusion_module,
                  haar_cascade, dnn_net, config, progress_bar=None):
    
    audio_path = extract_audio_from_video(video_path)
    audio_mels = []
    
    if audio_path and os.path.exists(audio_path):
        audio_mels = preprocess_audio(audio_path, overlap=0.5)
    
    audio_available = len(audio_mels) > 0
    audio_segments_count = len(audio_mels)  # Store actual unique segments
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(total_frames, config['max_frames'])
    frame_skip = config['frame_skip']
    
    results = {
        'video_probs': [], 'audio_probs': [], 'fused_probs': [],
        'fusion_weights': [], 'faces_found': 0,
        'audio_segments_total': 0,  # Total unique audio segments extracted
        'sample_frame': None, 'sample_face_box': None
    }
    
    frame_count = 0
    faces_found = 0
    sample_saved = False
    
    with torch.no_grad():
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                face_result = detect_face_hybrid(frame, haar_cascade, dnn_net)
                
                if face_result is not None:
                    x, y, w, h, _ = face_result
                    
                    # Save sample frame (5th face for better quality)
                    if not sample_saved and faces_found == 5:
                        results['sample_frame'] = frame.copy()
                        results['sample_face_box'] = (x, y, w, h)
                        sample_saved = True
                    
                    face_img = frame[y:y+h, x:x+w]
                    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    face_tensor = video_transform(face_pil).unsqueeze(0).to(device)
                    
                    video_logits = video_model(face_tensor)
                    video_probs = F.softmax(video_logits, dim=1)
                    r_video = fusion_module.compute_reliability(video_logits)
                    
                    if audio_available:
                        audio_idx = faces_found % len(audio_mels)
                        audio_tensor = audio_mels[audio_idx].unsqueeze(0).to(device)
                        audio_logits = audio_model(audio_tensor) / config['audio_temperature']
                        audio_probs = F.softmax(audio_logits, dim=1)
                        r_audio = fusion_module.compute_reliability(audio_logits)
                        
                        if config['fusion_strategy'] == 'equal':
                            alpha_v, alpha_a = 0.5, 0.5
                        elif config['fusion_strategy'] == 'video_trust':
                            alpha_v, alpha_a = 0.85, 0.15
                        else:
                            alpha_v, alpha_a = fusion_module.compute_adaptive_weights(r_video, r_audio, tau=0.5)
                            alpha_v, alpha_a = alpha_v.item(), alpha_a.item()
                        
                        fused_probs = fusion_module.fused_prediction(
                            video_logits, audio_logits,
                            torch.tensor(alpha_v).to(device),
                            torch.tensor(alpha_a).to(device)
                        )
                        
                        results['audio_probs'].append(audio_probs.cpu().numpy())
                        results['fusion_weights'].append((alpha_v, alpha_a))
                    else:
                        fused_probs = video_probs
                        results['fusion_weights'].append((1.0, 0.0))
                    
                    results['video_probs'].append(video_probs.cpu().numpy())
                    results['fused_probs'].append(fused_probs.cpu().numpy())
                    faces_found += 1
                
                if progress_bar:
                    progress_bar.progress(min(frame_count / max_frames, 1.0))
            
            frame_count += 1
    
    cap.release()
    results['faces_found'] = faces_found
    results['audio_segments_total'] = audio_segments_count
    
    # If no sample saved yet, get first available frame with face
    if results['sample_frame'] is None and faces_found > 0:
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            face_result = detect_face_hybrid(frame, haar_cascade, dnn_net)
            if face_result is not None:
                x, y, w, h, _ = face_result
                results['sample_frame'] = frame.copy()
                results['sample_face_box'] = (x, y, w, h)
                break
        cap.release()
    
    if audio_path and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except:
            pass
    
    return results


def get_verdict(results, threshold=0.5):
    if not results or len(results['video_probs']) == 0:
        return None
    
    video_probs = np.array(results['video_probs']).squeeze()
    fused_probs = np.array(results['fused_probs']).squeeze()
    fusion_weights = np.array(results['fusion_weights'])
    
    video_real = video_probs[:, 1].mean()
    fused_real = fused_probs[:, 1].mean()
    
    audio_real = 0
    if len(results['audio_probs']) > 0:
        audio_probs = np.array(results['audio_probs']).squeeze()
        audio_real = audio_probs[:, 1].mean()
    
    verdict = 'REAL' if fused_real > threshold else 'FAKE'
    confidence = fused_real if verdict == 'REAL' else (1 - fused_real)
    
    return {
        'verdict': verdict,
        'confidence': float(confidence),
        'video_real': float(video_real),
        'audio_real': float(audio_real),
        'video_weight': float(fusion_weights[:, 0].mean()),
        'audio_weight': float(fusion_weights[:, 1].mean()),
        'faces': results['faces_found'],
        'audio_segments': results.get('audio_segments_total', 0),
        'sample_frame': results.get('sample_frame'),
        'sample_face_box': results.get('sample_face_box')
    }


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="main-title"> Deepfake Detector</div>
        <div class="main-subtitle"> Advanced Multimodal Detection with Adaptive Fusion </div>
    </div>
    """, unsafe_allow_html=True)
    
    audio_model_path = "best_audio_xception.pth"
    video_model_path = "epoch_007.pt"
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        # Section title - no box
        st.markdown('<div class="section-title">📤 Upload Video</div>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "upload",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            st.video(uploaded_file)
            
            ground_truth = st.radio(
                "Ground Truth (Optional)",
                options=["Unknown", "Real", "Fake"],
                horizontal=True
            )
        
        # Settings button
        if st.button("⚙️ Settings" if not st.session_state.show_config else "✕ Close Settings", 
                     use_container_width=True, key="config_btn"):
            st.session_state.show_config = not st.session_state.show_config
            st.rerun()
        
        if st.session_state.show_config:
            fusion_strategy = st.selectbox(
                "Fusion Strategy",
                options=['equal', 'video_trust', 'adaptive'],
                index=['equal', 'video_trust', 'adaptive'].index(st.session_state.config['fusion_strategy']),
                help="Equal: 50/50 | Video Trust: 85/15 | Adaptive: Dynamic"
            )
            
            audio_temp = st.slider(
                "Audio Temperature",
                min_value=1.0, max_value=10.0,
                value=st.session_state.config['audio_temperature'],
                step=0.5
            )
            
            threshold = st.slider(
                "Decision Threshold",
                min_value=0.3, max_value=0.7,
                value=st.session_state.config['decision_threshold'],
                step=0.05
            )
            
            frame_skip = st.slider(
                "Frame Skip",
                min_value=1, max_value=15,
                value=st.session_state.config['frame_skip']
            )
            
            st.session_state.config.update({
                'fusion_strategy': fusion_strategy,
                'audio_temperature': audio_temp,
                'decision_threshold': threshold,
                'frame_skip': frame_skip
            })
            
            st.info(f"Mode: {fusion_strategy.upper()} | Temp: {audio_temp} | Threshold: {threshold}")
        
        # Analyze button
        if uploaded_file:
            if st.button("🔍 ANALYZE VIDEO", use_container_width=True, key="analyze_btn"):
                if not os.path.exists(audio_model_path) or not os.path.exists(video_model_path):
                    st.error("Model files not found!")
                    return
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(uploaded_file.read())
                    video_path = tmp.name
                
                with st.spinner("Loading AI models..."):
                    try:
                        audio_model, video_model, haar, dnn = load_models_cached(
                            audio_model_path, video_model_path
                        )
                        fusion = AdaptiveFusionModule()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        return
                
                progress = st.progress(0, text="Analyzing...")
                start = time.time()
                
                results = process_video(
                    video_path, video_model, audio_model, fusion,
                    haar, dnn, st.session_state.config, progress
                )
                
                elapsed = time.time() - start
                progress.progress(1.0, text=f"Done in {elapsed:.1f}s")
                
                try:
                    os.remove(video_path)
                except:
                    pass
                
                if results and results['faces_found'] > 0:
                    st.session_state.result = get_verdict(
                        results, st.session_state.config['decision_threshold']
                    )
                    st.session_state.ground_truth = ground_truth
                    st.rerun()
                else:
                    st.error("No faces detected!")
    
    with col2:
        # Section title - no box
        st.markdown('<div class="section-title">📊 Detection Results</div>', unsafe_allow_html=True)
        
        if 'result' in st.session_state and st.session_state.result:
            r = st.session_state.result
            gt = st.session_state.get('ground_truth', 'Unknown')
            
            # Show sample frame with colored bounding box FIRST
            if r.get('sample_frame') is not None and r.get('sample_face_box') is not None:
                sample_frame = r['sample_frame'].copy()
                x, y, w, h = r['sample_face_box']
                
                # GREEN for REAL, RED for FAKE
                if r['verdict'] == 'REAL':
                    box_color = (0, 255, 0)  # Green (BGR)
                else:
                    box_color = (0, 0, 255)  # Red (BGR)
                
                cv2.rectangle(sample_frame, (x, y), (x+w, y+h), box_color, 4)
                sample_rgb = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)
                st.image(sample_rgb, use_container_width=True)
            
            # Then show verdict
            if r['verdict'] == 'REAL':
                st.markdown(f"""
                <div class="result-box result-box-real">
                    <div class="verdict-icon">✅</div>
                    <div class="verdict-real">VERIFIED AUTHENTIC</div>
                    <div class="confidence-label">CONFIDENCE: {r['confidence']*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box result-box-fake">
                    <div class="verdict-icon">⚠️</div>
                    <div class="verdict-fake">DEEPFAKE DETECTED</div>
                    <div class="confidence-label">CONFIDENCE: {r['confidence']*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Calculate fake percentages
            video_fake = (1 - r['video_real']) * 100
            audio_fake = (1 - r['audio_real']) * 100
            video_real_pct = r['video_real'] * 100
            audio_real_pct = r['audio_real'] * 100
            
            st.markdown(f"""
            <div class="modality-section">
                <div class="modality-title">🎥 Video Analysis</div>
                <div class="modality-bar-container">
                    <div class="modality-bar-label">
                        <span class="label-real">REAL: {video_real_pct:.1f}%</span>
                        <span class="label-fake">FAKE: {video_fake:.1f}%</span>
                    </div>
                    <div class="modality-bar">
                        <div class="bar-real" style="width: {video_real_pct}%;"></div>
                        <div class="bar-fake" style="width: {video_fake}%;"></div>
                    </div>
                </div>
            </div>
            
            <div class="modality-section">
                <div class="modality-title">🔊 Audio Analysis</div>
                <div class="modality-bar-container">
                    <div class="modality-bar-label">
                        <span class="label-real">REAL: {audio_real_pct:.1f}%</span>
                        <span class="label-fake">FAKE: {audio_fake:.1f}%</span>
                    </div>
                    <div class="modality-bar">
                        <div class="bar-real" style="width: {audio_real_pct}%;"></div>
                        <div class="bar-fake" style="width: {audio_fake}%;"></div>
                    </div>
                </div>
            </div>
            
            <div class="analysis-info">
                <div class="info-item">
                    <span class="info-icon">👤</span>
                    <span class="info-text">{r['faces']} faces analyzed</span>
                </div>
                <div class="info-item">
                    <span class="info-icon">🎵</span>
                    <span class="info-text">{r.get('audio_segments', 0)} audio segments analyzed</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("**Fusion Weights**")
            
            st.markdown(f"""
            <div class="weight-section">
                <div class="weight-item">
                    <div class="weight-header">
                        <span class="weight-name">🎥 Video</span>
                        <span class="weight-value">{r['video_weight']*100:.0f}%</span>
                    </div>
                    <div class="progress-bg">
                        <div class="progress-video" style="width: {r['video_weight']*100}%;"></div>
                    </div>
                </div>
                <div class="weight-item">
                    <div class="weight-header">
                        <span class="weight-name">🔊 Audio</span>
                        <span class="weight-value">{r['audio_weight']*100:.0f}%</span>
                    </div>
                    <div class="progress-bg">
                        <div class="progress-audio" style="width: {r['audio_weight']*100}%;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if gt != "Unknown":
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                actual = gt.upper()
                if r['verdict'] == actual:
                    st.success(f"✓ Correct! Ground truth: {actual}")
                else:
                    st.error(f"✗ Wrong. Ground truth: {actual}")
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🎬</div>
                <div class="empty-text">Upload a video and click<br><strong>ANALYZE VIDEO</strong><br>to see results</div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
