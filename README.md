# 🎭 Deepfake Detector - Multimodal Detection System

A state-of-the-art deepfake detection system that combines video and audio analysis using deep learning. This project implements a multimodal approach with adaptive fusion to detect deepfake videos with high accuracy.

![Deepfake Detection](https://img.shields.io/badge/Deepfake-Detection-red) ![Multimodal](https://img.shields.io/badge/Multimodal-Video%20%2B%20Audio-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-green)

## 🚀 Try It Now!

**👉 [Use the Online Demo on Hugging Face](https://huggingface.co/spaces/Mbulsssss/deepfake-detector) - No installation required!**

Or [run it locally](#-quick-start) for full control and customization.


## 🎯 Overview

This deepfake detection system leverages multimodal learning by analyzing both visual (video frames) and auditory (audio spectrograms) features. The system uses Xception-based neural networks for both modalities and combines their predictions using an adaptive fusion mechanism that dynamically weights each modality based on confidence scores.

**🎯 Two Ways to Use:**
- **🌐 Online Demo**: [Try it instantly on Hugging Face](https://huggingface.co/spaces/Mbulsssss/deepfake-detector) - No setup needed!
- **💻 Local Installation**: Full control and customization (see [Installation](#-installation) below)

### Key Highlights

- **Multimodal Analysis**: Simultaneously processes video frames and audio spectrograms
- **Adaptive Fusion**: Dynamically combines predictions based on modality reliability
- **Hybrid Face Detection**: Uses DNN + Haar Cascade for robust face detection
- **Web Interface**: Beautiful Streamlit-based web application for easy interaction
- **Production Ready**: Includes preprocessing pipelines, model training scripts, and evaluation tools

## ✨ Features

### Core Capabilities

- ✅ **Video Analysis**: Extracts and analyzes face regions from video frames
- ✅ **Audio Analysis**: Processes audio into mel-spectrograms for deepfake detection
- ✅ **Adaptive Fusion**: Three fusion strategies (Equal, Video Trust, Adaptive)
- ✅ **Real-time Detection**: Fast inference with GPU acceleration support
- ✅ **Confidence Scoring**: Provides detailed confidence metrics for each modality
- ✅ **Face Detection**: Hybrid approach (DNN + Haar Cascade) for robust detection
- ✅ **Web Interface**: Modern, responsive Streamlit UI with real-time results

### Advanced Features

- **Temperature Scaling**: Calibrates audio predictions for better fusion
- **Frame Skipping**: Configurable frame sampling for efficient processing
- **Abstention Mode**: Option to abstain from predictions when confidence is low
- **Cross-Dataset Evaluation**: Support for testing on different datasets
- **Comprehensive Metrics**: Accuracy, F1-score, precision, recall, and ROC curves

## 🏗️ Architecture

### Model Architecture

#### Video Model
- **Backbone**: Xception (from `timm`)
- **Input**: 299×299 RGB face crops
- **Output**: Binary classification (Real/Fake)
- **Architecture**: Simple classifier (2048 → 2)

#### Audio Model
- **Backbone**: Xception (from `timm`)
- **Input**: 299×299 mel-spectrogram images (3-channel RGB)
- **Output**: Binary classification (Real/Fake)
- **Architecture**: Complex classifier (2048 → 512 → 256 → 2)

#### Fusion Module
- **Adaptive Fusion**: Confidence-aware weight computation
- **Reliability Score**: `r_i = max_c(p_i,c)` where p is softmax probability
- **Adaptive Weights**: `α_i = exp(τ × r_i) / Σ exp(τ × r_j)`
- **Fused Prediction**: `Z_fused = α_v × Z_v + α_a × Z_a`

### Processing Pipeline


<img width="385" height="296" alt="image" src="https://github.com/user-attachments/assets/1dbb512a-de8e-477f-ac16-ee5a9ae2a7f2" />


## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- FFmpeg (for audio extraction)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd Deepfake-Detecetor
```

### Step 2: Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install streamlit opencv-python pillow
pip install librosa soundfile
pip install timm pandas numpy scikit-learn
pip install matplotlib seaborn tqdm
```

### Step 3: Install FFmpeg

**Windows:**
```bash
# Download from https://ffmpeg.org/download.html
# Or use chocolatey:
choco install ffmpeg
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### Step 4: Download Model Checkpoints

Download the pre-trained model checkpoints from Hugging Face and place them in the project root:

- **Audio Model**: Download [`best_audio_xception.pth`](https://huggingface.co/spaces/Mbulsssss/deepfake-detector/blob/main/best_audio_xception.pth) (264 MB)
- **Video Model**: Download [`epoch_007.pt`](https://huggingface.co/spaces/Mbulsssss/deepfake-detector/blob/main/epoch_007.pt) (207 MB)

**Quick Download:**
```bash
# Using wget (Linux/Mac)
wget https://huggingface.co/spaces/Mbulsssss/deepfake-detector/resolve/main/best_audio_xception.pth
wget https://huggingface.co/spaces/Mbulsssss/deepfake-detector/resolve/main/epoch_007.pt

# Using curl (Linux/Mac)
curl -L https://huggingface.co/spaces/Mbulsssss/deepfake-detector/resolve/main/best_audio_xception.pth -o best_audio_xception.pth
curl -L https://huggingface.co/spaces/Mbulsssss/deepfake-detector/resolve/main/epoch_007.pt -o epoch_007.pt
```

*Note: Model files are too large for GitHub, so they are hosted on [Hugging Face](https://huggingface.co/spaces/Mbulsssss/deepfake-detector). If you want to train your own models, see [Model Training](#model-training) section.*

## 🎬 Quick Start

You have **two options** to use the Deepfake Detector:

### Option 1: Use Online Demo (Easiest - No Installation Required) 🌐

**Try it instantly on Hugging Face Spaces!**

👉 **[Open Deepfake Detector on Hugging Face](https://huggingface.co/spaces/Mbulsssss/deepfake-detector)**

Simply upload a video and get instant results - no setup required! The demo is fully functional and runs on Hugging Face's servers.

### Option 2: Run Locally

#### 2a. Web Interface (Recommended for Local Use)

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

#### 2b. Command Line

```bash
python multimodal.py
```

This will open a file dialog to select a video file for analysis.

## 📖 Usage

### Web Interface

1. **Upload Video**: Click "Upload Video" and select a video file (MP4, AVI, MOV, MKV, WEBM)
2. **Configure Settings** (Optional): Click "⚙️ Settings" to adjust:
   - Fusion Strategy: Equal, Video Trust, or Adaptive
   - Audio Temperature: Calibration factor (1.0-10.0)
   - Decision Threshold: Classification threshold (0.3-0.7)
   - Frame Skip: Frames to skip between analysis (1-15)
3. **Analyze**: Click "🔍 ANALYZE VIDEO" to start detection
4. **View Results**: See detailed analysis including:
   - Final verdict (REAL/FAKE)
   - Confidence score
   - Individual modality predictions
   - Fusion weights
   - Sample frame with face detection

### Python API

```python
from multimodal import (
    AudioXceptionClassifier,
    VideoXceptionModel,
    AdaptiveFusionModule,
    load_checkpoint_auto,
    process_video_multimodal,
    setup_face_detection,
    analyze_results
)

# Load models
audio_model = AudioXceptionClassifier(num_classes=2).to(device)
load_checkpoint_auto(audio_model, "best_audio_xception.pth", is_video_model=False)

video_model = VideoXceptionModel(num_classes=2).to(device)
load_checkpoint_auto(video_model, "epoch_007.pt", is_video_model=True)

# Setup face detection
haar_cascade, dnn_net = setup_face_detection()

# Initialize fusion module
fusion_module = AdaptiveFusionModule()

# Process video
results = process_video_multimodal(
    video_path="path/to/video.mp4",
    video_model=video_model,
    audio_model=audio_model,
    fusion_module=fusion_module,
    haar_cascade=haar_cascade,
    dnn_net=dnn_net,
    fusion_strategy='adaptive',
    audio_temperature=2.0
)

# Analyze results
analysis = analyze_results(results, decision_threshold=0.5)
print(f"Prediction: {'REAL' if analysis['predicted_label'] == 1 else 'FAKE'}")
print(f"Confidence: {analysis['confidence']:.3f}")
```

## 📂 Dataset

### FakeAVCeleb Dataset

This project uses the **FakeAVCeleb** dataset, which contains synchronized video and audio deepfakes.

#### Dataset Links

- **Original Dataset**: [FakeAVCeleb (Full Dataset)](https://www.kaggle.com/datasets/mbulsss/fakeavceleb)
  - Contains original real and deepfake videos with synchronized audio
  
- **Preprocessed Video Frames**: [FakeAVCeleb Preprocessed Frames](https://www.kaggle.com/datasets/mbulsss/fakeavceleb-preprocessed-frame)
  - Pre-extracted face frames for efficient training
  
- **Preprocessed Audio**: [FakeAVCeleb Audio](https://www.kaggle.com/datasets/mbulsss/fakeavceleb-audio)
  - Preprocessed mel-spectrograms for audio modality

#### Dataset Structure

```
FakeAVCeleb_v1.2/
├── RealVideo-RealAudio/
│   ├── African/
│   ├── Asian/
│   ├── Caucasian/
│   └── Indian/
└── FakeVideo-RealAudio/
    └── ...
```

#### Preprocessing

The project includes preprocessing notebooks:
- `Preprocessing_Video.ipynb`: Extracts and preprocesses video frames
- `Preprocessing_Audio.ipynb`: Converts audio to mel-spectrograms

## 🎓 Model Training

### Video Model Training

Use `VIDEO_FINALLLLLL.ipynb` for training the video model:

1. **Setup**: Mount Google Drive and download preprocessed dataset
2. **Configuration**: Adjust hyperparameters in Config class
3. **Training**: Run all cells to train Xception model
4. **Checkpoints**: Models saved to Google Drive

**Key Training Parameters:**
- Model: Xception (pretrained)
- Batch Size: 128 (A100 GPU)
- Learning Rate: 0.001
- Epochs: 25
- Mixed Precision: Enabled

### Audio Model Training

Use `AUDIO_FINAL.ipynb` for training the audio model:

1. **Setup**: Download audio dataset from Kaggle
2. **Data Loading**: Load preprocessed mel-spectrograms
3. **Training**: Train Xception-based audio classifier
4. **Evaluation**: Test on validation set

**Key Training Parameters:**
- Model: Xception with custom classifier
- Input: 299×299 mel-spectrograms (3-channel)
- Augmentation: SpecAugment
- Dropout: 0.5

### Multimodal Experiments

Use `Multimodal_Experiment.ipynb` for:
- Cross-dataset evaluation
- Fusion strategy comparison
- Hyperparameter tuning
- Performance analysis

## 📁 Project Structure

```
Deepfake-Detecetor/
├── streamlit_app.py              # Web application interface
├── multimodal.py                 # Core multimodal processing code
├── VIDEO_FINALLLLLL.ipynb        # Video model training notebook
├── AUDIO_FINAL.ipynb             # Audio model training notebook
├── Multimodal_Experiment.ipynb   # Multimodal experiments
├── Preprocessing_Video.ipynb      # Video preprocessing pipeline
├── Preprocessing_Audio.ipynb     # Audio preprocessing pipeline
├── cross_dataset_lavdf.ipynb     # Cross-dataset evaluation
├── README.md                      # This file
│
├── Model Checkpoints (download from Hugging Face):
│   ├── best_audio_xception.pth   # Audio model weights (264 MB)
│   └── epoch_007.pt              # Video model weights (207 MB)
│   └── See Installation Step 4 for download links
│
└── Face Detection Models (auto-downloaded):
    ├── haarcascade_frontalface_default.xml
    ├── deploy.prototxt
    └── res10_300x300_ssd_iter_140000.caffemodel
```

## 🔧 Technical Details

### Face Detection

**Hybrid Approach:**
1. **Primary**: DNN-based face detector (OpenCV DNN)
   - Model: ResNet-10 SSD
   - Confidence threshold: 0.3
   - More accurate for frontal faces
2. **Fallback**: Haar Cascade
   - Used when DNN fails
   - Better for profile/angled faces

### Audio Processing

**Mel-Spectrogram Generation:**
- Sample Rate: 16,000 Hz
- Window Size: 2.0 seconds
- Overlap: 50% (configurable)
- N_MELS: 128
- N_FFT: 2048
- Frequency Range: 20-8000 Hz
- Resize: 299×299 (for Xception input)
- Normalization: Mean-std normalization

### Video Processing

**Frame Processing:**
- Face Crop: Extracted from detected face regions
- Resize: 299×299 (Xception input size)
- Normalization: Mean=[0.5, 0.5, 0.5], Std=[0.5, 0.5, 0.5]
- Frame Skip: Configurable (default: 5)
- Max Frames: Configurable (default: 300-500)

### Fusion Strategies

1. **Equal (50/50)**: Simple average of both modalities
2. **Video Trust (85/15)**: Prioritizes video predictions
3. **Adaptive**: Dynamic weights based on reliability scores
   - Formula: `α_i = exp(τ × r_i) / Σ exp(τ × r_j)`
   - τ (tau): Temperature parameter (default: 0.2-1.0)

## ⚙️ Configuration

### Streamlit App Configuration

Edit `streamlit_app.py` to modify:
- Model paths
- Default fusion strategy
- UI styling
- Processing parameters

### Multimodal Processing Configuration

Edit `multimodal.py` main section:

```python
AUDIO_MODEL_PATH = "best_audio_xception.pth"
VIDEO_MODEL_PATH = "epoch_007.pt"
AUDIO_TEMPERATURE = 2.0
FUSION_TAU = 0.2
DECISION_THRESHOLD = 0.5
FUSION_STRATEGY = 'equal'  # 'equal', 'video_trust', 'adaptive'
AUDIO_OVERLAP = 0.5
ENABLE_ABSTENTION = True
ABSTENTION_THRESHOLD = 0.6
```

## 🐛 Troubleshooting

### Common Issues

**1. Model files not found**
```
Error: Model files not found!
```
**Solution**: Download the model files from Hugging Face:
- [best_audio_xception.pth](https://huggingface.co/spaces/Mbulsssss/deepfake-detector/blob/main/best_audio_xception.pth)
- [epoch_007.pt](https://huggingface.co/spaces/Mbulsssss/deepfake-detector/blob/main/epoch_007.pt)

Place both files in the project root directory.

**2. FFmpeg not found**
```
Error: ffmpeg not found
```
**Solution**: Install FFmpeg and ensure it's in your system PATH.

**3. No faces detected**
```
Error: No faces detected!
```
**Solution**: 
- Check video quality and lighting
- Ensure faces are clearly visible
- Try adjusting `min_size` parameter in face detection

**4. CUDA out of memory**
```
RuntimeError: CUDA out of memory
```
**Solution**:
- Reduce batch size
- Reduce `max_frames` parameter
- Use CPU mode (slower but works)

**5. Audio extraction fails**
```
Error: Audio extraction error
```
**Solution**:
- Check FFmpeg installation
- Verify video has audio track
- Check video file format compatibility

### Performance Tips

- **GPU Acceleration**: Use CUDA-enabled PyTorch for faster inference
- **Frame Skipping**: Increase `frame_skip` for faster processing (may reduce accuracy)
- **Max Frames**: Reduce `max_frames` for shorter processing time
- **Batch Processing**: Process multiple videos in sequence for efficiency

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- Model architecture improvements
- Additional fusion strategies
- Performance optimizations
- Documentation improvements
- Bug fixes
- New features

## 🙏 Acknowledgments

- **FakeAVCeleb Dataset**: [mbulsss on Kaggle](https://www.kaggle.com/datasets/mbulsss/fakeavceleb)
- **Xception Architecture**: Original paper by François Chollet
- **OpenCV**: Face detection models
- **Streamlit**: Web framework
- **PyTorch & timm**: Deep learning frameworks

## 📧 Contact

For questions, issues, or collaborations, please open an issue on GitHub.

---
