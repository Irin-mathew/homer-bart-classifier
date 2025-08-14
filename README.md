# 🏠 Homer vs Bart Image Classifier

[![FastAI](https://img.shields.io/badge/ML-FastAI-orange)](https://fast.ai)
[![ResNet-50](https://img.shields.io/badge/Architecture-ResNet--50-red)](https://arxiv.org/abs/1512.03385)
[![Flask](https://img.shields.io/badge/Web-Flask-blue)](https://flask.palletsprojects.com)
[![Deep Learning](https://img.shields.io/badge/AI-Deep%20Learning-green)](https://pytorch.org)

> **AI-powered web app that instantly classifies images as Homer Simpson or Bart Simpson using state-of-the-art deep learning**

![Demo](https://img.shields.io/badge/Demo-Live-success) ![Accuracy](https://img.shields.io/badge/Accuracy-High-brightgreen) ![Real Time](https://img.shields.io/badge/Inference-Real%20Time-yellow)

## 🚀 What It Does

Upload any image → Get instant AI classification → Homer or Bart? ⚡

## 🧠 Core Technologies

### **Deep Learning Architecture**
- **🏗️ ResNet-50**: 50-layer Residual Neural Network with skip connections
- **📚 Transfer Learning**: Pre-trained on ImageNet (1.4M images, 1000 classes)
- **🎯 Fine-tuning**: Adapted for Homer vs Bart binary classification
- **⚡ FastAI Framework**: Research-grade results with minimal code

### **Technical Stack**
| Component | Technology | Purpose |
|-----------|------------|---------|
| **🧠 ML Model** | ResNet-50 + FastAI | Image classification |
| **🌐 Web Backend** | Flask | API & file handling |
| **🎨 Frontend** | HTML5/CSS3/JS | Modern responsive UI |
| **⚙️ Deep Learning** | PyTorch | Neural network operations |

## 📊 Architecture Highlights

### **ResNet-50 Architecture**
```
Input (224×224×3) → Conv Layers → 16 Residual Blocks → 
Global Avg Pool → FC Layer → 2 Classes (Homer/Bart)
```

- **Skip Connections**: Solves vanishing gradient problem
- **50 Layers Deep**: 25.6M parameters
- **Batch Normalization**: Stable training
- **ReLU Activations**: Non-linear transformations

### **Training Pipeline**
```python
# FastAI Magic in 4 lines
dls = ImageDataLoaders.from_folder(path, valid_pct=0.2)
model = vision_learner(dls, resnet50, metrics=accuracy)
model.fit(epochs)
model.export('model.pkl')
```

## 🎯 Key Features

- **🔥 Real-time Inference**: Sub-second predictions
- **📱 Responsive Design**: Works on all devices  
- **🖱️ Drag & Drop**: Intuitive file upload
- **🔄 Reset Function**: Try multiple images easily
- **⚡ Transfer Learning**: Leverages ImageNet knowledge

## 🏗️ Project Structure

```
homer-bart-classifier/
├── 🌐 app.py                       # Flask web application  
├── 📊 Copy of FatstAI.ipynb       # Model training code
├── 🎨 templates/index.html         # Modern web interface
├── 📦 requirements.txt             # Dependencies
└── 📁 static/uploads/              # Image storage
```

## 🚀 Quick Start

```bash
# Clone & Setup
git clone https://github.com/Irin-mathew/homer-bart-classifier.git
cd homer-bart-classifier
pip install -r requirements.txt

# Run Application  
python app.py
# Open: http://localhost:5000
```

## 🔬 Technical Deep Dive

### **Model Training Process**
1. **Data Augmentation**: Random transforms for robustness
2. **ImageNet Normalization**: RGB mean/std standardization  
3. **Progressive Resizing**: 224×224 input resolution
4. **Transfer Learning**: Frozen backbone + trainable head
5. **Validation Split**: 80/20 train/validation split

### **Web Application Flow**
```
Image Upload → File Validation → Model Inference → 
Result Display → Reset Option
```

### **FastAI Advantages**
- **🎯 Best Practices Built-in**: Research-grade techniques automatically
- **📈 State-of-the-art Results**: Competition-winning approaches
- **⚡ Rapid Prototyping**: PhD-level ML with minimal code
- **🔧 Production Ready**: Easy model export and deployment

## 📈 Performance

- **⚡ Inference Time**: < 1 second per image
- **🎯 Architecture**: 50-layer ResNet with 25.6M parameters
- **📊 Training**: Transfer learning with progressive unfreezing
- **💾 Model Size**: Optimized for web deployment

## 🛠️ Technologies Used

### **Machine Learning**
- **FastAI 2.7+**: High-level ML framework
- **PyTorch**: Deep learning backend
- **ResNet-50**: Convolutional neural network
- **Transfer Learning**: ImageNet pre-training

### **Web Development**  
- **Flask**: Python web framework
- **HTML5/CSS3**: Modern frontend
- **JavaScript**: Interactive UI elements
- **Responsive Design**: Mobile-friendly

## 🎨 UI Features

- **Modern Gradient Design**: Eye-catching visual appeal
- **Drag & Drop Interface**: Intuitive file upload
- **Loading Animations**: Visual feedback during processing  
- **Mobile Responsive**: Works on all screen sizes
- **Reset Functionality**: Easy multi-image testing

## 📦 Installation

```bash
# Dependencies
pip install fastai>=2.7.0 flask>=2.0.0 torch>=1.9.0
```

## 🚀 Future Enhancements

- [ ] **Multi-character Classification**: Extend to all Simpsons characters
- [ ] **Confidence Scores**: Show prediction probabilities  
- [ ] **Batch Processing**: Multiple image uploads
- [ ] **API Endpoints**: Programmatic access
- [ ] **Model Interpretability**: Visualize decision process

## 🏆 Why This Project Rocks

✅ **Production-Ready**: Complete ML pipeline from training to deployment  
✅ **Modern Architecture**: State-of-the-art ResNet-50 with transfer learning  
✅ **User-Friendly**: Intuitive web interface with modern design  
✅ **Fast & Accurate**: Real-time inference with high accuracy  
✅ **Scalable**: Easy to extend for more characters or applications  

---

**⭐ Star this repo if you found it helpful!** 

**📬 Questions?** Open an issue or contact [irinmathew264@gmail.com](https://github.com/Irin-mathew) 
