# Task-4-Muhammad-Idrees: Image Recognition System

## 📋 Project Overview

**Artificial Intelligence Project 4** is a comprehensive image recognition system that implements two powerful computer vision techniques:

1. **OCR (Optical Character Recognition)** - Text recognition from images
2. **Object Detection** - Identifying and locating objects within images

This project leverages OpenCV, PyTesseract, and deep learning models to provide flexible image analysis capabilities.

---

## 🎯 Project Features

### 1. OCR Text Recognition
- **Reads text from images** using Tesseract OCR engine
- **Image preprocessing** for improved accuracy:
  - Grayscale conversion
  - Gaussian blur for noise reduction
  - Binary thresholding with Otsu's method
- **Visual output** showing original and processed images

### 2. Object Detection
- **Detects 20 different object classes** using MobileNetSSD model
- **Deep learning-based detection** using pre-trained Caffe model
- **Confidence scoring** (50% threshold) for reliable detections
- **Visual annotations** with bounding boxes and labels on detected objects

---

## 🛠️ Installation & Requirements

### Prerequisites
- Python 3.x
- OpenCV
- PyTesseract
- NumPy
- Tesseract-OCR engine (for text recognition)
- MobileNetSSD model files (for object detection)

### Install Dependencies

```bash
pip install opencv-python pytesseract numpy
```

### Additional Setup

#### For OCR (Tesseract)
- **Linux/Mac**: Install via package manager
  ```bash
  # Ubuntu/Debian
  sudo apt-get install tesseract-ocr
  
  # macOS
  brew install tesseract
  ```

- **Windows**: Download and install from [Tesseract-OCR GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
  - Then uncomment and configure in the code:
    ```python
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    ```

#### For Object Detection (MobileNetSSD)
Download the model files:
- `MobileNetSSD_deploy.caffemodel`
- `MobileNetSSD_deploy.prototxt`

Place these files in your project directory.

---

## 🚀 Usage

### Configuration
Edit the **SETTINGS** section in `project_4.py`:

```python
# Choose mode:
MODE = "OCR"              # Change to "OBJECT" for object detection
IMAGE_PATH = "sample.jpg" # Path to your image file
```

### Running the Program

```bash
python project_4.py
```

### Mode Selection

#### OCR Mode
```python
MODE = "OCR"
```
- Recognizes text from the image
- Displays original and processed images
- Prints extracted text to console

#### Object Detection Mode
```python
MODE = "OBJECT"
```
- Detects objects in the image
- Displays bounding boxes and confidence scores
- Prints detected objects to console

---

## 📊 Detectable Objects (20 Classes)

The object detection model can identify:

| Category | Objects |
|----------|---------|
| **Vehicles** | Aeroplane, Bicycle, Boat, Bus, Car, Motorbike, Train |
| **Animals** | Bird, Cat, Cow, Dog, Horse, Sheep |
| **People** | Person |
| **Furniture** | Chair, Diningtable, Sofa |
| **Plants** | Pottedplant |
| **Electronics** | TVmonitor |
| **Other** | Bottle, Background |

---

## 🔧 Key Functions

### `ocr_recognition(image_path)`
Performs optical character recognition on an image.

**Process:**
1. Load image
2. Convert to grayscale
3. Apply Gaussian blur (5×5 kernel)
4. Binary thresholding with Otsu's method
5. Extract text using Tesseract
6. Display results

**Parameters:**
- `image_path` (str): Path to image file

---

### `object_detection(image_path)`
Detects objects using MobileNetSSD deep learning model.

**Process:**
1. Load model and configuration
2. Read image
3. Create blob (normalize and resize to 300×300)
4. Forward pass through neural network
5. Filter detections (confidence > 50%)
6. Draw bounding boxes and labels
7. Display annotated image

**Parameters:**
- `image_path` (str): Path to image file

**Confidence Threshold:** 50% (adjustable in code)

---

## 📁 Project Structure

```
Task-4-Muhammad-Idrees/
├── project_4.py                          # Main program
├── README.md                             # This file
├── sample.jpg                            # Example image (add your own)
├── MobileNetSSD_deploy.caffemodel        # (Required for object detection)
└── MobileNetSSD_deploy.prototxt          # (Required for object detection)
```

---

## 💡 Example Workflows

### Recognize Text from a Document
```python
MODE = "OCR"
IMAGE_PATH = "document.jpg"
# Extracts and displays all text from the document
```

### Detect Objects in a Photo
```python
MODE = "OBJECT"
IMAGE_PATH = "photo.jpg"
# Identifies and marks all objects with bounding boxes
```

---

## 📈 Performance Notes

- **OCR Accuracy**: Depends on image quality, resolution, and text clarity
- **Object Detection Speed**: Real-time on most modern systems (GPU recommended for faster processing)
- **Confidence Threshold**: Set to 50% to balance precision and recall (adjust as needed)

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Image not found" | Verify `IMAGE_PATH` is correct and file exists |
| "Model file not found" | Download MobileNetSSD files and place in project directory |
| Tesseract not found (Windows) | Configure `pytesseract.pytesseract.tesseract_cmd` path |
| Poor OCR results | Ensure image has good contrast and resolution |
| Slow object detection | Consider using GPU acceleration or smaller input images |

---

## 📚 Libraries Used

- **OpenCV (cv2)**: Image processing and DNN module
- **PyTesseract**: Python wrapper for Tesseract OCR
- **NumPy**: Numerical operations and array handling
- **os**: File system operations

---

## 🎓 Learning Objectives

This project demonstrates:
- ✅ Image preprocessing techniques
- ✅ Optical Character Recognition (OCR)
- ✅ Deep learning with pre-trained models
- ✅ Object detection and localization
- ✅ Bounding box annotation
- ✅ Confidence scoring and thresholding

---

## 📝 License

This project is part of an Artificial Intelligence course assignment.

---

## 👨‍💻 Author

**Muhammad Idrees**

---

## 📞 Support

For issues or questions, refer to the code comments or consult the documentation for:
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTesseract Documentation](https://pytesseract.readthedocs.io/)
- [Tesseract-OCR Wiki](https://github.com/UB-Mannheim/tesseract/wiki)

---

**Last Updated:** June 2026
