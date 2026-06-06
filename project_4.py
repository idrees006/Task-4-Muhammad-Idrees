import cv2
import pytesseract
import numpy as np
import os

# ==========================================================
# Artificial Intelligence Project 4
# Image Recognition using OCR and Object Detection
# ==========================================================

# ------------------------------
# SETTINGS
# ------------------------------

# Choose mode:
# "OCR"    -> Text Recognition
# "OBJECT" -> Object Detection

MODE = "OCR"

# Image path
IMAGE_PATH = "sample.jpg"

# Uncomment and edit if using Windows
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ==========================================================
# OCR FUNCTION
# ==========================================================

def ocr_recognition(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5,5), 0)

    gray = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    text = pytesseract.image_to_string(gray)

    print("\n==============================")
    print("Recognized Text")
    print("==============================")
    print(text)

    cv2.imshow("Original Image", image)
    cv2.imshow("Processed Image", gray)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==========================================================
# OBJECT DETECTION FUNCTION
# ==========================================================

def object_detection(image_path):

    CLASSES = [
        "background","aeroplane","bicycle","bird","boat",
        "bottle","bus","car","cat","chair",
        "cow","diningtable","dog","horse","motorbike",
        "person","pottedplant","sheep","sofa","train","tvmonitor"
    ]

    COLORS = np.random.uniform(
        0,
        255,
        size=(len(CLASSES),3)
    )

    model = "MobileNetSSD_deploy.caffemodel"
    config = "MobileNetSSD_deploy.prototxt"

    if not os.path.exists(model):
        print("Model file not found.")
        return

    if not os.path.exists(config):
        print("Prototxt file not found.")
        return

    net = cv2.dnn.readNetFromCaffe(
        config,
        model
    )

    image = cv2.imread(image_path)

    if image is None:
        print("Image not found.")
        return

    (h,w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image,(300,300)),
        scalefactor=0.007843,
        size=(300,300),
        mean=127.5
    )

    net.setInput(blob)

    detections = net.forward()

    print("\nDetected Objects\n")

    for i in range(detections.shape[2]):

        confidence = detections[0,0,i,2]

        if confidence > 0.50:

            idx = int(detections[0,0,i,1])

            box = detections[0,0,i,3:7] * np.array([w,h,w,h])

            (startX,startY,endX,endY) = box.astype("int")

            label = "{} : {:.2f}%".format(
                CLASSES[idx],
                confidence*100
            )

            print(label)

            cv2.rectangle(
                image,
                (startX,startY),
                (endX,endY),
                COLORS[idx],
                2
            )

            y = startY - 10 if startY > 20 else startY + 20

            cv2.putText(
                image,
                label,
                (startX,y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLORS[idx],
                2
            )

    cv2.imshow("Detected Objects", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==========================================================
# MAIN PROGRAM
# ==========================================================

print("===================================")
print("Artificial Intelligence Project 4")
print("Image Recognition System")
print("===================================")

if MODE.upper() == "OCR":
    print("\nRunning OCR...")
    ocr_recognition(IMAGE_PATH)

elif MODE.upper() == "OBJECT":
    print("\nRunning Object Detection...")
    object_detection(IMAGE_PATH)

else:
    print("Invalid MODE selected.")