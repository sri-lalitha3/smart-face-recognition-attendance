# Smart Face Recognition Attendance System

An automated face recognition attendance system built in Python using OpenCV. The system trains an LBPH (Local Binary Pattern Histogram) face recognition model on a custom dataset of labelled face images, detects and recognises faces in uploaded photos, and automatically logs attendance with timestamps into a downloadable CSV file. Built and tested in Google Colab.

---

## What This Project Does

1. **Uploads and extracts** a labelled face image dataset (ZIP format)
2. **Trains an LBPH face recognition model** using `cv2.face.LBPHFaceRecognizer_create()` on the dataset
3. **Saves the trained model** as `lbph_model.yml` for reuse
4. **Detects faces** in uploaded test images using OpenCV Haar Cascade Classifier
5. **Predicts identity** of each detected face using the trained LBPH model (confidence threshold < 60)
6. **Draws bounding boxes** and labels on recognised faces
7. **Logs attendance** — recognised person ID and timestamp — into a CSV file
8. **Downloads the CSV** attendance report automatically

---

## Technologies Used

| Technology | How It Was Used |
|---|---|
| Python | Core programming language |
| OpenCV (cv2) | Haar Cascade face detection + LBPH face recognition |
| LBPH Algorithm | `cv2.face.LBPHFaceRecognizer_create()` for face recognition |
| Haar Cascade | `haarcascade_frontalface_default.xml` for face detection |
| NumPy | Image array processing |
| Pandas | Attendance DataFrame creation and CSV export |
| datetime | Timestamp generation for attendance logging |
| Google Colab | Development, execution, file upload/download |
| Jupyter Notebook | Interactive code and output |

---

## Dataset Structure

The dataset uses numbered folder names in this format:

```
dataset/
├── 0_John/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
├── 1_Mary/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
```

- Folder name format: `{label}_{PersonName}`
- The number before the underscore becomes the label ID in the model
- Each folder contains face images of that person

> Sample dataset with 2 people (John and Mary, 3 images each) is included in `dataset.zip`

---

## How to Run

### Run in Google Colab (Recommended)

1. Open the notebook in Google Colab using the link below
2. Click **Runtime → Run All**
3. When prompted, upload `dataset.zip`
4. After training, upload a test image when prompted
5. The system will detect and recognise faces, show the annotated image, and download `attendance.csv`

**Open in Google Colab:**
[Face Recognition Attendance Notebook](https://colab.research.google.com/drive/1GW1_q2XBRdaOWda3SeqW1ZodH1jvock8?usp=sharing)

### Run Locally

```bash
git clone https://github.com/sri-lalitha3/smart-face-recognition-attendance.git
cd smart-face-recognition-attendance
pip install opencv-contrib-python pandas
jupyter notebook Face_Recognition_Attendance_System.ipynb
```

---

## Project Structure

```
smart-face-recognition-attendance/
├── Face_Recognition_Attendance_System.ipynb   # Main notebook
├── dataset.zip                                # Sample face dataset (2 people)
├── requirements (1).txt                       # Dependencies
└── README.md                                  # Project documentation
```

---

## How LBPH Face Recognition Works

**LBPH (Local Binary Pattern Histogram)** is a classic computer vision algorithm for face recognition:

1. Each face image is divided into small local regions
2. For each region, pixel values are compared to neighbours to generate a binary pattern
3. These patterns are collected into a histogram describing the face's texture
4. During recognition, the stored histogram is compared to new face histograms
5. The closest match below the confidence threshold is returned as the predicted identity

This makes LBPH robust to changes in lighting and is computationally efficient for small datasets.

---

## Output

- Annotated image with green bounding boxes and ID labels on detected faces
- `attendance.csv` file with columns:

| ID | Time |
|---|---|
| 0 | 2024-04-19 10:32:15 |
| 1 | 2024-04-19 10:32:16 |

---

## Key Learnings

- Trained a custom face recognition model using OpenCV LBPH algorithm on a labelled dataset
- Understood the difference between face detection (finding where faces are) and face recognition (identifying who)
- Implemented a full pipeline from dataset extraction to model training to identity prediction
- Used confidence thresholds to filter uncertain predictions
- Automated CSV attendance report generation using Pandas with timestamps

---

## Future Improvements

- Add real-time webcam face recognition for live attendance marking
- Connect to a MySQL database to store and retrieve attendance records
- Build a simple web interface using Streamlit for non-technical users
- Improve accuracy with a larger dataset and deep learning based recognition (FaceNet or DeepFace)
- Add support for registering new people without retraining the entire model

---
## About

**Varanasi Gayathri Vijaya Sri Lalitha**
