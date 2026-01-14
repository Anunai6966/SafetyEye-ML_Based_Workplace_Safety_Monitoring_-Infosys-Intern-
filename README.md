# SafetyEye – ML-Based Workplace Safety Monitoring (Infosys Intern)

SafetyEye is an **ML-based workplace safety monitoring system** that leverages **computer vision and deep learning** to detect people and personal protective equipment (PPE) from live video feeds, identify safety compliance violations, and present actionable insights through an interactive analytics dashboard.

---

## Project Summary

Manual workplace safety monitoring is inefficient, error-prone, and difficult to scale.  
SafetyEye addresses this challenge by applying **YOLO-based object detection** to automate PPE compliance monitoring in real time.

The system detects people and safety equipment such as helmets and vests, identifies violations when required PPE is missing, logs safety events, and visualizes compliance trends for decision-making.

---

## Key Features

- Real-time object detection using a YOLO-based deep learning model  
- PPE compliance monitoring (Helmet / Vest detection)  
- Rule-based safety violation detection  
- Structured logging of safety events for analysis  
- Interactive dashboard with compliance analytics and trends  

---

## System Architecture (High Level)

``` text
Video Input (Webcam / Video File)
↓
YOLO-Based Object Detection
↓
PPE-to-Person Mapping Logic
↓
Violation Detection Engine
↓
Centralized Logging
↓
Streamlit Dashboard (Alerts & Analytics)
```


---

## Technology Stack

- **Programming Language:** Python  
- **Machine Learning:** YOLO (Ultralytics)  
- **Computer Vision:** OpenCV  
- **Dashboard & UI:** Streamlit  
- **Data Processing:** Pandas  
- **Visualization:** Plotly  

---

## Deep Learning & Model Details

- The project uses a **YOLO-based object detection model** fine-tuned on a PPE detection dataset.
- Detected classes include:
  - `person`
  - `helmet`
  - `no_helmet`
  - `vest`
  - `no_vest`
- The model outputs bounding boxes, class labels, and confidence scores for each video frame.
- Safety violations are identified using **rule-based logic** that associates detected PPE with individual persons.

> The model was fine-tuned from pretrained weights rather than trained from scratch, following standard industry practices.

---

## Analytics & Dashboard

The Streamlit-based dashboard provides:
- Recent safety violation logs  
- Overall PPE compliance overview  
- Violation trends over time  
- Breakdown of missing PPE types  

These analytics help stakeholders understand safety performance and identify recurring risk patterns.

---

## Project Structure

``` text
SafetyEye/
├── core/                    # Core ML & business logic
│ ├── detector.py            # YOLO inference wrapper
│ ├── rules.py               # PPE violation logic
│ ├── storage.py             # Centralized logging system
│ ├── emailer.py             # Email alert utility
│ ├── utils.py               # Helper & drawing functions
│ └── inference.py           # Inference flow control
│
├── pages/                   # Streamlit dashboard pages
│ ├── alerts.py
│ ├── analytics.py
│ └── init.py
│
├── models/
│ └── best.pt                # Trained YOLO model weights
│
├── app.py                   # Application entry point
├── ppe_data.yaml            # Dataset configuration
├── requirements.txt         # Dependencies
├── README.md
└── .gitignore
```

---

## Limitations

- Detection accuracy can be affected by lighting conditions and camera angles.  
- The system does not currently track individuals across frames.  
- Performance depends on the quality and diversity of the training dataset.  

---

## Future Enhancements

- Person tracking across frames  
- Support for additional PPE categories  
- Database-backed logging instead of CSV  
- SMS / WhatsApp alert integration  
- Deployment on edge devices or cloud platforms  

---

## How to Run the Project

### Create a virtual environment
python -m venv venv

### Activate the environment
venv\Scripts\activate        # Windows
source venv/bin/activate   # Linux / macOS

### Install dependencies
pip install -r requirements.txt

### Run the application
streamlit run app.py

---

## Conclusion

SafetyEye demonstrates the practical application of machine learning and computer vision to solve a real-world workplace safety problem by converting raw video data into actionable compliance insights through automated detection, logging, and analytics.

---

## Internship Note

This project was developed as part of an Infosys Internship for learning and demonstration purposes.
