#  Drone / Aerial Object Detection using Deformable DETR (COCO-91 Classes)

This project implements **Drone-based / Aerial Object Detection** using the **pretrained SenseTime Deformable DETR** model. The model is trained on the **COCO dataset (91 classes)** and is highly optimized for detecting objects from challenging aerial viewpoints such as drones, surveillance cameras, and high-elevation shots.

The system performs detection on:

- Streets & intersections  
- Crowded public areas  
- Vehicles & pedestrians  
- Animals  
- Highways  
- Marketplaces  

##  Model Details

### **Model:** `SenseTime/deformable-detr`  
### **Trained On:** COCO-2017 Dataset  
### **Classes:** **91 object categories**, including:

- Person  
- Car, Truck, Bus, Motorcycle  
- Bicycle  
- Animals (elephant, zebra, horse, cow, sheep, etc.)  
- Traffic lights, signs  
- Sports equipment  
- Household objects  
- Many more…  

The model uses:

- **Transformer Encoder–Decoder architecture**  
- **Multi-scale deformable attention**  
- **Improved accuracy on small / distant objects**  
- **Faster convergence than standard DETR**  

##  Features

- ✔️ Detects COCO-91 objects from **drone footage, public CCTV, aerial photos**  
- ✔️ High accuracy for pedestrians, vehicles, and animals  
- ✔️ Real-time performance with GPU  
- ✔️ Wide-angle scene understanding  
- ✔️ Streamlit UI for image upload & visualization  
- ✔️ Bounding boxes + confidence scores  

---

## 🪁 Example Outputs

### **1️⃣ Public Road Scene**
![image alt](https://github.com/pavankalyan-127/dronedetection_deformable_DETR/blob/main/drone_1.jpg?raw=true)
![image alt](https://raw.githubusercontent.com/pavankalyan-127/dronedetection_deformable_DETR/refs/heads/main/public_.jfif)
![image alt](https://github.com/pavankalyan-127/dronedetection_deformable_DETR/blob/main/public_2.2.jpg?raw=true)
### **2️⃣ Market Crowd Scene**
![image alt](https://raw.githubusercontent.com/pavankalyan-127/dronedetection_deformable_DETR/refs/heads/main/public.jfif)
![image alt](https://github.com/pavankalyan-127/dronedetection_deformable_DETR/blob/main/public_1.1.jpg?raw=true)
### **3️⃣ Animals Dataset (COCO Animals)**
![image alt](https://github.com/pavankalyan-127/dronedetection_deformable_DETR/blob/main/animals.jpg?raw=true)
![image alt](https://github.com/pavankalyan-127/dronedetection_deformable_DETR/blob/main/animals_2.2.jpg?raw=true)

//https://raw.githubusercontent.com/pavankalyan-127/dronedetection_deformable_DETR/refs/heads/main/public.jfif)

