# Road pothole  detection
### The YOLO v8 small was fined tuned on custom data (which include the bounding box coordinates for the pothole object in an images) for an 40 epoch of training time with a batch size of 8. 
#### Adam optimizer was used for maintaining stable training

Tools and technique used :
  Library: Ultralytics, opencv, streamlit, 
  Model : Yolov8s fine-tuned on custom road dataset with proper annotation
  Evaluation metric : Confusion matrix, mAP (Mean absolute precision), Precision recall curve
  
Deployment link : https://sajinamatya-road-pothole-detection-deploy-dfdscu.streamlit.app/

## System flowchart 
![image](https://github.com/user-attachments/assets/9e103a60-3b2d-4833-8861-01b2855e9fe9)

## Sample of data annotation using robo flow 
![image](https://github.com/user-attachments/assets/1ad0c0c7-8061-4984-8d88-b752723f00bd)


## Streamlit Deployment 
![image](https://github.com/user-attachments/assets/49344db0-a97d-47db-ba3a-d695da47751a)
