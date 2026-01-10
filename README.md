#YOLOv11s to detect moving traffic cars with and without sunroof along with dataset.

This project utilizes ultralytics <a href = 'https://docs.ultralytics.com/tasks/detect/#models'>YOLOv11s</a> model to train a custom data set of moving traffic CARS with sunroof and without sunroof to detect and classify them. that is split 90:10 into train and val sets. train dataset with 161 images and val dataset with 18 images
with a total instances of 487 split between ['Sunroof', 'NotSunroof'] classes.

<img src= 'runs\detect\train\labels.jpg' alt = 'Labels Picture' width = 600/>

The dataset is labelled using <a href = 'https://labelstud.io/'>Label-Studio</a>, an Open Source labeling platform.
The model is specifically trained on CARS (no trucks, pick-up or bikes are detected) moving left to right, this means cars moving opposite lane are not detected.Flip video feed horizontally before feeding for countries with right-hand drive.

<img src= 'runs\detect\predict15\1df405d8-20260108_111858.jpg' alt = 'Example Result'/>

Trained on CPU (~3hrs),I was able to reach validation mAP50-95 score of 85.4% with precision and recall of over 0.9.
<img src= 'runs\detect\train\BoxP_curve.png' alt = 'Precission Curve' width = 400/>
<img src= 'runs\detect\train\BoxR_curve.png' alt = 'Recall Curve' width = 400/>

The Test Data directory contains TestVideo1.mp4, shot at a different time than the train data set and therefore performs slightly worse due to poor lighting, and TestVideo2.mp4 shot at around the same of training set which the model excels at detecting.