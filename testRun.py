from ultralytics import YOLO
import cv2


#Loading best weight from training
model = YOLO('best.pt') #make sure to use forward slashes for  absolute path
source = 'Test Data/TestImage1.jpg'
#shows video while predicting
if source.endswith('.mp4'):
    model.predict(
    source=source,
    conf=0.7,
    line_width=1,
    save=True,
    show=True,      
    save_crop=False,
    save_txt=False,
    show_labels=True,
    show_conf=True,
    classes=[0, 1],
    verbose = False
)
else: 
    results = model.predict(
    source=source,
    conf=0.7,
    line_width=1,
    save=True,
    show=False,      
    save_crop=False,
    save_txt=False,
    show_labels=True,
    show_conf=True,
    classes=[0, 1]
)
#doens't close cv2 window when showing images without key
    if source.endswith(('jpeg', 'jpg')):
        for r in results:
            img = r.plot()
            cv2.imshow("YOLO Prediction", img)
            cv2.waitKey(0)   # wait for any key
            cv2.destroyAllWindows()
