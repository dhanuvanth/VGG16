import numpy as np
from cv2 import cv2
from tensorflow.keras.models import Sequential, save_model, load_model
from keras.preprocessing import image
import tensorflow as tf

cap = cv2.VideoCapture(0)

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

tf.device('/CPU:0')

model = load_model("DogOrCat.h5")

while True:
    ret,frame = cap.read()

    # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # frame = image.load_img(frame, target_size = (64, 64))
    frame1 = cv2.resize(frame,(64,64))
    frame1 = image.img_to_array(frame1)
    frame1 = np.expand_dims(frame1, axis = 0)

    if(model.predict(frame1) == 1):
        print("dog")
    else:
        print("cat")
    
    cv2.imshow("frame",frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()