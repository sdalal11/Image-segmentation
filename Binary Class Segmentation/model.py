from tensorflow.keras.models import load_model as lm
import pandas as pd
import numpy as np
import cv2

class_dict_df = pd.DataFrame([(255, 255, 255),
                  (0, 0, 0)],
                ['building', 'background'])
class_dict_df.columns = ['r', 'g', 'b']

label_names= list(class_dict_df.index)
label_codes = []
r= np.asarray(class_dict_df.r)
g= np.asarray(class_dict_df.g)
b= np.asarray(class_dict_df.b)

for i in range(len(class_dict_df)):
    label_codes.append(tuple([r[i], g[i], b[i]]))

label_codes, label_names

code2id = {v:k for k,v in enumerate(label_codes)}
id2code = {k:v for k,v in enumerate(label_codes)}

name2id = {v:k for k,v in enumerate(label_names)}
id2name = {k:v for k,v in enumerate(label_names)}

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def load_model(model_path):
    model = lm(model_path, custom_objects = {"dice_coef": dice_coef})
    return model


def rgb_to_onehot(rgb_image, colormap = id2code):
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,) 
    encoded_image = np.zeros(shape, dtype=np.int8)
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    
    return encoded_image


def onehot_to_rgb(onehot, colormap = id2code):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) ) 
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)

def predict(img_path,model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (512,512)) / 255.0
    img = np.expand_dims(img, axis = 0)  
    mask = onehot_to_rgb(model.predict(img)[0], id2code)
    return mask