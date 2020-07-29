#!/usr/bin/env python
# coding: utf-8

# ## General Python Stuff 

# In[3]:


import av
import io
import os
import cv2
import tqdm
import pafy
import pims
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyimagesearch.gradcam import GradCAM


from math import sqrt
from PIL import Image, ImageDraw, ImageFont
from hurry.filesize import size as sizer

import tensorflow as tf
from tensorflow import keras


# ## Keras Stuff 

# In[4]:


from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions


# ## Constants 

# In[5]:

STATS_FILE_PATH = 'testvideo.stats.csv'
TRAIN_DIR = "datasets/train/"
VAL_DIR = "datasets/test/"
BATCH_SIZE = 128
IMG_HEIGHT = 224
IMG_WIDTH  = 224


# ## Load Dataset

# In[6]:


# Tensorboard writer
logdir = "logs/MobileNetV2_{}".format(str(datetime.datetime.now()))
writer = tf.summary.create_file_writer(logdir)

# Data loader
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_data_gen = image_generator.flow_from_directory(directory=str(TRAIN_DIR),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH))
# val_data_gen = image_generator.flow_from_directory(directory=str(VAL_DIR),
#                                                      batch_size=BATCH_SIZE,
#                                                      shuffle=False,
#                                                      target_size=(IMG_HEIGHT, IMG_WIDTH))


# ## Make Model
#We add an FC and a GAP layer and choose the last conv layer for finetuning. First train with rmsprop at a higher learning rate then switch to SGD at a lower learning rate
# In[7]:


## create the base pre-trained model
base_model = MobileNetV2(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

#for i, layer in enumerate(base_model.layers):
#   print(i, layer.name)

#for layer in model.layers[:145]:
#   layer.trainable = False
for layer in model.layers[:]:
   layer.trainable = True

model.load_weights('./checkpoints/lslsc-mv2-retrain')

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy')


# In[6]:


# Train first with rmsprop at a higher learning rate
model.fit(x=train_data_gen,
          epochs=15,
          use_multiprocessing=True,
          workers=6)
model.save_weights('./checkpoints/lslsc-mv2-stage1-1pass')


# In[ ]:


model.load_weights('./checkpoints/lslsc-mv2-stage1-1pass')
# Second pass training with SGD at a lower learning rate
model.compile(optimizer=SGD(lr=0.00001, momentum=0.9), loss='categorical_crossentropy')
model.fit(x=train_data_gen,
          epochs=10,
          use_multiprocessing=True,
          workers=6)
model.save_weights('./checkpoints/lslsc-mv2-stage1')


# In[ ]:


model.load_weights('./checkpoints/lslsc-mv2-stage1')


# ## Cells to Test on Random Youtube Videos From A List

# In[ ]:


# # YouTube Testset API
# vmeta = pd.read_csv("youtube_tagset.csv", na_values="")
# vmeta


# In[ ]:


def infer_on_video(vname):
    media = "./media/{}".format(vname)
    container = pims.Video(media)
    sample = Image.fromarray(container[0])
    width, height = sample.width, sample.height
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter("./outputs/output_{}".format(vname), fourcc, 20.0, (width, height))  

    font = ImageFont.truetype('/amz/Ubuntu-R.ttf', size=32)
    color = ['rgb(189, 0, 0)', 'rgb(4, 92, 20)'] # white color

    #randomly sample frames in the video
    classnames = ["NTH", "TH"]
    
    oimgs = []
    x = []
    BS = 512
    for idx, frame in tqdm.tqdm(enumerate(container)):
        
        img  = Image.fromarray(frame)  # PIL/Pillow image
        oimg = img.copy()
        img  = img.resize((224, 224))
        arr  = np.asarray(img)  # numpy array
        oimgs.append(oimg)
        x.append(arr)

        if idx!=0 and idx%BS==0:
            #x = np.expand_dims(arr, axis=0)
            x = np.array(x)/255.

            preds = model.predict(x)
            for y in range(len(x)):
                
                p = preds[y]
                ox = oimgs[y]
                confidence = np.max(p)
                classid = np.argmax(p)
                classname = classnames[classid]

                draw = ImageDraw.Draw(ox)
                draw.text((10, 10), "{}: {:.0%}".format(classname, confidence), fill=color[classid], font=font)
                output = np.asarray(ox)  # numpy array
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                video.write(output)
            oimgs = []
            x = []
        
    video.release()


# In[ ]:


# vlinks = vmeta["Video link"].dropna()
# url = vlinks.sample(n=1).values[0]
# print(url)

urls = ["https://www.youtube.com/watch?v=C8Yfm0A6Oog", "https://www.youtube.com/watch?v=5NmIav-DttI", "https://www.youtube.com/watch?v=3Opht0RFaVA&feature=youtu.be", "https://www.youtube.com/watch?v=8ndgM1k1MTc"]

for url in urls:
    video = pafy.new(url)
    best = video.getbest()
    print("Title: {}\nLength: {}\nResolution: {}\nSize: {}".format(video.title, video.length, best.resolution, sizer(best.get_filesize())))
    best.download(quiet=False, filepath="./media/")
    infer_on_video("{}.mp4".format(video.title))


# In[ ]:





# ## Start From Here To Evaluate On Media in Disk 

# In[ ]:


def infer_and_plot_random(video):

    media = "./media/{}.mp4".format(video)
    container = pims.Video(media)
    ceil = 50

    figure = plt.figure(figsize=(10,20))
    
    #randomly sample frames in the video
    rframes = random.sample(range(0, len(container)), ceil)
    print("Choosing frames: ", rframes)
    classnames = ["NTH", "TH"]
    for idx, frame in tqdm.tqdm(enumerate(container[rframes])):
        if idx>=ceil:
            break
        img = Image.fromarray(frame)  # PIL/Pillow image
        img = img.resize((224, 224))
        arr = np.asarray(img)  # numpy array

        x = np.expand_dims(arr, axis=0)
        x = (x)/255.

        preds = model.predict(x)
        confidence = np.max(preds)
        classid = np.argmax(preds)
        classname = classnames[classid]
        
        cam = GradCAM(model, classid)
        heatmap = cam.compute_heatmap(x)
        heatmap = cv2.resize(heatmap, (224, 224))
        (heatmap, output) = cam.overlay_heatmap(heatmap, arr, alpha=0.5)
        
        plt.subplot(10, 5, idx+1, title="{} {:.0%}".format(classname, confidence))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(output, cmap=plt.cm.binary)

        #plt.subplot(20, 5, 2*idx+2)
        #plt.xticks([])
        #plt.yticks([])
        #plt.grid(False)
        #plt.imshow(output*255, cmap=plt.cm.binary)
    return figure


# In[ ]:


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


# ## Infer on Test Set and Dump Frames

# In[ ]:


for v in os.listdir('./media'):
    # Tensorboard writer
    logdir = "mediaeval/{}".format(v)
    valwriter = tf.summary.create_file_writer(logdir)
    print(v)
    figure = infer_and_plot_random(v[:-4])
    # Convert to image and log
    with valwriter.as_default():
      tf.summary.image("Demo MobileNetV2: {} New DT Balanced".format(v), plot_to_image(figure), step=0)


# In[ ]:


def infer_and_plot_random_frames(folder):

    container = np.array([cv2.imread(os.path.join(folder, x)) for x in os.listdir(folder)])
    if len(container)<50:
        ceil = len(container)
    else:
        ceil = 50

    figure = plt.figure(figsize=(10,20))
    
    #randomly sample frames in the video
    rframes = random.sample(range(0, len(container)), ceil)
    print("Choosing frames: ", rframes)
    classnames = ["NTH", "TH"]
    for idx, frame in tqdm.tqdm(enumerate(container[rframes])):
        if idx>=ceil:
            break
        img = Image.fromarray(frame)  # PIL/Pillow image
        img = img.resize((224, 224))
        arr = np.asarray(img)  # numpy array

        x = np.expand_dims(arr, axis=0)
        x = (x)/255.

        preds = model.predict(x)
        confidence = np.max(preds)
        classid = np.argmax(preds)
        classname = classnames[classid]
        
        cam = GradCAM(model, classid)
        heatmap = cam.compute_heatmap(x)
        heatmap = cv2.resize(heatmap, (224, 224))
        (heatmap, output) = cam.overlay_heatmap(heatmap, arr, alpha=0.5)
        
        plt.subplot(10, 5, idx+1, title="{} {:.0%}".format(classname, confidence))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(output, cmap=plt.cm.binary)

        #plt.subplot(20, 5, 2*idx+2)
        #plt.xticks([])
        #plt.yticks([])
        #plt.grid(False)
        #plt.imshow(output*255, cmap=plt.cm.binary)
    return figure


# ## Test on NTH Test Set Frames

# In[ ]:


# Make 5 random 10x5 plots

for i in range(5):
    # Tensorboard writer
    logdir = "testeval/{}".format("Test NTH")
    valwriter = tf.summary.create_file_writer(logdir)
    figure = infer_and_plot_random_frames("./datasets/test/nth")
    # Convert to image and log
    with valwriter.as_default():
      tf.summary.image("Demo MobileNetV2: {} {}".format("Test NTH", i), plot_to_image(figure), step=0)


# ## Test on TH Test Set Frames

# In[ ]:


# Make 5 random 10x5 plots

for i in range(5):
    # Tensorboard writer
    logdir = "testeval/{}".format("Test TH")
    valwriter = tf.summary.create_file_writer(logdir)
    figure = infer_and_plot_random_frames("./datasets/test/th")
    # Convert to image and log
    with valwriter.as_default():
      tf.summary.image("Demo MobileNetV2: {} {}".format("Test TH", i), plot_to_image(figure), step=0)


# In[ ]:


logdir = "testeval/{}".format("Test Misc")
valwriter = tf.summary.create_file_writer(logdir)
figure = infer_and_plot_random_frames("./datasets/misc")
# Convert to image and log
with valwriter.as_default():
  tf.summary.image("Demo MobileNetV2: {} {}".format("Test Misc", i), plot_to_image(figure), step=0)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix

num_of_test_samples = 5016
batch_size = 128

Y_pred = model.predict_generator(val_data_gen, num_of_test_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(val_data_gen.classes, y_pred))
print('Classification Report')
target_names = ['NTH', 'TH']
print(classification_report(val_data_gen.classes, y_pred, target_names=target_names))


# In[ ]:




