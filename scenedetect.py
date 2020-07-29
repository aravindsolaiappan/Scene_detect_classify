import os
import cv2
import datetime
import numpy as np
import argparse
import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

def parse_args():
    parser = argparse.ArgumentParser(description='Scene Detection and Classification')
    parser.add_argument("-i", "--input", help='Enter the video file', required=True)
    parser.add_argument('--checkpoint', help='Enter the checkpoint file', required=True)
    args = parser.parse_args()
    return args
def count_frames_manual(video):
    total = 0
    while True:
        (grabbed, frame) = video.read()
        if not grabbed:
            break
        total += 1
    return total
if __name__ == "__main__":
    args = parse_args()
    video_manager = VideoManager([args.input])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(base_timecode)
    frames = [scene[0].get_frames() for scene in scene_list]
    print(frames)
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(args.checkpoint)
         
    capture = cv2.VideoCapture(args.input)
    total = count_frames_manual(capture)
    print(total)
    capture=cv2.VideoCapture(args.input)
    idx = 0
    cnt = 0
    while True:
         ret, frame = capture.read()
         if not ret:
            break
         if (idx==frames[cnt]+10):
            cnt = (cnt + 1) % len(frames)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224,224))
            frame = np.expand_dims(frame, axis=0)
            frame = frame / 255.0
            result= "Talking" if np.argmax(model.predict(frame)) else "Non Talking"
            print("Unique frame id: ",idx,", involes ",result," people.")
            if result=="Talking":
                if frames[cnt-1]==frames[-1]:
                    diff=total-frames[cnt-1]
                else:
                    diff=frames[cnt]-frames[cnt-1]
                print(diff)
                dump_path = os.path.join("dumps", str(idx))
                if not os.path.exists(dump_path):
                    os.makedirs(dump_path)
                #for i in range(5):
                #    ret,frame=capture.read()                    
                for i in range(diff-5):
                    ret,frame=capture.read()
                    cv2.imwrite(os.path.join(dump_path, str(i)+".png"), frame)
         idx = idx + 1

    
