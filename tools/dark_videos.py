import os
import cv2
import numpy as np
from decord import VideoReader

mit_classes = ['whistling', 'playing_videogames', 'juggling', 'bubbling', 'erupting', 'spilling', 'dancing', 'ascending', 'balancing', 'drumming', 'turning', 'burning', 'storming', 'playing_music', 'raining', 'shouting', 'adult_male_speaking', 'rocking', 'driving', 'cheering', 'bouncing', 'talking', 'coughing', 'playing', 'rising', 'combusting', 'spinning', 'adult_female_singing', 'smoking', 'child_singing', 'descending', 'adult_male_singing', 'performing', 'clapping', 'floating', 'applauding', 'dropping', 'singing']
kinetics_classes = ['spelunking', 'spinning_poi', 'breathing_fire', 'riding_mechanical_bull', 'shooting_off_fireworks', 'headbanging', 'karaoke', 'playing_laser_tag', 'juggling_fire', 'silent_disco']

def dark_img(img, threshold = 0.877):
    YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    Y = YCrCb[:,:,0]
    # Determine whether image is bright or dimmed
    exp_in = 112 # Expected global average intensity 
    M,N = img.shape[:2]
    mean_in = np.sum(Y/(M*N)) 
    t = (mean_in - exp_in)/ exp_in
    
    # Check image
    if t < -threshold: # Dimmed Image
        return True
    else:
        return False

def dark_video(video, segments=8, threshold = 0.877):
    vr = VideoReader(video)
    seg = int(len(vr) / segments)
    sample_id = [seg*i+int(seg/2) for i in range(0, segments)]
    frames = vr.get_batch(sample_id).asnumpy()
    video_t = 0
    for i in range(len(sample_id)):
        img = frames[i]
        img_t = dark_img(img)
        video_t += img_t
    if video_t/segments >= threshold:
        return True
    else:
        return False

if __name__ == '__main__':
    video = 'PATH/video.mp4'
    print(dark_video(video))