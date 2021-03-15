import cv2
import numpy as np
from scipy import misc
from tensorflow.keras.preprocessing.image import array_to_img
from PIL import Image


FFMPEG_BIN="ffmpeg"
import subprocess as sp


def play_video(path,name):
    cap = cv2.VideoCapture(path)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow(name, frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def create_video(allframes,filepath,fps,downscale):
    dim = (allframes.shape[2], allframes.shape[1])
    #print(size)
    fourcc=cv2.VideoWriter_fourcc(*'XVID')#('M','J','P','G')
    video = cv2.VideoWriter(filepath,fourcc , fps, dim)

    for frame in allframes:
        #print(frame.shape)
        if downscale==True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        video.write(frame)

    video.release()
    cv2.destroyAllWindows()

def sample_scaled_frames(video_path,stride,scale_factor):
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print('Can not open %s.' % video_path)
        pass
    frames = []
    frame_count = 0
    w=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    old_dim=(int(w*2),int(h*2))
    width = int(w /scale_factor)
    height = int(h /scale_factor)
    dim = (width, height)
    frames_root=[]

    while True:
        ret, frame = cap.read()

        if ret is False:
            break
        frame = frame[:, :, ::-1]
        frames_root.append(frame)
        if (scale_factor!=1):
            resized = resize_image_by_pil(frame, scale_factor, resampling_method="bicubic")
            frames.append(resized)
        else:
            frames.append(frame)
        frame_count += 1


    indices = list(range(0, frame_count , stride))
    #print(indices)
    frames = np.array(frames)
    frames_root = np.array(frames_root)
    frame_list = frames[indices]
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_root=frames_root[indices]

    return frame_list, frame_count,fps,dim,frames_root

def sample_scaled_frames2(video_path,stride,scale_factor):
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print('Can not open %s.' % video_path)
        pass
    frames = []
    frame_count = 0
    w=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(w /scale_factor)
    height = int(h /scale_factor)
    dim = (width, height)
    frames_root=[]

    while True:
        ret, frame = cap.read()

        if ret is False:
            break
        frame = frame[:, :, ::-1]
        frames_root.append(frame)
        if (scale_factor!=1):
            resized = resize_image_by_pil(frame, scale_factor, resampling_method="bicubic")
            frames.append(resized)
        else:
            frames.append(frame)
        frame_count += 1


    indices = list(range(8, frame_count - 7, stride))
    #print(indices)
    frames = np.array(frames)
    frames_root = np.array(frames_root)

    frame_list = frames[indices]
    frames_root=frames_root[indices]

    return frame_list, frame_count,fps,dim,frames_root


def resize_image_by_pil(image, scale, resampling_method="bicubic"):
    width, height = image.shape[1], image.shape[0]
    new_width = int(width / scale)
    new_height = int(height / scale)

    if resampling_method == "bicubic":
        method = Image.BICUBIC
    elif resampling_method == "bilinear":
        method = Image.BILINEAR
    elif resampling_method == "nearest":
        method = Image.NEAREST
    else:
        method = Image.LANCZOS

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # the image may has an alpha channel
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    else:
        image = Image.fromarray(image.reshape(height, width))
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
        image = image.reshape(new_height, new_width, 1)
    return image



def sample_clips(video_path,stride):
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print('Can not open %s.' % video_path)
        pass

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        frame = frame[:, :, ::-1]
        frames.append(frame)
        frame_count += 1

    indices = list(range(8, frame_count - 7, stride))

    frames = np.array(frames)
    clip_list = []
    for index in indices:
        clip_list.append(frames[index - 8: index + 8])
    clip_list = np.array(clip_list)
    return clip_list, frame_count

