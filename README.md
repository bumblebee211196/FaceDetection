# FaceDetection

A simple face detection application using python and opencv dnn. For detail explanataion check out my [blog](https://bumblebee2196.netlify.app/face-detection-using-python-and-opencv/).

This was possible only due to the simple and clear explanation by Adrian Rosebrock's [blog](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/).

## How to run?

### 1. Create virtual environment

```shell
python3 -m venv my_env
source my_env/bin/activate
```

### 2. Install required dependencies

```shell
pip3 install -r requirements.txt
```

### 3. Execute

#### a. To detect faces from Images
```shell
python3 -m detect_faces_from_image -i /path/to/image/file.jpg
```

#### b. To detect faces from video

###### i. To detect faces from video files
```shell
python3 -m detect_faces_from_video -v /path/to/video/file.mov
```

###### ii. To detect faces realtime using camera
```shell
python3 -m detect_faces_from_video
```

