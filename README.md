# AutoSpotCV

### Main Goal: Create and demonstrate functional computer vision scripts with reasonable sensitivity and accuracy. 

### Short-term Goal: create a "fan-cam maker" program to track faces and crop videos to focus on a person.
- Track a face and create a running average of their recognition result to identify the path each face travels in a video, as well as when cameras cut from scene to scene.
- Create a box around the face, and crop that as the output video.
- Add interpolation (linear or bezier curve) to smoothen the motion.

### Med-term Goal: create a "point-and-capture" program for quick face detection+recognition that is usable without knowledge of computer vision
- A computer with a webcam could quickly capture training data (user tells it who its looking at), train itself with a single click, and switch to recogniton mode.
- This could be very useful for home security camera systems.
  
# Scripts

## File-based face recognition script + experiments (face_rec_for_filesV2.py, face_rec_for_filesV1.py)
face_rec_for_filesV2 uses LBPH face detection to train off a directory of faces. V2 follows OOP and uses a class called FaceRecognizer, and has autosave functions to store processed images and models to allow it to pick up where it left off. V2 also generates graphs and sensitivity numbers after a test, and saves them to a results directory. V2 LBPH leaves much to be desired in terms of sensitivity.
Both scripts are inside the face_rec folder. The image dataset was created with the Downloader script and has 2940 images of celebrities with differing ethnicities. V1 only contains functions, and is an old version.
### graphs produced by V2
![Brad Pitt_predictions](https://github.com/user-attachments/assets/f1ebf4ac-655d-4e7a-8a5f-dbc2ab26d770)

![Jensen Huang_predictions](https://github.com/user-attachments/assets/77735e56-46a6-43e8-b0d5-fc4b832d9c89)

![Yujin_predictions](https://github.com/user-attachments/assets/00aaa26a-9b1d-4428-af76-40d190e56474)

![Liz_predictions](https://github.com/user-attachments/assets/389d501d-8c2a-4c3b-a064-e5fba8966e1a)

![Morgan Freeman_predictions](https://github.com/user-attachments/assets/61b33911-11bb-4c0c-a678-9845bd364e88)

### Sensitivity Numbers for V2
Brad Pitt: sensitivity = 0.8519
Chris Hemsworth: sensitivity = 0.7381
Ed Sheeran: sensitivity = 0.6222
Gaeul: sensitivity = 0.2000
Jensen Huang: sensitivity = 0.8462
Jimin: sensitivity = 0.4468
Kim K: sensitivity = 0.6977
Leeseo: sensitivity = 0.5667
Leonardo DiCaprio: sensitivity = 0.6667
Lisa: sensitivity = 0.5714
Lisa Su: sensitivity = 0.5333
Liz: sensitivity = 0.6667
Morgan Freeman: sensitivity = 0.7500
Naheed Nenshi: sensitivity = 0.7143
Rei: sensitivity = 0.3548
Taylor Swift: sensitivity = 0.8125
Whitney Houston: sensitivity = 0.5938
Wonyoung: sensitivity = 0.5667
Yujin: sensitivity = 0.2927

### "test_one" mode used on Jensen Huang
![Screenshot 2024-08-21 201658](https://github.com/user-attachments/assets/429584c1-c24f-48ae-bba2-369fcc88d7b3)

## Image Downloader (downloader.py) (FINISHED) 
Uses Selenium Webdriver to download large numbers of images from Google Images based on a search term. Uses multithreading to improve performance (0.695 seconds per image at 10 threads and 50 images). Occasionally double-downloads images. Demo video below 👇👇👇 

[![Downloader Demo Video](https://img.youtube.com/vi/U-La3EGI8As/maxresdefault.jpg)](https://youtu.be/U-La3EGI8As)

## File manager (face_rec_file_manager.py) (FINISHED) 
Splits image directories into training and testing datasets, or convert all images to fit in a specified size.

