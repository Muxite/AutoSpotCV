# AutoSpotCV

### Main Goal: Create and demonstrate functional computer vision scripts with reasonable sensitivity and accuracy. 
### Short-term Goal: create a "point-and-capture" program for quick face detection+recognition that is usable without knowledge of computer vision
- A computer with a webcam could quickly capture training data (user tells it who its looking at), train itself with a single click, and switch to recogniton mode.
- This could be very useful for home security camera systems.
### Long-term Goal: Expand "point-and-capture" to recognize+detect anything the program is trained on (more than just faces). This will likely be processing/RAM intensive.
 - This could be used to train a CV system on the fly when in a new environment.

# Scripts

## File-based face recognition script + experiments (face_rec_for_filesV2.py, face_rec_for_filesV1.py)
face_rec_for_filesV2 uses LBPH face detection to train off a directory of faces. V2 follows OOP and uses a class called FaceRecognizer, and has autosave functions to store processed images and models to allow it to pick up where it left off. V2 also generates graphs and sensitivity numbers after a test, and saves them to a results directory.
Both scripts are inside the face_rec folder. The image dataset was created with the Downloader script and has 2940 images of celebrities with differing ethnicities. V1 only contains functions, and is an old version. At a glance, V2 is significantly better than V1, and is easier to use.
V2 provides high sensitivity for all people:
![image](https://github.com/user-attachments/assets/178880c9-0420-4386-b418-a66593b1cd52)
Here are two results of V2:
![image](https://github.com/user-attachments/assets/be9d5abd-f999-4bd2-b9df-a416cd0df7b4) ![image](https://github.com/user-attachments/assets/f43df9f8-cd77-41ca-a506-a46186f808f9)



V1 Result:
3000 pictures artificially inflated to >10,000.
- Brad Pitt: sensitivity = 0.8333
- Chris Hemsworth: sensitivity = 0.7059
- Ed Sheeran: sensitivity = 0.8261
- Gaeul: sensitivity = 0.5385
- Jensen Huang: sensitivity = 0.8095
- Jimin: sensitivity = 0.4545
- Leeseo: sensitivity = 0.7143
- Leonardo DiCaprio: sensitivity = 0.8421
- Lisa: sensitivity = 0.6452
- Lisa Su: sensitivity = 0.8750
- Liz: sensitivity = 0.4400
- Morgan Freeman: sensitivity = 0.8571
- Naheed Nenshi: sensitivity = 0.6087
- Rei: sensitivity = 0.3750
- Taylor Swift: sensitivity = 0.7241
- Whitney Houston: sensitivity = 0.7333
- Wonyoung: sensitivity = 0.5000
- Yujin: sensitivity = 0.6500



## Image Downloader (downloader.py) (FINISHED) 
Uses Selenium Webdriver to download large numbers of images from Google Images based on a search term. Uses multithreading to improve performance (0.695 seconds per image at 10 threads and 50 images). Occasionally double-downloads images. Demo video below 👇👇👇 

[![Downloader Demo Video](https://img.youtube.com/vi/U-La3EGI8As/maxresdefault.jpg)](https://youtu.be/U-La3EGI8As)

## File manager (face_rec_file_manager.py) (FINISHED) 
Splits image directories into training and testing datasets, or convert all images to fit in a specified size.

