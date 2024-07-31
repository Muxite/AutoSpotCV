# AutoSpotCV

### Main Goal: Create and demonstrate functional computer vision scripts with reasonable sensitivity and accuracy. 
### Short-term Goal: create a "point-and-capture" program for quick face detection+recognition that is usable without knowledge of computer vision
- A computer with a webcam could quickly capture training data (user tells it who its looking at), train itself with a single click, and switch to recogniton mode.
- This could be very useful for home security camera systems.
### Long-term Goal: Expand "point-and-capture" to recognize+detect anything the program is trained on (more than just faces). This will likely be processing/RAM intensive.
 - This could be used to train a CV system on the fly when in a new environment.

# Scripts

## File-based face recognition script + experiments (face_rec_for_files.py) (FINISHED) 
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
Uses Selenium Webdriver to download large numbers of images from Google Images based on a search term. Uses multithreading to improve performance (0.695 seconds per image at 10 threads and 50 images). Occaisionally double-downloads images. 

 
## File manager (face_rec_file_manager.py) (FINISHED) 
Splits image directories into training and testing datasets, or convert all images to fit in a specified size.
### folder format
 face_rec  
 ├── training  
 │   ├── person0  
 │   ├── person1  
 │   ├── person2  
 ├── testing  
 │   ├── person0  
 │   ├── person1  
 │   ├── person2  
