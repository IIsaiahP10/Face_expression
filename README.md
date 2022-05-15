# CS-549 Machine Learning Group 8: Andres Maturin & Isaiah Pelayo
We chose to do the Face: Facial expression (emotion) analysis.

To run the facial expression analysis with our images, first fork or clone the repo, then run the file FaceExpressionDetection.py.
Our implementation of FER was inspired by https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch, and https://www.geeksforgeeks.org/facial-expression-recognizer-using-fer-using-deep-neural-net/ .

The crop.py file will take in the image frames that were extracted from the mp4 video file. You would pass an image frame as input and it will detect the faces from the image and then make seperate image files for each face that is detected. This python file was sued so that we could have the training data images.

The encoding.py file will take in Known_images directory as input and will get the encodings for each face and will make a dictionary that will contain the knwon encodings and known names. After each image is read and the process is compelete we make the dictionary to a pickle file to be used for other programs.

The encodings.py file will take in the output directory as input and will try to recognize each individual in each image frame. If the indiviudal is detected it will make a green box and display the name of that person. Once each person has their name and green box made, the program will wirte the edited imaged to theworking face matches directory.
