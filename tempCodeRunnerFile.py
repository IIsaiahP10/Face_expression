captured_emotions_List = []
# emotion_Score_List = []
# path1 = 'Known_Images/teacher'
# files = Path(path1).glob('*')
# for file in files:
#     fileImages = cv2.imread(str(file))
#     # using the FER trained model to detect an emotion in the image
#     emotion_detector = FER(mtcnn=True)
#     # Capture all the emotions on the image
#     captured_emotions = emotion_detector.detect_emotions(fileImages)
#     # Use the top Emotion() function to call for the dominant emotion in the image
#     dominant_Emotion, emotion_Score = emotion_detector.top_emotion(fileImages)
#     captured_emotions_List.append(dominant_Emotion)
#     emotion_Score_List.append(emotion_Score)
#     # To see the images being analyzed for their emotion and score, uncomment the lines below
#     #cv2.imshow(str(file), fileImages)
#     #cv2.waitKey()
#     #print(dominant_Emotion, emotion_Score)
# print('made it outside the loop')
# x = np.linspace(1,18,17)
# y = emotion_Score_List# time passed between the time each frame was taken

# plt.plot(x,y, label = 'Teachers Emotions Over Time')
# plt.show
# print('graphed')