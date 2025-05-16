from tkinter import messagebox
import tkinter
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import cv2
import torch
from pathlib import Path

global filename, model
labels = ['Person', 'Crowd']
CONFIDENCE_THRESHOLD = 0.3
GREEN = (0, 255, 0)
global count

main = tkinter.Tk()
main.title("Using Existing CCTV Network for Crowd Management, Crime Prevention & Work Monitoring using AI & ML")
main.geometry("1300x1200")
def graph():
    graph_img = cv2.imread('yolov5_model/results.png')
    graph_img = cv2.resize(graph_img, (800, 600))
    cv2.imshow("YoloV5 Training Graph", graph_img)
    cv2.waitKey(0)
    
def loadModel():
    global model
    text.delete('1.0', END)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    text.insert(END, "YoloV5 Model Loaded\n")

def imageDetection():
    global model
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="images", title="Select Image",
                                          filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png"), ("All files", "*.*")))
    if not filename:
        text.insert(END, "No image file selected. Please select a file.\n\n")
        return

    image = cv2.imread(filename)
    if image is None:
        text.insert(END, "Failed to load image. Please try again.\n\n")
        return
    
    # Resize image if it is too large for display
    screen_width = main.winfo_screenwidth()  # Get screen width
    screen_height = main.winfo_screenheight()  # Get screen height
    image_height, image_width, _ = image.shape
    text.insert(END, f"Original Image Shape: {image_height}x{image_width}\n")
    
    max_display_height = screen_height - 100  # leave some space for window decorations
    max_display_width = screen_width - 100
    
    if image_height > max_display_height or image_width > max_display_width:
        # Calculate the ratio to scale down to max dimensions
        scaling_factor = min(max_display_height / image_height, max_display_width / image_width)
        new_width = int(image_width * scaling_factor)
        new_height = int(image_height * scaling_factor)
        image = cv2.resize(image, (new_width, new_height))
        text.insert(END, f"Resized Image Shape: {new_height}x{new_width}\n")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image)
    detections = results.xyxy[0].numpy()  # Bounding boxes
    
    count = 0
    for det in detections:
        if int(det[5]) == 0:  # Assuming class 0 is 'person'
            xmin, ymin, xmax, ymax = map(int, det[:4])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            count += 1
    
    cv2.putText(image, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Convert back to BGR for displaying in OpenCV window
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def videoDetection():
    global model
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Video", title="Select Video",
                                          filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")))
    if not filename:
        text.insert(END, "No video file selected. Please select a file.\n\n")
        return
    
    text.insert(END, filename + " loaded\n\n")
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        text.insert(END, "Failed to open video. Please try again.\n\n")
        return

    # Get screen dimensions and calculate max dimensions for display
    screen_width = main.winfo_screenwidth()
    screen_height = main.winfo_screenheight()
    max_display_height = screen_height - 200
    max_display_width = screen_width - 200

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    text.insert(END, f"Original Video Resolution: {frame_width}x{frame_height}\n")

    scaling_factor = min(max_display_height / frame_height, max_display_width / frame_width)
    new_width = int(frame_width * scaling_factor)
    new_height = int(frame_height * scaling_factor)
    text.insert(END, f"Resized Video Resolution: {new_width}x{new_height}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (new_width, new_height))  # Resize frame
        results = model(frame)
        detections = results.xyxy[0].numpy()  # Bounding boxes

        count = 0
        for det in detections:
            if int(det[5]) == 0:  # Assuming class 0 is 'person'
                xmin, ymin, xmax, ymax = map(int, det[:4])
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                count += 1

        cv2.putText(frame, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Video Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


font = ('times', 16, 'bold')
title = Label(main, text='Using Existing CCTV Network for Crowd Management, Crime Prevention & Work Monitoring using AI & ML')
title.config(bg='greenyellow', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)

font1 = ('times', 13, 'bold')
loadButton = Button(main, text="Generate & Load YoloV5 Model", command=loadModel)
loadButton.place(x=50,y=550)
loadButton.config(font=font1)  

imageButton = Button(main, text="Crowd Management from Images", command=imageDetection)
imageButton.place(x=330,y=550)
imageButton.config(font=font1) 

videoButton = Button(main, text="Crowd Management from Videos", command=videoDetection)
videoButton.place(x=630,y=550)
videoButton.config(font=font1)

graphButton = Button(main, text="YoloV5 Training Graph", command=graph)
graphButton.place(x=50,y=600)
graphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=main.destroy)
exitButton.place(x=330,y=600)
exitButton.config(font=font1) 

main.config(bg='LightSkyBlue')
main.mainloop()
