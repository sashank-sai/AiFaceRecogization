import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

camera_stream = None

def detect_faces_in_image():
    global selected_img
    global img_display

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image_path = filedialog.askopenfilename()

    if image_path:
        selected_img = cv2.imread(image_path)
        selected_img = cv2.resize(selected_img, (340, 380))

        grayscale_img = cv2.cvtColor(selected_img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_classifier.detectMultiScale(grayscale_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        face_count = len(faces_detected)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(selected_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        img_converted = cv2.cvtColor(selected_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_converted)
        tk_img = ImageTk.PhotoImage(pil_img)

        img_display.config(image=tk_img)
        img_display.image = tk_img

        result_label.config(text=f'Detected Faces: {face_count}', fg="#000", bg="#d6cadd")

def detect_faces_with_camera():
    global camera_stream

    if camera_stream is not None:
        camera_stream.release()

    camera_stream = cv2.VideoCapture(0)
    camera_stream.set(3, 320)
    camera_stream.set(4, 240)

    process_camera_frames()

def process_camera_frames():
    global camera_stream
    global img_display
    global root_window

    if camera_stream is not None:
        ret, live_frame = camera_stream.read()

        if ret:
            face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            gray_frame = cv2.cvtColor(live_frame, cv2.COLOR_BGR2GRAY)

            faces_detected = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            face_count = len(faces_detected)

            for (x, y, w, h) in faces_detected:
                cv2.rectangle(live_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            tk_frame = ImageTk.PhotoImage(pil_frame)

            img_display.config(image=tk_frame)
            img_display.image = tk_frame

            result_label.config(text=f'Faces Detected: {face_count}', fg="#990000", bg="#d6cadd")

            root_window.after(10, process_camera_frames)

root_window = tk.Tk()
root_window.title("Real-Time Face Detection")
root_window.geometry("800x700")
root_window.configure(bg="#f0f0f0")

header_label = tk.Label(root_window, text="Face Detection Application", font=("Arial", 40), fg="#333333", bg="#ffd700")
header_label.pack(pady=20)

open_image_btn = tk.Button(root_window, text="Select Image", command=detect_faces_in_image, font=("Arial", 16), fg="white", bg="#007acc", padx=10, pady=5)
open_image_btn.pack(pady=10)

camera_btn = tk.Button(root_window, text="Start Camera Detection", command=detect_faces_with_camera, font=("Arial", 16), fg="white", bg="#007acc", padx=10, pady=5)
camera_btn.pack(pady=10)

img_display = tk.Label(root_window, bg="#f0f0f0")
img_display.pack()

result_label = tk.Label(root_window, text='', font=('Arial', 22), fg="#333333", bg="#f0f0f0", padx=10, pady=5)
result_label.pack()

root_window.mainloop()
