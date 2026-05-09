import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model                                                                               # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array                                                     # type: ignore
import numpy as np
import pyttsx3  # for text-to-speech
import rembg  # for background removal

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load the trained model
model = load_model('emf.h5')

def detect_and_route(image_path):
    img = load_img(image_path, target_size=(128, 128))  # Same as img_height and img_width
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0]
    conveyor_belts = ['Conveyor Belt C (Colourful)', 'Conveyor Belt B (Transparent)', 'Conveyor Belt A (Black)']
    conveyor_belt = conveyor_belts[np.argmax(prediction)]
    
    return conveyor_belt

def remove_background(input_path, output_path):
    with Image.open(input_path) as img:
        input_image = np.array(img)
    
    output_image = rembg.remove(input_image)
    output_img = Image.fromarray(output_image)
    output_img.save(output_path)

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kursi - Image Classifier")
        self.root.geometry("1000x600")
        # Add background image
        self.background_label = tk.Label(root, bg='DarkCyan')
        self.background_label.place(relwidth=1, relheight=4)
        
        # Add heading with border and background color
        self.heading_frame = tk.Frame(root, bg='LightSlateGray', bd=1, relief=tk.SOLID)
        self.heading_frame.pack(pady=20)
        
        self.heading = tk.Label(self.heading_frame, text="Object Detection Using Trained Module", font=('Helvetica', 16, 'bold'), bg='LightSlateGray', fg='Black')
        self.heading.pack(padx=10, pady=10)

        self.label = tk.Label(root, text="Upload an image to classify", font=('Helvetica', 14), fg='Black')
        self.label.pack(pady=10)

        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image, bg='blue', fg='white', font=('Helvetica', 12, 'bold'))
        self.upload_button.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=('Helvetica', 14), fg='DarkSlateGray')
        self.result_label.pack(pady=10)

        self.canvas = tk.Canvas(root, width=1000, height=500)
        self.canvas.pack()

        # Draw conveyor belts
        self.canvas.create_rectangle(50, 150, 750, 200, fill="Black")
        self.canvas.create_rectangle(50, 250, 750, 300, fill="SkyBlue", stipple="gray25")
        
        # Draw rainbow conveyor belt
        self.create_rainbow_belt(50, 350, 750, 400)
        
        # Centered text on conveyor belts
        self.canvas.create_text(400, 175, text="Conveyor Belt A (Black)", fill="white", font=('Helvetica', 12))
        self.canvas.create_text(400, 275, text="Conveyor Belt B (Transparent)", fill="MidnightBlue", font=('Helvetica', 12))
        self.canvas.create_text(400, 375, text="Conveyor Belt C (Colorful)", fill="MidnightBlue", font=('Helvetica', 12))

        # Draw the detection module (represented as a circle here)
        self.canvas.create_oval(350, 75, 450, 125, fill="CadetBlue")
        self.canvas.create_text(400, 50, text="Detection Module", font=('Helvetica', 12), fill="Black")

        # Draw boxes at the end of each conveyor belt
        self.box_coords = [(800, 150, 850, 200), (800, 250, 850, 300), (800, 350, 850, 400)]
        for coords in self.box_coords:
            self.canvas.create_rectangle(*coords, fill="CadetBlue")

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        # Conveyor labels
        self.conveyor_labels = ['Conveyor Belt A (Black)', 'Conveyor Belt B (Transparent)', 'Conveyor Belt C (Colourful)']

    def create_rainbow_belt(self, x1, y1, x2, y2):
        colors = ["DeepSkyBlue", "MediumSpringGreen", "Thistle", "LightSalmon", "PaleVioletRed", "Crimson"]
        num_colors = len(colors)
        belt_height = y2 - y1
        stripe_height = belt_height / num_colors

        for i, color in enumerate(colors):
            self.canvas.create_rectangle(x1, y1 + i * stripe_height, x2, y1 + (i + 1) * stripe_height, fill=color, outline=color)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            # Display original image first
            img = Image.open(file_path)
            img.thumbnail((50, 50))
            self.original_img = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.original_img)
            self.image_label.image = self.original_img
            self.result_label.config(text="Removing background...")
            # Announce background removal
            engine.say("Removing background from the image.")
            engine.runAndWait()

            # Remove background from selected image
            no_bg_image_path = "no_bg_image.png"
            self.root.after(2000, lambda: self.remove_background_and_classify(file_path, no_bg_image_path))

    def remove_background_and_classify(self, file_path, no_bg_image_path):
        remove_background(file_path, no_bg_image_path)
        
        # Load and display the image with background removed
        img = Image.open(no_bg_image_path)
        img.thumbnail((50, 50))
        self.img = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.img)
        self.image_label.image = self.img
        self.result_label.config(text="Classifying the object...")

        # Classify the image and get the conveyor belt
        self.conveyor_belt = detect_and_route(no_bg_image_path)
        self.result_label.config(text=f'The object should go to {self.conveyor_belt}.')

        # Animate the image moving to the detection module first, then to the correct belt, and finally to the box
        self.animate_movement(self.conveyor_belt)

    def animate_movement(self, conveyor_belt):
        belt_index = self.conveyor_labels.index(conveyor_belt)
        start_x, start_y = 50, 100
        detection_x, detection_y = 400, 100
        belt_x, belt_y = 400, 175 + belt_index * 100
        box_coords = self.box_coords[belt_index]
        box_x, box_y = (box_coords[0] + box_coords[2]) // 2, (box_coords[1] + box_coords[3]) // 2

        self.moving_image = self.canvas.create_image(start_x, start_y, image=self.img)
        self.move_image_to_detection(start_x, start_y, detection_x, detection_y, belt_x, belt_y, box_x, box_y)

    def move_image_to_detection(self, start_x, start_y, mid_x, mid_y, belt_x, belt_y, box_x, box_y, step=5):
        if abs(start_x - mid_x) < step and abs(start_y - mid_y) < step:
            # Announce the result after reaching the detection module
            engine.say(f'The object should go to {self.conveyor_belt}.')
            engine.runAndWait()
            
            # Now move to the belt
            self.move_image_to_belt(mid_x, mid_y, belt_x, belt_y, box_x, box_y, step)
        else:
            new_x = start_x + (mid_x - start_x) // step
            new_y = start_y + (mid_y - start_y) // step
            self.canvas.coords(self.moving_image, new_x, new_y)
            self.root.after(50, self.move_image_to_detection, new_x, new_y, mid_x, mid_y, belt_x, belt_y, box_x, box_y, step)

    def move_image_to_belt(self, start_x, start_y, belt_x, belt_y, box_x, box_y, step=5):
        if abs(start_x - belt_x) < step and abs(start_y - belt_y) < step:
            # Now move to the box
            self.move_image_to_box(belt_x, belt_y, box_x, box_y, step)
        else:
            new_x = start_x + (belt_x - start_x) // step
            new_y = start_y + (belt_y - start_y) // step
            self.canvas.coords(self.moving_image, new_x, new_y)
            self.root.after(50, self.move_image_to_belt, new_x, new_y, belt_x, belt_y, box_x, box_y, step)

    def move_image_to_box(self, start_x, start_y, end_x, end_y, step=5):
        if abs(start_x - end_x) < step and abs(start_y - end_y) < step:
            self.canvas.coords(self.moving_image, end_x, end_y)
        else:
            new_x = start_x + (end_x - start_x) // step
            new_y = start_y + (end_y - start_y) // step
            self.canvas.coords(self.moving_image, new_x, new_y)
            self.root.after(50, self.move_image_to_box, new_x, new_y, end_x, end_y, step)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
