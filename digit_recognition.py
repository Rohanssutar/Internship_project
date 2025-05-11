import tkinter as tk
from tkinter import ttk
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from PIL import Image, ImageDraw, ImageOps

# Load and train model
digits = datasets.load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# App class
class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Handwritten Digit Recognizer")
        master.geometry("600x600")
        master.configure(bg='lightgray')  # Set the background color of the main window

        ttk.Label(master, text=f"Model Accuracy: {accuracy:.2f}", font=("Helvetica", 12), background='lightgray').pack(pady=5)

        # Draw section
        ttk.Label(master, text="Draw your own digit below", font=("Helvetica", 12), background='lightgray').pack(pady=10)
        
        # Create a frame for the canvas to separate it visually
        self.canvas_frame = tk.Frame(master, bg='lightgray')
        self.canvas_frame.pack(pady=10)

        self.draw_canvas = tk.Canvas(self.canvas_frame, width=280, height=280, bg='white', highlightbackground='gray')
        self.draw_canvas.pack()
        self.draw_canvas.bind("<B1-Motion>", self.paint)

        self.image1 = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image1)

        # Change button colors
        ttk.Button(master, text="Predict Drawn Digit", command=self.predict_drawn_digit, 
                   style='TButton').pack(pady=5)
        ttk.Button(master, text="Clear Drawing", command=self.clear_canvas, 
                   style='TButton').pack(pady=5)

        self.result_label = ttk.Label(master, text="", font=("Helvetica", 14), background='lightgray')
        self.result_label.pack(pady=5)

        # Create a style for the buttons
        style = ttk.Style()
        style.configure('TButton', background='lightblue', foreground='black')

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.draw_canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.line([x1 + 10, y1 + 10, x2 + 10, y2 + 10], fill=0, width=20)

    def clear_canvas(self):
        self.draw_canvas.delete("all")
        self.image1 = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image1)
        self.result_label.config(text="")

    def predict_drawn_digit(self):
        # Invert colors: white background, black digit
        img = ImageOps.invert(self.image1)

        # Resize to 8x8
        img_resized = img.resize((8, 8), Image.LANCZOS)

        # Normalize pixel values similar to sklearn digits (0–16 scale)
        img_resized_np = np.array(img_resized)
        img_resized_np = img_resized_np / 255.0  # Normalize to [0, 1]
        img_resized_np = img_resized_np * 16  # Scale to [0, 16]
        img_resized_np = np.clip(img_resized_np, 0, 16)  # Ensure values stay in range
        input_data = img_resized_np.flatten().reshape(1, -1)

        # Predict
        prediction = model.predict(input_data)[0]
        self.result_label.config(text=f"Predicted Digit: {prediction}")

# Run the app
root = tk.Tk()
app = DigitRecognizerApp(root)
root.mainloop()