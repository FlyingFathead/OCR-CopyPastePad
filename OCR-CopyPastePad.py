# OCR-CopyPastePad //  https://github.com/FlyingFathead/OCR-CopyPastePad/
# v0.08 // Aug 2023 // FlyingFathead + ghost code by ChaosWhisperer

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageGrab, ImageOps
import pytesseract
import cv2
import numpy as np

VERSION = "v0.08"  # Updated version

class OCRCopyPastePad:
    def __init__(self, root):
        self.root = root
        self.root.title(f"OCR-CopyPastePad {VERSION}")

        # Bind CTRL+V and Shift+Insert for paste events
        self.root.bind('<Control-v>', self.paste_image_event)
        self.root.bind('<Shift-Insert>', self.paste_image_event)

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        # PanedWindow
        self.paned_window = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=1)

        # Image panel
        self.image_label = tk.Label(self.paned_window, text="Image will be displayed here")
        self.paned_window.add(self.image_label)

        # Text panel
        self.text_area = tk.Text(self.paned_window, wrap=tk.WORD)
        self.paned_window.add(self.text_area)

        # Load image button
        self.load_button = tk.Button(self.root, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=10)

        # Detect text areas button
        self.detect_button = tk.Button(self.root, text="Detect Text Areas", command=self.detect_text_areas_and_ocr)
        self.detect_button.pack(pady=10)

        # Invert colors button
        self.invert_button = tk.Button(self.root, text="Invert Colors", command=self.invert_colors)
        self.invert_button.pack(pady=10)

    def invert_colors(self):
        # Convert the image to RGB mode if it's not already
        if self.image.mode != 'RGB':
            self.image = self.image.convert('RGB')
            
        # Invert the colors of the image
        inverted_image = ImageOps.invert(self.image)
        self.image = inverted_image
        self.process_image(inverted_image)

    def preprocess_image(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Convert back to PIL Image
        image = Image.fromarray(thresh)
        
        # Resize for better OCR accuracy
        image = image.resize((int(image.width * 1.5), int(image.height * 1.5)))
        
        return image

    """ def preprocess_image(self, image):
        # Convert to grayscale
        image = image.convert('L')
        # Apply binary threshold
        image = image.point(lambda x: 0 if x < 128 else 255, '1')
        # Resize for better OCR accuracy
        image = image.resize((int(image.width * 1.5), int(image.height * 1.5)))
        return image """

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
        if not file_path:
            return

        self.image = Image.open(file_path)
        self.process_image(self.image)

    def paste_image_event(self, event=None):
        self.paste_image()

    def paste_image(self):
        try:
            image = ImageGrab.grabclipboard()
            if isinstance(image, Image.Image):
                self.image = image
                self.process_image(image)
            else:
                raise ValueError("No image data found in the clipboard.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def resize_image_for_display(self, image, max_width=500, max_height=400):
        """Resize the image to fit within the specified dimensions."""
        width, height = image.size
        aspect_ratio = width / height
        if width > max_width:
            width = max_width
            height = int(width / aspect_ratio)
        if height > max_height:
            height = max_height
            width = int(height * aspect_ratio)
        return image.resize((width, height))

    def process_image(self, image):
        # Preprocess the image for better OCR accuracy
        processed_image = self.preprocess_image(image)

        # OCR the processed image
        ocr_text = pytesseract.image_to_string(processed_image)
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, ocr_text)

        # Resize the original image for display
        display_image = self.resize_image_for_display(image)

        # Display the resized original image
        photo = ImageTk.PhotoImage(display_image)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo

    def detect_text_areas_and_ocr(self):
        # Convert the image to grayscale for processing
        gray = cv2.cvtColor(cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        
        # Use OpenCV to detect text areas
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area to filter out small noise
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        # Draw rectangles around detected text areas
        annotated_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
        
        # Convert the annotated image back to PIL format
        annotated_image_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        
        ocr_results = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = self.image.crop((x, y, x + w, y + h))
            ocr_text = pytesseract.image_to_string(roi)
            ocr_results.append(ocr_text)
        
        # Append the OCR results to the text area
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, "\n\n".join(ocr_results))
        
        # Display the annotated image
        photo = ImageTk.PhotoImage(annotated_image_pil)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRCopyPastePad(root)
    root.mainloop()
