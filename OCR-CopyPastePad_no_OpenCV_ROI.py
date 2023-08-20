# OCR-CopyPastePad //  https://github.com/FlyingFathead/OCR-CopyPastePad/
# v0.07 // Aug 2023 // FlyingFathead + ghost code by ChaosWhisperer

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageGrab
import pytesseract

VERSION = "v0.06"

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
        self.load_button.pack(pady=20)

    def preprocess_image(self, image):
        # Convert to grayscale
        image = image.convert('L')
        # Apply binary threshold
        image = image.point(lambda x: 0 if x < 128 else 255, '1')
        # Resize for better OCR accuracy
        image = image.resize((int(image.width * 1.5), int(image.height * 1.5)))
        return image

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
        if not file_path:
            return

        image = Image.open(file_path)
        self.process_image(image)

    def paste_image_event(self, event=None):
        self.paste_image()

    def paste_image(self):
        try:
            image = ImageGrab.grabclipboard()
            if isinstance(image, Image.Image):
                self.process_image(image)
            else:
                raise ValueError("No image data found in the clipboard.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def process_image(self, image):
        # Preprocess the image for better OCR accuracy
        processed_image = self.preprocess_image(image)

        # OCR the processed image
        ocr_text = pytesseract.image_to_string(processed_image)
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, ocr_text)

        # Display the original image
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRCopyPastePad(root)
    root.mainloop()