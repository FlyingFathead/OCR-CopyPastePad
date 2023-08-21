# OCR-CopyPastePad //  https://github.com/FlyingFathead/OCR-CopyPastePad/
# v0.11 // Aug 2023 // FlyingFathead + ghost code by ChaosWhisperer

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageGrab, ImageOps
import pytesseract
import cv2
import numpy as np
import os
import urllib.request
import easyocr

# Current version
VERSION = "v0.11"

# reader = easyocr.Reader(['en'])  # Load once at the beginning

class OCRCopyPastePad:
    def __init__(self, root):
        self.root = root
        self.root.title(f"OCR-CopyPastePad {VERSION}")

        # Bind CTRL+V and Shift+Insert for paste events
        self.root.bind('<Control-v>', self.paste_image_event)
        self.root.bind('<Shift-Insert>', self.paste_image_event)

        # Supported languages for EasyOCR based on the official documentation
        self.languages = [
            'abq', 'ady', 'af', 'ang', 'ar', 'as', 'ava', 'az', 'be', 'bg',
            'bh', 'bho', 'bn', 'bs', 'ch_sim', 'ch_tra', 'che', 'cs', 'cy',
            'da', 'dar', 'de', 'en', 'es', 'et', 'fa', 'fr', 'ga', 'gom', 'hi',
            'hr', 'hu', 'id', 'inh', 'is', 'it', 'ja', 'kbd', 'kn', 'ko', 'ku',
            'la', 'lbe', 'lez', 'lt', 'lv', 'mah', 'mai', 'mi', 'mn', 'mr', 'ms',
            'mt', 'ne', 'new', 'nl', 'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'ru',
            'rs_cyrillic', 'rs_latin', 'sck', 'sk', 'sl', 'sq', 'sv', 'sw', 'ta',
            'tab', 'te', 'th', 'tjk', 'tl', 'tr', 'ug', 'uk', 'ur', 'uz', 'vi'
        ]

        # Create GUI components
        self.create_widgets()

        # Language selection
        self.language_var = tk.StringVar(self.root)
        self.language_var.set('en')  # default value
        self.language_dropdown = tk.OptionMenu(self.root, self.language_var, *self.languages)
        self.language_dropdown.pack(pady=10)
        self.language_label = tk.Label(self.root, text="Select OCR Language:")
        self.language_label.pack(pady=10, before=self.language_dropdown)
        
        self.reader = easyocr.Reader([self.language_var.get()])  # Load once at the beginning

        # Bind the language dropdown to the change handler
        self.language_var.trace_add('write', self.handle_language_change)

    def handle_language_change(self, *args):
        """Re-initialize the EasyOCR reader with the new language."""
        self.reader = easyocr.Reader([self.language_var.get()])

    def create_widgets(self):
        # PanedWindow
        self.paned_window = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=1)

        # Image panel
        # self.image_label = tk.Label(self.paned_window, text="Image will be displayed here")
        # self.paned_window.add(self.image_label)

        # Image panel (using Canvas instead of Label)
        self.image_canvas = tk.Canvas(self.paned_window)
        self.paned_window.add(self.image_canvas)

        # Text panel
        self.text_area = tk.Text(self.paned_window, wrap=tk.WORD)
        self.paned_window.add(self.text_area)

        # Load image button
        self.load_button = tk.Button(self.root, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=10)

        # Detect text areas button
        self.detect_button = tk.Button(self.root, text="Text area detection OCR with EasyOCR", command=self.detect_text_areas_and_ocr)
        self.detect_button.pack(pady=10)

        # Invert colors button
        self.invert_button = tk.Button(self.root, text="Invert Colors", command=self.invert_colors)
        self.invert_button.pack(pady=10)

        # Area selection
        self.select_area_button = tk.Button(self.root, text="Select Area", command=self.activate_select_mode)
        self.select_area_button.pack(pady=10)

    # Rectangle-drawing mode
    def activate_select_mode(self):
        self.image_label.bind("<Button-1>", self.start_rect)
        self.image_label.bind("<B1-Motion>", self.draw_rect)
        self.image_label.bind("<ButtonRelease-1>", self.end_rect)
        self.start_x = None
        self.start_y = None
        self.rect_id = None

    def start_rect(self, event):
        # Store starting point
        self.start_x = self.image_label.canvasx(event.x)
        self.start_y = self.image_label.canvasy(event.y)

    def draw_rect(self, event):
        # Update rectangle as mouse is dragged
        if not self.rect_id:
            self.rect_id = self.image_label.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')
        else:
            self.image_label.coords(self.rect_id, self.start_x, self.start_y, self.image_label.canvasx(event.x), self.image_label.canvasy(event.y))

    def end_rect(self, event):
        # Finalize rectangle and process the selected area
        self.image_label.coords(self.rect_id, self.start_x, self.start_y, self.image_label.canvasx(event.x), self.image_label.canvasy(event.y))
        self.process_selected_area()

    def process_selected_area(self):
        # Get rectangle coordinates
        coords = self.image_label.coords(self.rect_id)
        roi = self.image.crop((coords[0], coords[1], coords[2], coords[3]))

        # OCR the selected area
        ocr_text = pytesseract.image_to_string(roi)
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, ocr_text)

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
        
        # Upscale the image
        upscale_factor = 3  # You can adjust this factor as needed
        upscaled = cv2.resize(gray, (gray.shape[1] * upscale_factor, gray.shape[0] * upscale_factor))
        
        # Remove salt-and-pepper noise
        denoised = cv2.medianBlur(upscaled, 3)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Dilation with a modified kernel size
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        # Convert back to PIL Image
        image = Image.fromarray(dilated)
        
        return image

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
            # First, try to get the image directly
            clipboard_content = ImageGrab.grabclipboard()
            
            if isinstance(clipboard_content, Image.Image):
                self.image = clipboard_content
                self.process_image(clipboard_content)
                return

            # Second, if clipboard_content has a 'name' attribute (like a file object), use it
            file_path_or_url = getattr(clipboard_content, 'name', None)  
            
            if file_path_or_url:
                if os.path.exists(file_path_or_url) and os.path.isfile(file_path_or_url):
                    # If it's a local file path
                    self.image = Image.open(file_path_or_url)
                    self.process_image(self.image)
                    return
                elif file_path_or_url.startswith("http"):
                    # If it's a URL
                    with urllib.request.urlopen(file_path_or_url) as response:
                        self.image = Image.open(response)
                        self.process_image(self.image)
                        return

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
        self.image_canvas.config(scrollregion=self.image_canvas.bbox(tk.ALL), width=display_image.width, height=display_image.height)
        self.image_canvas.delete("all")  # Remove previous images
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.image_canvas.image = photo

    def detect_text_areas_and_ocr(self):
        # Use easyocr for text detection
        results = self.reader.readtext(np.array(self.image))
        
        # Image to draw bounding boxes on
        annotated_image = np.array(self.image)
        
        # Sort results based on vertical position, and then by horizontal position
        sorted_results = sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))
        
        # Helper functions
        def is_horizontally_close(box1, box2, threshold=50):
            _, _, right1, _ = box1
            left2, _, _, _ = box2
            return (left2[0] - right1[0]) < threshold

        # Group adjacent bounding boxes and merge their texts
        grouped_texts = []
        current_group = [sorted_results[0][1]]

        for i in range(1, len(sorted_results)):
            prev_bbox = sorted_results[i - 1][0]
            current_bbox = sorted_results[i][0]

            if is_horizontally_close(prev_bbox, current_bbox):
                current_group.append(sorted_results[i][1])
            else:
                grouped_texts.append(' '.join(current_group))
                current_group = [sorted_results[i][1]]

        # Add the last group if any
        if current_group:
            grouped_texts.append(' '.join(current_group))
        
        # Draw bounding boxes for visualization
        for (bbox, text, prob) in sorted_results:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            startX, startY, endX, endY = int(top_left[0]), int(top_left[1]), int(bottom_right[0]), int(bottom_right[1])
            cv2.rectangle(annotated_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Convert the annotated image back to PIL format
        annotated_image_pil = Image.fromarray(annotated_image)
        
        # Display the annotated image on the canvas
        photo = ImageTk.PhotoImage(annotated_image_pil)
        self.image_canvas.config(scrollregion=self.image_canvas.bbox(tk.ALL), width=annotated_image_pil.width, height=annotated_image_pil.height)
        self.image_canvas.delete("all")  # Remove previous images
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.image_canvas.image = photo
        
        # Display the grouped and merged texts
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, "\n\n".join(grouped_texts))
    
    def extract_boxes(self, scores, geometry):
        (numRows, numCols) = scores.shape[2:4]
        rects = []  # Initialization for the bounding box (rect) coordinates for text regions
        confidences = []  # Confidence scores for each bounding box

        # Loop over the rows
        for y in range(0, numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # Loop over the columns
            for x in range(0, numCols):
                # Ignore low confidence scores
                if scoresData[x] < 0.5:
                    continue

                # Compute the offset factor
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # Extract the rotation angle and compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # Use geometry volume to derive the width and height of the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # Compute start and end for the text region bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # Apply non-maxima suppression to suppress weak overlapping bounding boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        return boxes
    
def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    # If there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # If the bounding boxes are integers, convert them to floats
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes and sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probs)

    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap between the computed bounding box and the bounding box in the area list
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have overlap greater than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # Return only the bounding boxes that were picked using the integer data type
    return boxes[pick].astype("int")

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRCopyPastePad(root)
    root.mainloop()