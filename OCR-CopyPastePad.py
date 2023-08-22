# OCR-CopyPastePad //  https://github.com/FlyingFathead/OCR-CopyPastePad/
# v0.145 // Aug 2023 // FlyingFathead + ghost code by ChaosWhisperer

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
VERSION = "v0.145"

# reader = easyocr.Reader(['en'])  # Load once at the beginning

class OCRCopyPastePad:
    def __init__(self, root):
        
        # Initialize the program
        self.root = root
        self.crop_rect_id = None
        
        # make sure no image is loaded when program starts up
        self.image_loaded = False
        
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
        
        self.reader = easyocr.Reader([self.language_var.get()])  # Load once at the beginning

        # Bind the language dropdown to the change handler
        self.language_var.trace_add('write', self.handle_language_change)

        # Right-hand panel
        self.right_panel = tk.Frame(self.root)
        self.paned_window.add(self.right_panel)
        self.right_panel.pack_propagate(True)  # Let the panel resize based on its content

        self.handle_resize()  # Set initial state of the right pane

    # resize the image to always fit the canvas
    def resize_and_display(self, image):
        # Resize the image to fit within the canvas dimensions
        max_width = self.image_canvas.winfo_width()
        max_height = self.image_canvas.winfo_height()
        resized_image = self.resize_image_for_display(image, max_width, max_height)

        # Display the resized image on the canvas
        photo = ImageTk.PhotoImage(resized_image)
        self.image_canvas.config(scrollregion=self.image_canvas.bbox(tk.ALL), width=resized_image.width, height=resized_image.height)
        self.image_canvas.delete("all")  # Remove previous images
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.image_canvas.image = photo

    # Crop tool -- 1/4: Activate the crop mode
    def activate_crop_mode(self):

        if not self.image_loaded:
            messagebox.showerror("Error", "No image loaded. Please load or paste an image first.")
            return

        self.crop_button.config(relief=tk.SUNKEN) # button graphics
        self.image_canvas.bind("<Button-1>", self.start_crop)
        self.image_canvas.bind("<B1-Motion>", self.draw_crop_rect)
        self.image_canvas.bind("<ButtonRelease-1>", self.end_crop)
        self.status_var.set("Crop mode activated. Draw a rectangle on the image area you want to OCR.")
        self.root.update_idletasks()  # Allow GUI to update
        
    # Crop tool -- 2/4: Start the rectangle drawing
    def start_crop(self, event):
        self.crop_start_x = self.image_canvas.canvasx(event.x)
        self.crop_start_y = self.image_canvas.canvasy(event.y)
        self.crop_rect_id = self.image_canvas.create_rectangle(self.crop_start_x, self.crop_start_y, self.crop_start_x, self.crop_start_y, outline='red')

    # Crop tool -- 3/4: Update rectangle while dragging
    def draw_crop_rect(self, event):
        self.image_canvas.coords(self.crop_rect_id, self.crop_start_x, self.crop_start_y, self.image_canvas.canvasx(event.x), self.image_canvas.canvasy(event.y))

    # Crop tool -- 4/4: Crop the image to selected region
    def end_crop(self, event):
        coords = self.image_canvas.coords(self.crop_rect_id)
        cropped_image = self.image.crop((coords[0], coords[1], coords[2], coords[3]))
        self.image = cropped_image
        self.process_image(cropped_image)

        self.resize_and_display(cropped_image)  # Use the new method here

        self.status_var.set("Image cropped to selected region.")
        self.root.update_idletasks()  # Allow GUI to update

        self.crop_button.config(relief=tk.RAISED) # button raised

    def handle_language_change(self, *args):
        """Re-initialize the EasyOCR reader with the new language."""
        self.reader = easyocr.Reader([self.language_var.get()])

    def handle_resize(self, event=None):
        # Calculate the new position of the sash (divider) based on the right panel's actual width
        self.right_panel.update_idletasks()  # Ensure right_panel width is updated
        right_panel_width = self.right_panel.winfo_width()
        self.paned_window.paneconfigure(self.right_panel, minsize=right_panel_width)  # Fix the right panel width

    def create_widgets(self):
        # PanedWindow
        self.paned_window = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=1)

        # Image panel (using Canvas instead of Label)
        self.image_canvas = tk.Canvas(self.paned_window)
        self.paned_window.add(self.image_canvas)

        # Text panel
        self.text_area = tk.Text(self.paned_window, wrap=tk.WORD)
        self.paned_window.add(self.text_area)

        # Right-hand panel
        self.right_panel = tk.Frame(self.root)
        self.paned_window.add(self.right_panel)

        # Load image button
        self.load_button = tk.Button(self.right_panel, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=10, fill=tk.X)

        # Detect text areas button with Pytesseract
        self.detect_tesseract_button = tk.Button(self.right_panel, text="OCR with Pytesseract", command=self.detect_with_tesseract)
        self.detect_tesseract_button.pack(pady=10, fill=tk.X)        

        # Detect text areas button
        self.detect_button = tk.Button(self.right_panel, text="Text area detection OCR with EasyOCR", command=self.detect_text_areas_and_ocr)
        self.detect_button.pack(pady=10, fill=tk.X)

        # Invert colors button
        self.invert_button = tk.Button(self.right_panel, text="Invert Colors", command=self.invert_colors)
        self.invert_button.pack(pady=10, fill=tk.X)

        # Area selection
        self.select_area_button = tk.Button(self.right_panel, text="Select Area", command=self.activate_select_mode)
        self.select_area_button.pack(pady=10, fill=tk.X)

        # Crop tool
        self.crop_button = tk.Button(self.right_panel, text="Crop Image", command=self.activate_crop_mode)
        self.crop_button.pack(pady=10, fill=tk.X)

        # Language selection
        self.language_var = tk.StringVar(self.right_panel)
        self.language_var.set('en')  # default value
        self.language_label = tk.Label(self.right_panel, text="Select OCR Language:")
        self.language_label.pack(pady=10, fill=tk.X)
        self.language_dropdown = tk.OptionMenu(self.right_panel, self.language_var, *self.languages)
        self.language_dropdown.pack(pady=10, fill=tk.X)

        # Status
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # Use pytesseract to detect and recognize text.
    def detect_with_tesseract(self):    
        self.status_var.set("Detecting text with Pytesseract...")
        self.root.update_idletasks()  # Allow GUI to update
        
        ocr_text = pytesseract.image_to_string(self.image)
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, ocr_text)
        
        self.status_var.set("Pytesseract OCR done.")
        self.root.update_idletasks()  # Allow GUI to update

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
        self.status_var.set("Inverting image colors...")
        self.root.update_idletasks()  # Allow GUI to update

        # Convert the image to RGB mode if it's not already
        if self.image.mode != 'RGB':
            self.image = self.image.convert('RGB')
            
        # Invert the colors of the image
        inverted_image = ImageOps.invert(self.image)
        self.image = inverted_image
        self.process_image(inverted_image)

        self.resize_and_display(inverted_image)  # Use the new method here

        self.status_var.set("Image colors inverted.")
        self.root.update_idletasks()  # Allow GUI to update

    def preprocess_image(self, image):
        self.status_var.set("Pre-processing image...")
        self.root.update_idletasks()  # Allow GUI to update

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

        self.status_var.set("Image pre-processing done.")
        self.root.update_idletasks()  # Allow GUI to update
        
        return image

    def load_image(self):
        self.status_var.set("Loading image...")   
        file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
        if not file_path:
            return

        self.image = Image.open(file_path)
        
        # Convert image to RGB if need be
        if self.image.mode != 'RGB':
            self.image = self.image.convert('RGB')
        
        self.process_image(self.image)

    def paste_image_event(self, event=None):
        self.paste_image()

    def paste_image(self):
        self.status_var.set("Image pasted in. Processing copy-pasted image...")
        self.root.update_idletasks()  # Allow GUI to update
        try:
            clipboard_content = ImageGrab.grabclipboard()

            # Convert the clipboard_content to RGB mode
            if isinstance(clipboard_content, Image.Image) and clipboard_content.mode != 'RGB':
                clipboard_content = clipboard_content.convert('RGB')

            # If the clipboard_content is a list, get the first item (assuming it's the path)
            if isinstance(clipboard_content, list) and len(clipboard_content) > 0:
                clipboard_content = clipboard_content[0]

            # Check if the clipboard_content is a string (i.e., a path)
            if isinstance(clipboard_content, str):
                if clipboard_content.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                    if os.path.exists(clipboard_content):
                        self.image = Image.open(clipboard_content)
                        if self.image.mode != 'RGB':
                            self.image = self.image.convert('RGB')
                        # self.process_image(self.image)
                        self.detect_with_tesseract()  # Directly call Tesseract OCR
                        self.display_image_on_canvas(self.image)  # Update the image display
                        self.status_var.set("Image processed from clipboard.")
                        self.root.update_idletasks()  # Allow GUI to update

                        return
                    else:
                        raise ValueError(f"Image path from clipboard does not exist: {clipboard_content}.")
                else:
                    raise ValueError(f"Unsupported content in the clipboard (string but not a known image path): {clipboard_content}.")

            # Check if the clipboard_content is an actual image
            elif isinstance(clipboard_content, Image.Image):
                self.image = clipboard_content
                if self.image.mode != 'RGB':
                    self.image = self.image.convert('RGB')
                # self.process_image(clipboard_content)
                self.detect_with_tesseract()  # Directly call Tesseract OCR
                self.display_image_on_canvas(self.image)  # Update the image display
                self.status_var.set("Image processed from clipboard.")
                self.root.update_idletasks()  # Allow GUI to update
                return

            else:
                raise ValueError(f"Unsupported content type in the clipboard: {type(clipboard_content)}. Content: {clipboard_content}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_image_on_canvas(self, image):
        """Display the provided image on the canvas."""
        # Resize the original image for display
        display_image = self.resize_image_for_display(image)

        # Display the resized original image
        photo = ImageTk.PhotoImage(display_image)
        self.image_canvas.config(scrollregion=self.image_canvas.bbox(tk.ALL), width=display_image.width, height=display_image.height)
        self.image_canvas.delete("all")  # Remove previous images
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.image_canvas.image = photo

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

        self.status_var.set("Processing image...")
        self.root.update_idletasks()  # Allow GUI to update

        self.image_loaded = True

        # Preprocess the image for better OCR accuracy
        processed_image = self.preprocess_image(image)

        # Check if the image should be inverted
        if self.should_invert(image):
            image = ImageOps.invert(image)

        # OCR the processed image
        ocr_text = pytesseract.image_to_string(processed_image)
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, ocr_text)

        # Resize the original image for display
        display_image = self.resize_image_for_display(image)

        # Clear cropping rectangle if it exists
        self.image_canvas.delete(self.crop_rect_id)

        # Display the resized original image
        photo = ImageTk.PhotoImage(display_image)
        self.image_canvas.config(scrollregion=self.image_canvas.bbox(tk.ALL), width=display_image.width, height=display_image.height)
        self.image_canvas.delete("all")  # Remove previous images
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.image_canvas.image = photo

        self.resize_and_display(image)  # Use the new method here        
        self.display_image_on_canvas(self.image)  # Update the image display

        self.status_var.set("Processing done.")
        self.root.update_idletasks()  # Allow GUI to update        

    def detect_text_areas_and_ocr(self):
        self.status_var.set("Detecting text areas with EasyOCR...")
        self.root.update_idletasks()  # Allow GUI to update

        # Check if the image should be inverted
        if self.should_invert(self.image):
            image = ImageOps.invert(self.image)

        # Use easyocr for text detection
        results = self.reader.readtext(np.array(self.image))

        # Sort results based on vertical position, then by horizontal position
        sorted_results = sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))

        # Group boxes by lines based on y-coordinate
        LINE_THRESHOLD = 10  # Adjust this based on your requirements
        lines = []
        current_line = [sorted_results[0]]

        for i in range(1, len(sorted_results)):
            if abs(sorted_results[i][0][0][1] - current_line[-1][0][0][1]) < LINE_THRESHOLD:
                current_line.append(sorted_results[i])
            else:
                lines.append(current_line)
                current_line = [sorted_results[i]]

        lines.append(current_line)  # Add the last line

        # Sort boxes within each line by x-coordinate
        for line in lines:
            line.sort(key=lambda r: r[0][0][0])

        # Flatten the sorted results
        sorted_results = [box for line in lines for box in line]

        # Continue with your logic to combine overlapping boxes
        combined_texts = []
        current_group = [sorted_results[0]]
        for i in range(1, len(sorted_results)):
            prev_bbox = current_group[-1][0]
            current_bbox = sorted_results[i][0]

            # Check if the boxes overlap vertically
            if prev_bbox[2][1] > current_bbox[0][1]:  
                current_group.append(sorted_results[i])
            else:
                # Merge the current group into a single bounding box
                combined_texts.append(
                    ([
                        min([box[0][0] for box, _, _ in current_group]),
                        min([box[0][1] for box, _, _ in current_group]),
                        max([box[2][0] for box, _, _ in current_group]),
                        max([box[2][1] for box, _, _ in current_group]),
                    ], ' '.join([text for _, text, _ in current_group]))
                )
                current_group = [sorted_results[i]]

        # Merge the last group if any
        if current_group:
            combined_texts.append(
                ([
                    min([box[0][0] for box, _, _ in current_group]),
                    min([box[0][1] for box, _, _ in current_group]),
                    max([box[2][0] for box, _, _ in current_group]),
                    max([box[2][1] for box, _, _ in current_group]),
                ], ' '.join([text for _, text, _ in current_group]))
            )

        # Image to draw bounding boxes on
        annotated_image = np.array(self.image)

        # Draw bounding boxes for visualization
        for (box, text) in combined_texts:
            startX, startY, endX, endY = box
            cv2.rectangle(annotated_image, (int(startX), int(startY)), (int(endX), int(endY)), (0, 255, 0), 2)

        # Convert the annotated image back to PIL format
        annotated_image_pil = Image.fromarray(annotated_image)

        # Display the annotated image on the canvas
        photo = ImageTk.PhotoImage(annotated_image_pil)
        self.image_canvas.config(scrollregion=self.image_canvas.bbox(tk.ALL), width=annotated_image_pil.width, height=annotated_image_pil.height)
        self.image_canvas.delete("all")  # Remove previous images
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.image_canvas.image = photo

        # Display the combined texts
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, "\n".join([text for _, text in combined_texts]))

        # Debugging: Print raw detections
        print(results)

        self.resize_and_display(annotated_image_pil)  # Use the new method here

        self.status_var.set("EasyOCR text detection done.")
        self.root.update_idletasks()  # Allow GUI to update

    # merge overlapping boxes
    def merge_overlapping_boxes(self, boxes):
        if not boxes:
            return []

        # Sort the boxes by their starting y-coordinate, then by their starting x-coordinate
        boxes = sorted(boxes, key=lambda x: (x[1], x[0]))

        merged_boxes = [boxes[0]]

        for i in range(1, len(boxes)):
            prev_box = merged_boxes[-1]
            curr_box = boxes[i]

            # Check for overlap; if the start of the current box is before the end of the previous box, they overlap
            if prev_box[2] >= curr_box[0] and prev_box[3] >= curr_box[1]:
                # Merge the current box into the previous box
                merged_box = (
                    min(prev_box[0], curr_box[0]),
                    min(prev_box[1], curr_box[1]),
                    max(prev_box[2], curr_box[2]),
                    max(prev_box[3], curr_box[3])
                )
                merged_boxes[-1] = merged_box
            else:
                # No overlap; add the current box as is
                merged_boxes.append(curr_box)

        return merged_boxes

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
        # boxes = non_max_suppression(np.array(rects), probs=confidences)

        # merging overlapping boxes
        boxes = self.merge_overlapping_boxes(rects)  # Note the `self.` prefix

        return boxes

    # check if image needs to be inverted
    def should_invert(self, image):
        """
        Determines if the image is primarily dark (e.g., white text on a black background).
        Returns True if the image should be inverted, False otherwise.
        """
        # Convert image to RGB if need be
        if image.mode != 'RGB':
            image = image.convert('RGB')
        grayscale = image.convert("L")

        grayscale = image.convert("L")
        mean_pixel = np.mean(np.array(grayscale))
        
        return mean_pixel < 128

    def non_max_suppression(self, boxes, probs=None, overlapThresh=0.3):
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
