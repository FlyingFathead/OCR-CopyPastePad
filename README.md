# OCR-CopyPastePad

A simple GUI tool for OCR image-to-text processing that uses Python + `tkinter` + `pytesseract` + `python-opencv`.

![OCR-CopyPastePad screenshot](https://github.com/FlyingFathead/OCR-CopyPastePad/blob/main/OCR-CopyPastePad.jpg)

# About
With `OCR-CopyPastePad`, you can easily get your text-containing image files read into plaintext format with `pytesseract`'s Tesseract OCR, making it easy to then copy-paste the text data onward to i.e. a text editor, ChatGPT or some other AI LLM that you need to go text data through with. The idea is for the program to be as simple as possible when OCR conversion from image to text is needed in a given workflow.

# Features

- Uses `pytesseract` for OCR and `python-opencv` (`cv2`) to detect ROI's (= regions of interest) for higher accuracy.
- Easy Image Import: Load images directly from your computer or simply paste them using CTRL+V or Shift+Insert. Designed to be used i.e. in conjunction with the snippet tool in Windows (10, 11): `WinKey + Shift + S`
- Image Preprocessing: Before text extraction, images undergo preprocessing to enhance the accuracy of the OCR. This includes grayscale conversion, binary thresholding, and resizing.
- Intuitive Interface: The split-pane design allows users to view the original image side-by-side with the extracted text.
- Error Handling: Informative error messages guide users when issues arise, such as when non-image data is pasted.

# Install
1. Clone the repository
```
git clone https://github.com/FlyingFathead/OCR-CopyPastePad/
cd OCR-CopyPastePad/
```
2. Install the prerequisites
```
pip install -U requirements.txt
```
(or, manually: `pip install -U pytesseract Pillow python-opencv`)
3. Run the program
```
python OCR-CopyPastePad.py
```
or, for the non-`OpenCV` "lite" version:
```
python OCR-CopyPastePad_no_OpenCV_ROI.py
```

# Usage
1. Launch the OCR-CopyPastePad application (`python OCR-CopyPastePad.py`). You can also try out if your OCR results are better with the non-OpenCV version by running `python OCR-CopyPastePad_no_OpenCV_ROI.py`.
2. Load an image using the "Load Image" button or paste an image directly into the application
(in Windows you can use i.e. the snippet tool: `Shift + Winkey + S`).
4. If desired, use the "Detect Text Areas" button to see highlighted regions of text in the image.
5. The extracted text will automatically appear in the text pane on the right.

- Note that results may vary between source texts etc. -- in some cases, running the non-OpenCV version might actually yield more accurate results. OCR is a... thing.

# Changelog
- `v0.08`: Added the GUI option to invert image colors for better OCR accuracy.

# Todo
- [ ] Better implementation of the clipboard copy-paste-functionality
- [ ] User-drawable rectangle regions of interest on image

# About
- Code on GitHub: https://github.com/FlyingFathead/OCR-CopyPastePad/
- OCR-CopyPastePad is made by [FlyingFathead](https://github.com/FlyingFathead/) w/ ghost code by ChaosWhisperer
