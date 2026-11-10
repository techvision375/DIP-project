# Insurance Document Intelligence (DIP)

A small functional Streamlit application for insurance policy/claim document image processing.

## What works
- Upload PNG/JPG/TIFF/BMP or PDF documents (PDF pages are rendered).
- Grayscale conversion, optional noise removal and small-angle deskew.
- Adaptive thresholding, Otsu thresholding, grayscale-only, or sharpening.
- OCR using Tesseract.
- Simple label-based extraction for policy number, insured/customer name, claim amount and date.
- Preview original/intermediate/processed images.
- Download OCR text and processed pages as a PDF.
- Includes a sample claim image to test immediately.

> This is a functional prototype, not an insurer-grade claims system. OCR and field extraction can make mistakes; verify results manually and do not upload sensitive real customer documents to an untrusted environment.

## Setup

### 1. Install Tesseract (system OCR engine)
- macOS: `brew install tesseract`
- Ubuntu/Debian: `sudo apt update && sudo apt install tesseract-ocr`
- Windows: install Tesseract OCR and ensure its folder is on PATH. If needed set `TESSERACT_CMD` to the full executable path.

### 2. Create Python environment and install dependencies
```bash
cd insurance_document_intelligence
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run
```bash
streamlit run app.py
```
A browser tab opens (usually http://localhost:8501).

## Demo flow
1. Upload `sample_documents/sample_claim.png`.
2. Keep Adaptive threshold, noise removal and deskew enabled.
3. Click **Process document**.
4. Inspect intermediate images, OCR text and extracted fields.
5. Download processed PDF or OCR text.

## DIP experiments for report
Compare:
- Original OCR vs preprocessed OCR.
- Adaptive threshold vs Otsu vs grayscale.
- Noise removal ON/OFF.
- Deskew ON/OFF.

Report character/word accuracy on a small manually transcribed test set. This directly aligns with the course requirement for baseline-vs-improved comparisons and visible intermediate DIP results.
