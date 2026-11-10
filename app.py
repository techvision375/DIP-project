import os, re, io, tempfile
from datetime import datetime
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract
import fitz

st.set_page_config(page_title="Insurance Document Intelligence", page_icon="📄", layout="wide")
st.title("📄 Insurance Document Intelligence")
st.caption("Upload an insurance policy / claim document → preprocess with DIP → OCR → extract key fields.")

# Tesseract can be configured using an environment variable on systems where it is not in PATH.
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def pdf_to_images(data, dpi=220):
    doc = fitz.open(stream=data, filetype="pdf")
    imgs=[]
    for page in doc:
        pix=page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False)
        imgs.append(Image.frombytes("RGB", [pix.width,pix.height], pix.samples))
    return imgs

def pil_to_cv(pil):
    arr=np.array(pil.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def preprocess(bgr, method, denoise, deskew):
    gray=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if denoise:
        gray=cv2.fastNlMeansDenoising(gray, None, 12, 7, 21)
    angle=0.0
    if deskew:
        # Estimate skew from foreground pixels; rotate only small angles to avoid accidental rotation.
        inv=cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        coords=np.column_stack(np.where(inv>0))
        if len(coords)>50:
            angle=cv2.minAreaRect(coords.astype(np.float32))[-1]
            if angle < -45: angle=-(90+angle)
            else: angle=-angle
            if abs(angle) <= 12:
                h,w=gray.shape
                M=cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
                gray=cv2.warpAffine(gray,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
            else: angle=0.0
    if method=="Adaptive threshold":
        processed=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,11)
    elif method=="Otsu threshold":
        processed=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    elif method=="Sharpen + grayscale":
        blur=cv2.GaussianBlur(gray,(0,0),2)
        processed=cv2.addWeighted(gray,1.7,blur,-0.7,0)
    else:
        processed=gray
    return gray,processed,angle

def image_bytes(pil, fmt="PNG"):
    buf=io.BytesIO(); pil.save(buf,format=fmt); return buf.getvalue()

def ocr(pil, lang):
    return pytesseract.image_to_string(pil, lang=lang, config="--psm 6").strip()

def extract_fields(text):
    patterns={
      "Policy Number": r"(?i)(?:policy\s*(?:no\.?|number|#)|pol\.?\s*no\.?)\s*[:\-]?\s*([A-Z0-9\/\-]{4,})",
      "Customer Name": r"(?i)(?:customer\s*name|insured\s*name|name\s*of\s*insured|name)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{2,})",
      "Claim Amount": r"(?i)(?:claim\s*amount|amount\s*claimed|claim\s*value)\s*[:\-]?\s*(?:₹|Rs\.?|INR)?\s*([0-9,]+(?:\.[0-9]{1,2})?)",
      "Claim Date": r"(?i)(?:claim\s*date|date\s*of\s*claim|date)\s*[:\-]?\s*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4}|[0-9]{4}[\/\-][0-9]{1,2}[\/\-][0-9]{1,2})",
    }
    out={}
    for label,pat in patterns.items():
        m=re.search(pat,text)
        out[label]=m.group(1).strip() if m else "Not found"
    return out

uploaded=st.file_uploader("Upload document", type=["png","jpg","jpeg","tif","tiff","bmp","pdf"])
colA,colB=st.columns([1,1])
with colA:
    method=st.selectbox("DIP processing method",["Adaptive threshold","Otsu threshold","Grayscale only","Sharpen + grayscale"])
    denoise=st.checkbox("Apply noise removal",value=True)
    deskew=st.checkbox("Correct small document tilt",value=True)
with colB:
    lang=st.selectbox("OCR language",["eng"],help="English Tesseract language data is required.")
    page_choice=st.selectbox("PDF page",["Process all pages","First page only"]) if uploaded and uploaded.name.lower().endswith(".pdf") else None

if uploaded:
    data=uploaded.getvalue()
    try:
        pages=pdf_to_images(data) if uploaded.name.lower().endswith(".pdf") else [Image.open(io.BytesIO(data)).convert("RGB")]
        if page_choice=="First page only": pages=pages[:1]
        st.info(f"Loaded {len(pages)} page(s).")
        if st.button("⚙️ Process document",type="primary"):
            all_text=[]; processed_images=[]
            for idx,pil in enumerate(pages,1):
                bgr=pil_to_cv(pil)
                gray,proc,angle=preprocess(bgr,method,denoise,deskew)
                proc_pil=Image.fromarray(proc)
                text=ocr(proc_pil,lang)
                all_text.append(f"--- Page {idx} ---\n{text}")
                processed_images.append(proc_pil)
                st.markdown(f"### Page {idx} (estimated correction: {angle:.2f}°)")
                x,y,z=st.columns(3)
                x.image(pil,caption="Original",use_container_width=True)
                y.image(gray,caption="Grayscale / denoised",use_container_width=True)
                z.image(proc,caption="Processed output",use_container_width=True)
                with st.expander(f"OCR text — Page {idx}",expanded=(idx==1)):
                    st.text_area(f"Page {idx} extracted text",text,height=180,key=f"ocr_{idx}")
            combined="\n\n".join(all_text)
            st.session_state["combined_text"]=combined
            st.session_state["processed_images"]=processed_images
            st.session_state["fields"]=extract_fields(combined)
    except Exception as e:
        st.error(f"Could not process this file: {e}")

if "combined_text" in st.session_state:
    st.markdown("---")
    st.header("📋 Extracted information")
    fields=st.session_state.get("fields",{})
    st.table({"Field":list(fields.keys()),"Value":list(fields.values())})
    st.download_button("⬇️ Download extracted text (.txt)",st.session_state["combined_text"],file_name="insurance_ocr.txt",mime="text/plain")
    if st.session_state.get("processed_images"):
        pdf=fitz.open()
        for im in st.session_state["processed_images"]:
            b= image_bytes(im,"PNG")
            page=pdf.new_page(width=im.width,height=im.height)
            page.insert_image(page.rect,stream=b)
        out=pdf.tobytes()
        st.download_button("⬇️ Download processed document (.pdf)",out,file_name="processed_insurance_document.pdf",mime="application/pdf")
    st.caption("Field extraction uses simple label-based patterns. Review OCR output and verify values before using them.")

with st.expander("Troubleshooting / setup"):
    st.markdown("""
    **If OCR fails:** install Tesseract OCR separately (it is not installed by `pip`).
    - macOS: `brew install tesseract`
    - Ubuntu/Debian: `sudo apt install tesseract-ocr`
    - Windows: install Tesseract and set `TESSERACT_CMD` to its executable path if it is not on PATH.

    For a quick test, upload `sample_documents/sample_claim.png` included in this ZIP.
    """)
