import streamlit as st
import openai
import pandas as pd
import base64
import io
import time
import re
import json
from PIL import Image
import fitz  

# with open(r"C:/Users/ymehra/Downloads/Cash Flow/KEY.json") as f:
#     openai.api_key = json.load(f)["OPENAI_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]
MAX_PAGES = 15
DELAY_BETWEEN_PAGES = 3

st.set_page_config(page_title="Invoice Extractor", layout="wide")
st.title("📄 Invoice Extractor (PDF/Image to CSV)")

uploaded_files = st.file_uploader(
    "Upload up to 50 PDF/Image files",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# ---------- UTILITIES ---------- #
def image_to_base64(image: Image.Image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def get_prompt():
    return '''Extract invoice/remittance/collection slip data from this document image.
Return JSON array in this format:
[
  {
    "remitter_name": "", "invoice_number": "", "invoice_date": "",
    "invoice_amount": "0.00", "discount_taken": "0.00", "payment_amount": "0.00",
    "payment_date": "", "reference": "", "document_number": "", "company": "",
    "payment_method": "", "section_type": "", "extraction_quality": "", "notes": ""
  }
]'''

def call_gpt4_vision(image_base64, filename, page_number):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": get_prompt()},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]}
            ],
            max_tokens=2048
        )
        text = response.choices[0].message.content
        json_text = re.search(r"\[.*\]", text, re.DOTALL).group(0)
        data = json.loads(json_text)
        for i, d in enumerate(data):
            d.update({
                "source_file": filename,
                "page_number": page_number,
                "record_id": f"{filename}_{page_number}_{i+1}"
            })
        return data
    except Exception as e:
        st.error(f"❌ GPT error on page {page_number}: {e}")
        return []

# ---------- MAIN LOGIC ---------- #
if uploaded_files:
    st.markdown("---")
    st.subheader("🔄 Processing Uploaded Files")
    total_files = len(uploaded_files)
    overall_progress = st.progress(0, text="Starting...")

    combined_data = []

    for file_index, uploaded_file in enumerate(uploaded_files):
        st.subheader(f"📎 File: {uploaded_file.name}")
        file_results = []
        file_ext = uploaded_file.name.lower().split(".")[-1]

        if file_ext == "pdf":
            pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            total_pages = min(len(pdf_doc), MAX_PAGES)
            page_progress = st.progress(0, text="Processing pages...")

            for i in range(total_pages):
                page = pdf_doc[i]
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes()))
                img_b64 = image_to_base64(img)
                result = call_gpt4_vision(img_b64, uploaded_file.name, i + 1)
                file_results.extend(result)
                page_progress.progress((i + 1) / total_pages, text=f"Page {i + 1}/{total_pages}")
                if i + 1 < total_pages:
                    time.sleep(DELAY_BETWEEN_PAGES)

        elif file_ext in ["png", "jpg", "jpeg"]:
            img = Image.open(uploaded_file)
            img_b64 = image_to_base64(img)
            result = call_gpt4_vision(img_b64, uploaded_file.name, 1)
            file_results.extend(result)

        else:
            st.warning(f"⚠️ Unsupported file format: {uploaded_file.name}")
            continue

        # Display summary and output
        if file_results:
            df = pd.DataFrame(file_results)
            combined_data.extend(file_results)

            # 🔍 Summary panel
            high_conf = df['extraction_quality'].str.lower().str.contains("high").sum()
            med_conf = df['extraction_quality'].str.lower().str.contains("medium").sum()
            low_conf = df['extraction_quality'].str.lower().str.contains("low").sum()

            with st.expander(f"📊 Summary for {uploaded_file.name}"):
                st.markdown(f"**Total Pages Processed:** {len(df['page_number'].unique())}")
                st.markdown(f"**Total Rows Extracted:** {len(df)}")
                st.markdown(f"✅ High Confidence Rows: {high_conf}")
                st.markdown(f"⚠️ Medium Confidence Rows: {med_conf}")
                st.markdown(f"❌ Low Confidence Rows: {low_conf}")

            st.success(f"✅ Extracted {len(df)} records")
            st.dataframe(df, use_container_width=True)

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV for this file", data=csv_data, file_name=f"{uploaded_file.name}_extracted.csv")
        else:
            st.warning(f"⚠️ No data extracted from {uploaded_file.name}")

        overall_progress.progress((file_index + 1) / total_files, text=f"Processed {file_index + 1}/{total_files} files")

    # Final combined CSV
    if combined_data:
        st.markdown("---")
        st.subheader("📊 Combined Extracted Data")
        combined_df = pd.DataFrame(combined_data)
        st.dataframe(combined_df, use_container_width=True)
        combined_csv = combined_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Combined CSV", data=combined_csv, file_name="all_extracted_invoices.csv")
else:
    st.info("Upload PDF or Image files to begin.")
