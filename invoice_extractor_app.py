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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Hide GitHub fork and edit buttons
hide_github_options = """
<style>
.stActionButton {
    display: none !important;
}
[data-testid="stToolbar"] {
    display: none !important;
}
header[data-testid="stHeader"] {
    display: none !important;
}
.stAppToolbar {
    display: none !important;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {
    display: none !important;
}
</style>
"""
st.markdown(hide_github_options, unsafe_allow_html=True)

# Configuration
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Invoice Extractor", layout="wide", initial_sidebar_state="expanded")
st.title("📄 Invoice Extractor (PDF/Image to Structured CSV)")

# Add configuration options in sidebar
st.sidebar.header("⚙️ Configuration")
max_workers = st.sidebar.slider("Parallel Workers", min_value=1, max_value=10, value=3, 
                               help="Number of parallel API calls")
delay_between_calls = st.sidebar.slider("Delay Between Calls (seconds)", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
max_pages_per_file = st.sidebar.number_input("Max Pages per PDF (0 = no limit)", min_value=0, value=0)

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

def call_gpt4_vision_with_retry(image_base64, filename, page_number, max_retries=3):
    for attempt in range(max_retries):
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
            json_match = re.search(r"\[.*\]", text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON array found in response")
            
            json_text = json_match.group(0)
            data = json.loads(json_text)
            
            for i, d in enumerate(data):
                d.update({
                    "source_file": filename,
                    "page_number": page_number,
                    "record_id": f"{filename}_{page_number}_{i+1}"
                })
            return data
            
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"❌ Failed after {max_retries} attempts on {filename} page {page_number}: {e}")
                return []
            time.sleep(1 * (attempt + 1))
    return []

def process_single_page(args):
    image_base64, filename, page_number, delay = args
    if delay > 0:
        time.sleep(delay)
    return call_gpt4_vision_with_retry(image_base64, filename, page_number)

def extract_pdf_pages(pdf_file, filename, max_pages=None):
    pdf_doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    total_pages = len(pdf_doc)
    if max_pages and max_pages > 0:
        total_pages = min(total_pages, max_pages)
    
    pages_data = []
    for i in range(total_pages):
        page = pdf_doc[i]
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes()))
        img_b64 = image_to_base64(img)
        pages_data.append((img_b64, filename, i + 1))
    
    pdf_doc.close()
    return pages_data, total_pages

def process_file_parallel(file_data, filename, max_workers, delay_between_calls):
    if not file_data:
        return []
    
    tasks = [(img_b64, fname, page_num, delay_between_calls) for img_b64, fname, page_num in file_data]
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(process_single_page, task): task for task in tasks}
        
        completed_tasks = 0
        total_tasks = len(tasks)
        progress_bar = st.progress(0, text=f"Processing {filename}...")
        
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.extend(result)
            except Exception as e:
                st.error(f"❌ Error processing {task[1]} page {task[2]}: {e}")
            
            completed_tasks += 1
            progress_bar.progress(completed_tasks / total_tasks, 
                                text=f"Processing {filename}: {completed_tasks}/{total_tasks} pages")
    
    progress_bar.empty()
    return results

# ---------- MAIN LOGIC ---------- #
if uploaded_files:
    st.markdown("---")
    st.subheader("🔄 Processing Uploaded Files")
    
    st.info(f"⚙️ Using {max_workers} parallel workers with {delay_between_calls}s delay")
    
    total_files = len(uploaded_files)
    overall_progress = st.progress(0, text="Starting...")
    
    combined_data = []
    processing_stats = {"total_pages": 0, "total_records": 0}
    
    start_time = time.time()

    for file_index, uploaded_file in enumerate(uploaded_files):
        st.subheader(f"📎 File: {uploaded_file.name}")
        file_start_time = time.time()
        
        file_ext = uploaded_file.name.lower().split(".")[-1]
        file_data = []

        if file_ext == "pdf":
            try:
                file_data, total_pages = extract_pdf_pages(uploaded_file, uploaded_file.name, max_pages_per_file)
                st.info(f"📄 Processing {total_pages} pages from PDF")
                processing_stats["total_pages"] += total_pages
            except Exception as e:
                st.error(f"❌ Error reading PDF {uploaded_file.name}: {e}")
                continue

        elif file_ext in ["png", "jpg", "jpeg"]:
            try:
                img = Image.open(uploaded_file)
                img_b64 = image_to_base64(img)
                file_data = [(img_b64, uploaded_file.name, 1)]
                processing_stats["total_pages"] += 1
            except Exception as e:
                st.error(f"❌ Error reading image {uploaded_file.name}: {e}")
                continue
        else:
            st.warning(f"⚠️ Unsupported file format: {uploaded_file.name}")
            continue

        if file_data:
            file_results = process_file_parallel(file_data, uploaded_file.name, max_workers, delay_between_calls)
            
            file_end_time = time.time()
            file_processing_time = file_end_time - file_start_time

            if file_results:
                df = pd.DataFrame(file_results)
                combined_data.extend(file_results)
                processing_stats["total_records"] += len(file_results)

                # Summary
                high_conf = df['extraction_quality'].str.lower().str.contains("high", na=False).sum()
                med_conf = df['extraction_quality'].str.lower().str.contains("medium", na=False).sum()
                low_conf = df['extraction_quality'].str.lower().str.contains("low", na=False).sum()

                with st.expander(f"📊 Summary for {uploaded_file.name}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Pages:** {len(df['page_number'].unique())}")
                        st.markdown(f"**Records:** {len(df)}")
                        st.markdown(f"**Time:** {file_processing_time:.1f}s")
                    with col2:
                        st.markdown(f"✅ High: {high_conf}")
                        st.markdown(f"⚠️ Medium: {med_conf}")
                        st.markdown(f"❌ Low: {low_conf}")

                st.success(f"✅ Extracted {len(df)} records in {file_processing_time:.1f}s")
                st.dataframe(df, use_container_width=True)

                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"📥 Download CSV", 
                    data=csv_data, 
                    file_name=f"{uploaded_file.name}_extracted.csv",
                    key=f"download_{file_index}"
                )
            else:
                st.warning(f"⚠️ No data extracted from {uploaded_file.name}")

        overall_progress.progress((file_index + 1) / total_files)

    # Final results
    total_time = time.time() - start_time

    if combined_data:
        st.markdown("---")
        st.subheader("📊 Final Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Files", total_files)
        with col2:
            st.metric("Pages", processing_stats["total_pages"])
        with col3:
            st.metric("Records", processing_stats["total_records"])
        with col4:
            st.metric("Time", f"{total_time:.1f}s")

        combined_df = pd.DataFrame(combined_data)
        st.dataframe(combined_df, use_container_width=True)
        
        combined_csv = combined_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download All Results", 
            data=combined_csv, 
            file_name="all_extracted_invoices.csv"
        )

else:
    st.info("📁 Upload PDF or Image files to begin")
    st.markdown("""
    ### 🚀 Features:
    - ⚡ **Parallel Processing** - Process multiple pages simultaneously
    - 📄 **No Limits** - Process entire PDFs
    - 🎛️ **Configurable** - Adjust settings in sidebar
    - 📊 **Real-time Stats** - Monitor processing progress
    - 💾 **Flexible Downloads** - Individual or combined CSV files
    """)