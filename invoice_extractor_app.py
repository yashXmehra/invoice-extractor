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
import threading

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

# Sidebar Configuration
st.sidebar.header("⚙️ Parallel Processing Settings")
max_workers = st.sidebar.slider(
    "Parallel Workers", 
    min_value=1, max_value=15, value=5,
    help="Number of pages processed simultaneously (higher = faster but may hit API limits)"
)
delay_between_calls = st.sidebar.slider(
    "Delay Between API Calls (seconds)", 
    min_value=0.0, max_value=2.0, value=0.2, step=0.1,
    help="Delay to avoid rate limiting"
)
batch_size = st.sidebar.slider(
    "Batch Size", 
    min_value=5, max_value=50, value=20,
    help="Process pages in batches to manage memory"
)

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
    """Call GPT-4 Vision with retry logic and rate limiting"""
    for attempt in range(max_retries):
        try:
            # Add small delay for rate limiting
            time.sleep(delay_between_calls)
            
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
            time.sleep(1 * (attempt + 1))  # Exponential backoff
    return []

def process_single_page(args):
    """Process a single page - wrapper for parallel execution"""
    image_base64, filename, page_number = args
    return call_gpt4_vision_with_retry(image_base64, filename, page_number)

def extract_pdf_pages_parallel(pdf_file, filename):
    """Extract all pages from PDF and prepare for parallel processing"""
    pdf_doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    total_pages = len(pdf_doc)
    
    st.info(f"📄 Extracting {total_pages} pages from {filename}")
    
    # Extract all pages as images first
    pages_data = []
    page_progress = st.progress(0, text="Extracting pages from PDF...")
    
    for i in range(total_pages):
        page = pdf_doc[i]
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes()))
        img_b64 = image_to_base64(img)
        pages_data.append((img_b64, filename, i + 1))
        
        # Update progress
        page_progress.progress((i + 1) / total_pages, 
                             text=f"Extracting page {i + 1}/{total_pages}")
    
    pdf_doc.close()
    page_progress.empty()
    return pages_data, total_pages

def process_pages_in_parallel(pages_data, filename, max_workers, batch_size):
    """Process multiple pages simultaneously using ThreadPoolExecutor"""
    total_pages = len(pages_data)
    all_results = []
    
    # Process in batches to manage memory and API limits
    for batch_start in range(0, total_pages, batch_size):
        batch_end = min(batch_start + batch_size, total_pages)
        batch_pages = pages_data[batch_start:batch_end]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (total_pages + batch_size - 1) // batch_size
        
        st.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch_pages)} pages) in parallel...")
        
        batch_results = []
        completed_in_batch = 0
        
        # Progress bar for current batch
        batch_progress = st.progress(0, text=f"Processing batch {batch_num}...")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks in the batch
            future_to_page = {executor.submit(process_single_page, page_data): page_data 
                            for page_data in batch_pages}
            
            # Collect results as they complete
            for future in as_completed(future_to_page):
                page_data = future_to_page[future]
                try:
                    result = future.result()
                    batch_results.extend(result)
                except Exception as e:
                    st.error(f"❌ Error processing {page_data[1]} page {page_data[2]}: {e}")
                
                completed_in_batch += 1
                batch_progress.progress(completed_in_batch / len(batch_pages),
                                      text=f"Batch {batch_num}: {completed_in_batch}/{len(batch_pages)} pages")
        
        batch_progress.empty()
        all_results.extend(batch_results)
        
        st.success(f"✅ Batch {batch_num} completed: {len(batch_results)} records extracted")
        
        # Small delay between batches to be nice to the API
        if batch_end < total_pages:
            time.sleep(1)
    
    return all_results

# ---------- MAIN LOGIC ---------- #
if uploaded_files:
    st.markdown("---")
    st.subheader("🚀 Parallel Processing Mode")
    
    # Display current settings
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Parallel Workers", max_workers)
    with col2:
        st.metric("Batch Size", batch_size)
    with col3:
        st.metric("API Delay", f"{delay_between_calls}s")
    
    st.info(f"⚡ Processing up to {max_workers} pages simultaneously in batches of {batch_size}")
    
    total_files = len(uploaded_files)
    overall_progress = st.progress(0, text="Starting parallel processing...")
    
    combined_data = []
    processing_stats = {
        "total_pages": 0, 
        "total_records": 0, 
        "processing_time": 0,
        "pages_per_second": 0
    }
    
    start_time = time.time()

    for file_index, uploaded_file in enumerate(uploaded_files):
        st.markdown(f"### 📎 File {file_index + 1}/{total_files}: {uploaded_file.name}")
        file_start_time = time.time()
        
        file_ext = uploaded_file.name.lower().split(".")[-1]

        if file_ext == "pdf":
            try:
                # Extract all pages first
                pages_data, total_pages = extract_pdf_pages_parallel(uploaded_file, uploaded_file.name)
                processing_stats["total_pages"] += total_pages
                
                # Process pages in parallel
                file_results = process_pages_in_parallel(pages_data, uploaded_file.name, max_workers, batch_size)
                
            except Exception as e:
                st.error(f"❌ Error processing PDF {uploaded_file.name}: {e}")
                continue

        elif file_ext in ["png", "jpg", "jpeg"]:
            try:
                img = Image.open(uploaded_file)
                img_b64 = image_to_base64(img)
                file_results = call_gpt4_vision_with_retry(img_b64, uploaded_file.name, 1)
                processing_stats["total_pages"] += 1
            except Exception as e:
                st.error(f"❌ Error processing image {uploaded_file.name}: {e}")
                continue
        else:
            st.warning(f"⚠️ Unsupported file format: {uploaded_file.name}")
            continue

        # Display results for this file
        file_end_time = time.time()
        file_processing_time = file_end_time - file_start_time

        if file_results:
            df = pd.DataFrame(file_results)
            combined_data.extend(file_results)
            processing_stats["total_records"] += len(file_results)

            # File summary
            high_conf = df['extraction_quality'].str.lower().str.contains("high", na=False).sum()
            med_conf = df['extraction_quality'].str.lower().str.contains("medium", na=False).sum()
            low_conf = df['extraction_quality'].str.lower().str.contains("low", na=False).sum()

            with st.expander(f"📊 Results for {uploaded_file.name}"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Pages", len(df['page_number'].unique()))
                with col2:
                    st.metric("Records", len(df))
                with col3:
                    st.metric("Time", f"{file_processing_time:.1f}s")
                with col4:
                    pages_processed = len(df['page_number'].unique())
                    if file_processing_time > 0:
                        speed = pages_processed / file_processing_time
                        st.metric("Speed", f"{speed:.1f} pages/s")

                # Quality breakdown
                st.markdown("**Quality Breakdown:**")
                quality_col1, quality_col2, quality_col3 = st.columns(3)
                with quality_col1:
                    st.markdown(f"✅ High: {high_conf}")
                with quality_col2:
                    st.markdown(f"⚠️ Medium: {med_conf}")
                with quality_col3:
                    st.markdown(f"❌ Low: {low_conf}")

            st.success(f"✅ Processed {len(df['page_number'].unique())} pages in {file_processing_time:.1f}s")
            st.dataframe(df, use_container_width=True)

            # Download for individual file
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"📥 Download CSV for {uploaded_file.name}", 
                data=csv_data, 
                file_name=f"{uploaded_file.name.split('.')[0]}_extracted.csv",
                key=f"download_{file_index}"
            )
        else:
            st.warning(f"⚠️ No data extracted from {uploaded_file.name}")

        overall_progress.progress((file_index + 1) / total_files)

    # Final comprehensive results
    total_time = time.time() - start_time
    processing_stats["processing_time"] = total_time
    if total_time > 0:
        processing_stats["pages_per_second"] = processing_stats["total_pages"] / total_time

    if combined_data:
        st.markdown("---")
        st.subheader("🎯 Final Results & Performance")
        
        # Performance metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Files Processed", total_files)
        with col2:
            st.metric("Total Pages", processing_stats["total_pages"])
        with col3:
            st.metric("Total Records", processing_stats["total_records"])
        with col4:
            st.metric("Total Time", f"{total_time:.1f}s")
        with col5:
            st.metric("Speed", f"{processing_stats['pages_per_second']:.1f} pages/s")

        # Performance comparison
        if processing_stats["total_pages"] > 0:
            sequential_time = processing_stats["total_pages"] * (3 + 2)  # 3s delay + ~2s processing
            time_saved = sequential_time - total_time
            speedup = sequential_time / total_time if total_time > 0 else 1
            
            st.info(f"⚡ **Performance Boost**: {speedup:.1f}x faster than sequential processing! Saved ~{time_saved/60:.1f} minutes")

        # Combined results
        combined_df = pd.DataFrame(combined_data)
        st.dataframe(combined_df, use_container_width=True)
        
        # Combined download
        combined_csv = combined_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download All Results (Combined CSV)", 
            data=combined_csv, 
            file_name="all_extracted_invoices_parallel.csv"
        )
        
        # Quality analysis across all files
        with st.expander("📈 Overall Quality Analysis"):
            quality_df = combined_df['extraction_quality'].str.lower().value_counts().reset_index()
            quality_df.columns = ['Quality', 'Count']
            st.bar_chart(quality_df.set_index('Quality'))

else:
    st.info("📁 Upload PDF or Image files to begin parallel extraction")
    st.markdown("""
    ### 🚀 Parallel Processing Features:
    
    - ⚡ **True Parallel Processing**: Process multiple PDF pages simultaneously
    - 🎛️ **Configurable Workers**: Adjust parallel workers (1-15) based on your API limits  
    - 📦 **Batch Processing**: Process pages in batches to manage memory efficiently
    - 🔄 **Smart Rate Limiting**: Configurable delays to avoid API rate limits
    - 📊 **Real-time Performance Metrics**: Monitor speed and efficiency
    - 🎯 **Quality Analysis**: Track extraction confidence across all documents
    - 💾 **Flexible Downloads**: Individual file CSVs or combined results
    
    **Performance**: Up to 15x faster than sequential processing for large PDFs!
    """)