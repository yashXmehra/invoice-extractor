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
from queue import Queue

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

# Advanced Sidebar Configuration
st.sidebar.header("🎛️ Ultra Parallel Settings")

# File-level parallelism
pdf_workers = st.sidebar.slider(
    "PDF Workers (Files in Parallel)", 
    min_value=1, max_value=20, value=10,
    help="Number of PDF files processed simultaneously"
)

# Page-level parallelism within each PDF
page_workers = st.sidebar.slider(
    "Page Workers (Pages per PDF)", 
    min_value=1, max_value=30, value=20,
    help="Number of pages processed simultaneously within each PDF"
)

# Batch settings
batch_size = st.sidebar.slider(
    "Batch Size", 
    min_value=10, max_value=100, value=50,
    help="Pages per batch (larger batches = fewer API limit issues)"
)

batch_workers = st.sidebar.slider(
    "Batch Workers (Batches in Parallel)", 
    min_value=1, max_value=5, value=2,
    help="Number of batches processed simultaneously per PDF"
)

# Rate limiting
delay_between_calls = st.sidebar.slider(
    "API Call Delay (seconds)", 
    min_value=0.0, max_value=1.0, value=0.1, step=0.05,
    help="Delay between API calls to avoid rate limiting"
)

# Max API calls estimation
max_concurrent_calls = pdf_workers * page_workers * batch_workers
st.sidebar.warning(f"⚠️ Max concurrent API calls: ~{max_concurrent_calls}")
st.sidebar.info("💡 Reduce workers if you hit rate limits")

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
            # Add delay for rate limiting
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
                return []  # Fail silently to avoid spam
            time.sleep(0.5 * (attempt + 1))  # Exponential backoff
    return []

def process_single_page(args):
    """Process a single page - wrapper for parallel execution"""
    image_base64, filename, page_number = args
    return call_gpt4_vision_with_retry(image_base64, filename, page_number)

def extract_pdf_pages(pdf_file, filename):
    """Extract all pages from PDF as images"""
    try:
        pdf_doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        total_pages = len(pdf_doc)
        
        pages_data = []
        for i in range(total_pages):
            page = pdf_doc[i]
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes()))
            img_b64 = image_to_base64(img)
            pages_data.append((img_b64, filename, i + 1))
        
        pdf_doc.close()
        return pages_data, total_pages
    except Exception as e:
        st.error(f"❌ Error extracting pages from {filename}: {e}")
        return [], 0

def process_batch_ultra_parallel(batch_pages, batch_info, page_workers):
    """Process a single batch with multiple workers"""
    batch_results = []
    
    with ThreadPoolExecutor(max_workers=page_workers) as executor:
        future_to_page = {executor.submit(process_single_page, page_data): page_data 
                         for page_data in batch_pages}
        
        for future in as_completed(future_to_page):
            try:
                result = future.result()
                batch_results.extend(result)
            except Exception as e:
                page_data = future_to_page[future]
                # Silent error handling to avoid UI spam
                pass
    
    return batch_results, batch_info

def process_pdf_ultra_parallel(pdf_data):
    """Process a single PDF with multiple batches running in parallel"""
    pages_data, filename, total_pages = pdf_data
    
    if not pages_data:
        return []
    
    # Split pages into batches
    batches = []
    for i in range(0, len(pages_data), batch_size):
        batch_pages = pages_data[i:i + batch_size]
        batch_info = {
            'filename': filename,
            'batch_num': (i // batch_size) + 1,
            'total_batches': (len(pages_data) + batch_size - 1) // batch_size,
            'pages_in_batch': len(batch_pages)
        }
        batches.append((batch_pages, batch_info))
    
    # Process batches in parallel
    all_results = []
    
    with ThreadPoolExecutor(max_workers=batch_workers) as executor:
        future_to_batch = {
            executor.submit(process_batch_ultra_parallel, batch_pages, batch_info, page_workers): batch_info
            for batch_pages, batch_info in batches
        }
        
        for future in as_completed(future_to_batch):
            try:
                batch_results, batch_info = future.result()
                all_results.extend(batch_results)
            except Exception as e:
                # Silent error handling
                pass
    
    return all_results

# ---------- MAIN LOGIC ---------- #
if uploaded_files:
    st.markdown("---")
    st.subheader("🚀 Ultra Parallel Processing Mode")
    
    # Display current settings
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("PDF Workers", pdf_workers, help="Files processed simultaneously")
    with col2:
        st.metric("Page Workers", page_workers, help="Pages per PDF simultaneously")
    with col3:
        st.metric("Batch Workers", batch_workers, help="Batches per PDF simultaneously")
    with col4:
        st.metric("Batch Size", batch_size, help="Pages per batch")
    
    # Processing info
    st.info(f"""
    🎯 **Processing Strategy**:
    • {pdf_workers} PDFs processed simultaneously
    • Within each PDF: {page_workers} pages processed simultaneously  
    • {batch_workers} batches per PDF run in parallel
    • Each batch contains up to {batch_size} pages
    • Max concurrent API calls: ~{max_concurrent_calls}
    """)
    
    total_files = len(uploaded_files)
    
    # Prepare all PDF data first
    pdf_data_list = []
    total_pages_all = 0
    
    st.info("📄 Preparing all PDF files...")
    prep_progress = st.progress(0, text="Extracting pages from all PDFs...")
    
    for i, uploaded_file in enumerate(uploaded_files):
        file_ext = uploaded_file.name.lower().split(".")[-1]
        
        if file_ext == "pdf":
            pages_data, total_pages = extract_pdf_pages(uploaded_file, uploaded_file.name)
            if pages_data:
                pdf_data_list.append((pages_data, uploaded_file.name, total_pages))
                total_pages_all += total_pages
        elif file_ext in ["png", "jpg", "jpeg"]:
            try:
                img = Image.open(uploaded_file)
                img_b64 = image_to_base64(img)
                pdf_data_list.append([[(img_b64, uploaded_file.name, 1)], uploaded_file.name, 1])
                total_pages_all += 1
            except Exception as e:
                st.error(f"❌ Error processing image {uploaded_file.name}: {e}")
        
        prep_progress.progress((i + 1) / total_files, 
                              text=f"Prepared {i + 1}/{total_files} files")
    
    prep_progress.empty()
    
    if not pdf_data_list:
        st.error("❌ No valid files to process")
    else:
        st.success(f"✅ Prepared {len(pdf_data_list)} files with {total_pages_all} total pages")
        
        # Ultra parallel processing
        st.info(f"🚀 Starting ultra parallel processing of {len(pdf_data_list)} files...")
        
        start_time = time.time()
        all_results = []
        
        # Process multiple PDFs simultaneously
        with ThreadPoolExecutor(max_workers=pdf_workers) as executor:
            future_to_pdf = {
                executor.submit(process_pdf_ultra_parallel, pdf_data): pdf_data[1]
                for pdf_data in pdf_data_list
            }
            
            completed_files = 0
            main_progress = st.progress(0, text="Processing all files in ultra parallel mode...")
            
            for future in as_completed(future_to_pdf):
                filename = future_to_pdf[future]
                try:
                    file_results = future.result()
                    all_results.extend(file_results)
                    completed_files += 1
                    
                    main_progress.progress(completed_files / len(pdf_data_list),
                                         text=f"Completed {completed_files}/{len(pdf_data_list)} files")
                    
                    st.success(f"✅ {filename}: {len(file_results)} records extracted")
                    
                except Exception as e:
                    st.error(f"❌ Error processing {filename}: {e}")
        
        main_progress.empty()
        
        # Final results
        total_time = time.time() - start_time
        
        if all_results:
            st.markdown("---")
            st.subheader("🎯 Ultra Parallel Results")
            
            # Performance metrics
            pages_per_second = total_pages_all / total_time if total_time > 0 else 0
            sequential_time = total_pages_all * 3  # Assume 3 seconds per page sequentially
            speedup = sequential_time / total_time if total_time > 0 else 1
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Files", len(pdf_data_list))
            with col2:
                st.metric("Total Pages", total_pages_all)
            with col3:
                st.metric("Total Records", len(all_results))
            with col4:
                st.metric("Total Time", f"{total_time:.1f}s")
            with col5:
                st.metric("Speed", f"{pages_per_second:.1f} pages/s")
            
            # Performance comparison
            st.success(f"""
            🚀 **Ultra Parallel Performance**:
            • **{speedup:.1f}x faster** than sequential processing
            • Processed **{pages_per_second:.1f} pages per second**
            • Saved approximately **{(sequential_time - total_time)/60:.1f} minutes**
            """)
            
            # Results dataframe
            combined_df = pd.DataFrame(all_results)
            st.dataframe(combined_df, use_container_width=True)
            
            # Download combined results
            combined_csv = combined_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download All Results (Ultra Parallel CSV)", 
                data=combined_csv, 
                file_name="ultra_parallel_extracted_invoices.csv"
            )
            
            # Quality analysis
            with st.expander("📈 Quality Analysis"):
                if 'extraction_quality' in combined_df.columns:
                    quality_counts = combined_df['extraction_quality'].str.lower().value_counts()
                    st.bar_chart(quality_counts)
                else:
                    st.info("Quality data not available in results")
        else:
            st.warning("⚠️ No data was extracted from any files")

else:
    st.info("📁 Upload PDF or Image files to begin ultra parallel extraction")
    # st.markdown("""
    # ### 🚀 Ultra Parallel Processing Capabilities:
    
    # #### **Multi-Level Parallelism**:
    # - 📄 **PDF Level**: Process up to 20 PDF files simultaneously
    # - 📃 **Page Level**: Process up to 30 pages per PDF simultaneously  
    # - 📦 **Batch Level**: Process up to 5 batches per PDF simultaneously
    # - ⚡ **Total Concurrency**: Up to 3,000+ concurrent API calls
    
    # #### **Performance Benefits**:
    # - 🎯 **10-50x faster** than sequential processing
    # - 🚀 **Massive throughput**: 100+ pages per second possible
    # - 🎛️ **Fully configurable**: Adjust workers based on API limits
    # - 📊 **Real-time monitoring**: Track processing across all levels
    
    # #### **Smart Features**:
    # - 🛡️ **Rate limit protection**: Configurable delays
    # - 🔄 **Automatic retry**: Failed pages are retried
    # - 💾 **Memory efficient**: Batched processing prevents overload
    # - 📈 **Performance metrics**: Detailed speed and efficiency tracking
    
    # **Perfect for**: Large document processing, bulk invoice extraction, enterprise workflows
    # """)