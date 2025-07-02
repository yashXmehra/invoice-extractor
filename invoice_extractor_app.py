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
st.title("📄 Invoice Extractor")

# Configuration Settings
st.sidebar.header("⚙️ Configuration")

# File-level parallelism
pdf_workers = st.sidebar.slider(
    "Concurrent Files", 
    min_value=1, max_value=20, value=10,
    help="Number of files processed simultaneously"
)

# Page-level parallelism within each PDF
page_workers = st.sidebar.slider(
    "Pages per File", 
    min_value=1, max_value=30, value=20,
    help="Number of pages processed simultaneously per file"
)

# Batch settings
batch_size = st.sidebar.slider(
    "Batch Size", 
    min_value=10, max_value=100, value=50,
    help="Number of pages processed per batch"
)

batch_workers = st.sidebar.slider(
    "Concurrent Batches", 
    min_value=1, max_value=5, value=2,
    help="Number of batches processed simultaneously per file"
)

# Rate limiting
delay_between_calls = st.sidebar.slider(
    "API Delay (seconds)", 
    min_value=0.0, max_value=1.0, value=0.1, step=0.05,
    help="Delay between API calls to manage rate limits"
)

# Max API calls estimation
max_concurrent_calls = pdf_workers * page_workers * batch_workers
st.sidebar.warning(f"⚠️ Estimated concurrent API calls: {max_concurrent_calls}")
st.sidebar.info("💡 Adjust settings if rate limits are exceeded")

uploaded_files = st.file_uploader(
    "Upload PDF and image files",
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
                return []
            time.sleep(0.5 * (attempt + 1))
    return []

def process_single_page(args):
    """Process a single page"""
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
        st.error(f"Error extracting pages from {filename}: {e}")
        return [], 0

def process_batch(batch_pages, batch_info, page_workers):
    """Process a batch of pages"""
    batch_results = []
    
    with ThreadPoolExecutor(max_workers=page_workers) as executor:
        future_to_page = {executor.submit(process_single_page, page_data): page_data 
                         for page_data in batch_pages}
        
        for future in as_completed(future_to_page):
            try:
                result = future.result()
                batch_results.extend(result)
            except Exception as e:
                pass
    
    return batch_results, batch_info

def process_pdf(pdf_data):
    """Process a single PDF file"""
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
    
    # Process batches
    all_results = []
    
    with ThreadPoolExecutor(max_workers=batch_workers) as executor:
        future_to_batch = {
            executor.submit(process_batch, batch_pages, batch_info, page_workers): batch_info
            for batch_pages, batch_info in batches
        }
        
        for future in as_completed(future_to_batch):
            try:
                batch_results, batch_info = future.result()
                all_results.extend(batch_results)
            except Exception as e:
                pass
    
    return all_results

# ---------- MAIN LOGIC ---------- #
if uploaded_files:
    st.markdown("---")
    st.subheader("Processing Configuration")
    
    # Display current settings
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Concurrent Files", pdf_workers)
    with col2:
        st.metric("Pages per File", page_workers)
    with col3:
        st.metric("Concurrent Batches", batch_workers)
    with col4:
        st.metric("Batch Size", batch_size)
    
    # Processing info
    st.info(f"""
    **Processing Configuration**:
    • {pdf_workers} files will be processed simultaneously
    • {page_workers} pages per file will be processed concurrently
    • {batch_workers} batches per file will run in parallel
    • Each batch contains up to {batch_size} pages
    • Estimated concurrent API calls: {max_concurrent_calls}
    """)
    
    total_files = len(uploaded_files)
    
    # Prepare all files
    pdf_data_list = []
    total_pages_all = 0
    
    st.info("Preparing files for processing...")
    prep_progress = st.progress(0, text="Analyzing uploaded files...")
    
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
                st.error(f"Error processing image {uploaded_file.name}: {e}")
        
        prep_progress.progress((i + 1) / total_files, 
                              text=f"Analyzed {i + 1}/{total_files} files")
    
    prep_progress.empty()
    
    if not pdf_data_list:
        st.error("No valid files to process")
    else:
        st.success(f"Ready to process {len(pdf_data_list)} files containing {total_pages_all} total pages")
        
        # Processing
        st.info(f"Processing {len(pdf_data_list)} files...")
        
        start_time = time.time()
        all_results = []
        
        # Process files
        with ThreadPoolExecutor(max_workers=pdf_workers) as executor:
            future_to_pdf = {
                executor.submit(process_pdf, pdf_data): pdf_data[1]
                for pdf_data in pdf_data_list
            }
            
            completed_files = 0
            main_progress = st.progress(0, text="Processing files...")
            
            for future in as_completed(future_to_pdf):
                filename = future_to_pdf[future]
                try:
                    file_results = future.result()
                    all_results.extend(file_results)
                    completed_files += 1
                    
                    main_progress.progress(completed_files / len(pdf_data_list),
                                         text=f"Completed {completed_files}/{len(pdf_data_list)} files")
                    
                    st.success(f"Completed {filename}: {len(file_results)} records extracted")
                    
                except Exception as e:
                    st.error(f"Error processing {filename}: {e}")
        
        main_progress.empty()
        
        # Results
        total_time = time.time() - start_time
        
        if all_results:
            st.markdown("---")
            st.subheader("Processing Results")
            
            # Performance metrics
            pages_per_second = total_pages_all / total_time if total_time > 0 else 0
            sequential_time = total_pages_all * 3
            speedup = sequential_time / total_time if total_time > 0 else 1
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Files Processed", len(pdf_data_list))
            with col2:
                st.metric("Pages Processed", total_pages_all)
            with col3:
                st.metric("Records Extracted", len(all_results))
            with col4:
                st.metric("Processing Time", f"{total_time:.1f}s")
            with col5:
                st.metric("Processing Rate", f"{pages_per_second:.1f} pages/s")
            
            # Performance summary
            st.success(f"""
            **Processing Summary**:
            • {speedup:.1f}x faster than sequential processing
            • Average processing rate: {pages_per_second:.1f} pages per second
            • Time saved: approximately {(sequential_time - total_time)/60:.1f} minutes
            """)
            
            # Display results
            combined_df = pd.DataFrame(all_results)
            st.dataframe(combined_df, use_container_width=True)
            
            # Download results
            combined_csv = combined_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results (CSV)", 
                data=combined_csv, 
                file_name="invoice_extraction_results.csv"
            )
            
            # Quality analysis
            with st.expander("Quality Analysis"):
                if 'extraction_quality' in combined_df.columns:
                    quality_counts = combined_df['extraction_quality'].str.lower().value_counts()
                    st.bar_chart(quality_counts)
                else:
                    st.info("Quality metrics not available")
        else:
            st.warning("No data was extracted from the uploaded files")

else:
    st.info("Please upload PDF or image files to begin processing")
    # st.markdown("""
    # ### Invoice Extractor
    
    # **Features**:
    # - **Multi-file processing**: Process multiple PDF and image files simultaneously
    # - **Concurrent page processing**: Handle individual pages in parallel for faster extraction
    # - **Batch processing**: Organize pages into batches for optimal performance
    # - **Configurable settings**: Adjust processing parameters based on requirements
    # - **Rate limit management**: Built-in controls to manage API usage
    # - **Quality analysis**: Confidence scoring for extracted data
    # - **CSV export**: Download results in structured format
    
    # **Supported formats**: PDF, PNG, JPG, JPEG
    
    # **Use cases**: Invoice processing, document digitization, data extraction workflows
    # """)