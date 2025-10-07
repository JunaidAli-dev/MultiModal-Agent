import os
import base64
from io import BytesIO

import fitz 
from PIL import Image
from unstructured.partition.pdf import partition_pdf

# LangChain and model components
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS

# --- 1. System Configuration ---
class Config:
    PDF_PATH = "report.pdf"
    EMBEDDING_MODEL = "models/embedding-001"
    FAISS_INDEX_PATH = "faiss_index_multimodal"
    IMAGE_DPI = 200

# --- 2. Multimodal Processing Functions ---
def render_pdf_pages_as_images(pdf_path, dpi):
    """Renders all pages of a PDF into a list of PIL Image objects."""
    print("Rendering PDF pages to images...")
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    print(f"Rendered {len(images)} pages.")
    return images

def crop_element_from_image(element, page_image):
    """Crops an element from its page image using bounding box coordinates."""
    bbox = element.metadata.coordinates.points
    cropped_image = page_image.crop((bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]))
    return cropped_image

def encode_image_to_base64(image, format="JPEG"):
    """Encodes a PIL image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- 3. Main Execution ---
if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("Error: GOOGLE_API_KEY environment variable not set.")
    
    config = Config()

    print(f"Partitioning {config.PDF_PATH}...")
    elements = partition_pdf(
        filename=config.PDF_PATH,
        strategy="hi_res",
        infer_table_structure=True,
        extract_coordinates=True
    )

    text_elements = [el for el in elements if "Table" not in str(type(el))]
    table_elements = [el for el in elements if "Table" in str(type(el))]
    print(f"Found {len(text_elements)} text elements and {len(table_elements)} table elements.")
    
    page_images = render_pdf_pages_as_images(config.PDF_PATH, dpi=config.IMAGE_DPI)

    documents = []
    
    for text_el in text_elements:
        documents.append(
            Document(
                page_content=str(text_el),
                metadata={"source": config.PDF_PATH, "type": "text"}
            )
        )
        
    for table_el in table_elements:
        page_num = table_el.metadata.page_number
        if 1 <= page_num <= len(page_images):
            table_image = crop_element_from_image(table_el, page_images[page_num - 1])
            
            # FIX: Check if the cropped image is empty before proceeding
            if table_image.width == 0 or table_image.height == 0:
                print(f"WARNING: Skipping empty table element on page {page_num}.")
                continue

            b64_image = encode_image_to_base64(table_image)
            
            documents.append(
                Document(
                    page_content=f"Image of a table from page {page_num}. Bounding Box: {table_el.metadata.coordinates.points}",
                    metadata={
                        "source": config.PDF_PATH,
                        "type": "image",
                        "page_number": page_num,
                        "base64_image": b64_image
                    }
                )
            )

    print(f"\nCreated {len(documents)} total documents (text and image references).")

    print(f"\nInitializing embedding model: {config.EMBEDDING_MODEL}...")
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)

    print("Creating and saving FAISS vector store...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(config.FAISS_INDEX_PATH)

    print(f"\n--- ✅ TASK 1 COMPLETE ---")
    print(f"Vector store created in '{config.FAISS_INDEX_PATH}' folder.")
    print("The store contains vectors for text and placeholders for images, with image data stored in metadata.")