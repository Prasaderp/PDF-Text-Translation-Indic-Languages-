import time
import os

import config
from text_utils import parse_user_entities
from pdf_processor import extract_pdf_components
from translator import translate_chunk
from memory_utils import check_memory_and_reset
from pdf_builder import rebuild_pdf

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__ == "__main__":
    pdf_path = "/content/example.pdf"
    print("\n" + "="*40)
    print("Select target language (Hindi, Tamil, Telugu):")
    while True:
        target_language = input().strip().capitalize()
        if target_language in config.LANGUAGES:
            break
        print("Invalid choice. Please enter 'Hindi', 'Tamil', or 'Telugu'.")

    print("\n" + "="*40)
    print("Enter entities to preserve (comma-separated, e.g., 'Unni Jacobsen, Torstein Jahr, Suzanne Bolstad') (optional):")
    entities = parse_user_entities(input().strip())

    print("\n" + "="*40)
    print("Use white background for blocks? (yes/no):")
    use_white = input().strip().lower() in ('yes', 'y', 'true', 't', '1')

    components = extract_pdf_components(pdf_path)
    total_pages = len(components)
    fast_mode = total_pages <= 5

    start_time = time.time()
    print(f"\nStarting {target_language} translation")
    if fast_mode:
        translate_chunk(components, entities, target_language, fast_mode=True)
        print(f"Translated {total_pages} pages in one pass")
    else:
        chunk_size = 2
        num_chunks = (total_pages + chunk_size - 1) // chunk_size
        for i in range(0, total_pages, chunk_size):
            check_memory_and_reset(total_pages)
            chunk = components[i:i + chunk_size]
            translate_chunk(chunk, entities, target_language, fast_mode=False)
            print(f"Chunk {i // chunk_size + 1}/{num_chunks} translated ({len(chunk)} pages)")

    output_path = f"/content/translated_{target_language.lower()}.pdf"
    rebuild_pdf(components, output_path, pdf_path, target_language, use_white_background=use_white)
    print(f"\n{target_language} translation completed in {time.time()-start_time:.2f}s")