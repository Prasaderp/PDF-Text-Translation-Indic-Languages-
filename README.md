# Multilingual PDF Translation Suite

## Overview
The **Multilingual PDF Translation Suite** is a professional-grade solution designed to transform English PDF documents into Hindi, Tamil, or Telugu with exceptional accuracy and efficiency. This tool leverages advanced AI to extract text, preserve critical entities, translate content, and reconstruct PDFs while maintaining their original structure. Ideal for businesses, researchers, and multilingual workflows, it combines power and precision in an accessible package.

## Key Features
- **📑 Text Extraction**: Uses PyMuPDF (`fitz`) to dissect PDFs into text blocks and lines for precise processing.
- **🌐 Entity-Preserving Translation**: Powered by the NLLB-200-3.3B model, it translates to Hindi, Tamil, or Telugu while safeguarding user-specified entities (e.g., names, places).
- **⚡ Adaptive Performance**: Dynamic batch sizing and GPU memory management ensure smooth operation, with a fast mode for smaller files.
- **🛠️ PDF Reconstruction**: Rebuilds translated PDFs with options for white backgrounds or redactions, restoring links for functionality.
- **🔍 Interactive Workflow**: User-driven inputs for entities, languages, and styling, with real-time console feedback.

## Core Components
- **🔧 `initialize_model()`**: Loads the NLLB-200-3.3B model and tokenizer, optimized with `torch.float16` for GPU or `float32` for CPU, ensuring rapid startup.
- **📋 `parse_user_entities()`**: Parses comma-separated entities (e.g., "Mumbai, Ravi") to preserve during translation, using regex for accuracy.
- **🌍 `translate_batch()`**: Handles bulk translation with dynamic batch sizes (up to 16) and memory cleanup, adjusting for fast or thorough modes.
- **🖌️ `rebuild_pdf()`**: Reconstructs the PDF with translated text, aligning it to original line widths using `redistribute_translated_text()`.
- **⚙️ `reset_gpu_memory()`**: Monitors GPU usage, resetting at 80% capacity to prevent crashes on large files.

## Technical Specs
- **Models**: NLLB-200-3.3B for translation, managed with PyTorch and Transformers.
- **Languages**: Hindi (`hin_Deva`), Tamil (`tam_Taml`), Telugu (`tel_Telu`).
- **Dependencies**: Python 3.8+, `fitz`, `torch`, `transformers`—install with `pip install -r requirements.txt`.
- **Hardware**: GPU (CUDA) recommended; CPU supported with adaptive settings.

## Usage
1. **Input**: Provide a PDF (e.g., `sample-10-page-pdf-a4-size.pdf`).
2. **Configure**: Enter entities to preserve, select languages, and choose background style via prompts.
3. **Process**: Extract, translate, and rebuild in chunks (2 pages) or fast mode (≤5 pages).
4. **Output**: Get translated PDFs (e.g., `translated_Tamil.pdf`) with verified text.

## License
Licensed under the **MIT License**, offering flexibility for professional and open-source use.

## Contact
Questions or contributions? Reach out via GitHub issues or email for support.
