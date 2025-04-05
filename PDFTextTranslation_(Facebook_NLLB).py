import fitz  # PyMuPDF
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
import time
import os

# Configuration
MODEL_NAME = "facebook/nllb-200-3.3B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANGUAGES = {
    "Hindi": {"code": "hin_Deva", "iso": "hi"},
    "Tamil": {"code": "tam_Taml", "iso": "ta"},
    "Telugu": {"code": "tel_Telu", "iso": "te"}
}
MAX_LENGTH_DEFAULT = 256
MEMORY_THRESHOLD = 0.8

# Digit mappings for Hindi, Tamil, and Telugu
DIGIT_MAP = {
    "Hindi": "०१२३४५६७८९",  # Hindi Devanagari digits
    "Tamil": "௦௧௨௩௪௫௬௭௮௯",  # Tamil digits
    "Telugu": "౦౧౨౩౪౫౬౭౮౯"  # Telugu digits
}
LATIN_DIGITS = "0123456789"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Initialize model
def initialize_model():
    print("🔄 Initializing translation model...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE).eval()
    print(f"✅ Model loaded in {time.time()-start:.2f}s")
    return tokenizer, model

tokenizer, model = initialize_model()

# Utility functions
def parse_user_entities(user_input):
    entities = [e.strip() for e in user_input.split(',') if e.strip()]
    print(f"📌 Entities to preserve: {', '.join(entities) if entities else 'None'}")
    return sorted(set(entities), key=len, reverse=True)

def replace_with_placeholders(text, entities):
    placeholder_map = {}
    modified_text = text

    # Automatic patterns (emails and URLs only)
    patterns = [
        (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), "emails"),
        (re.compile(r'https?://\S+|www\.\S+'), "URLs")
    ]

    for pattern, _ in patterns:
        matches = pattern.findall(modified_text)
        for match in matches:
            placeholder = f"__PRESERVE{len(placeholder_map):03d}__"
            placeholder_map[placeholder] = match
            modified_text = modified_text.replace(match, placeholder)

    # User-specified entities
    for entity in entities:
        pattern = re.compile(r'\b' + re.escape(entity) + r'\b', re.IGNORECASE)
        def replacer(match):
            original = match.group()
            placeholder = f"__PRESERVE{len(placeholder_map):03d}__"
            placeholder_map[placeholder] = original
            return placeholder
        modified_text, count = pattern.subn(replacer, modified_text)
        if count > 0:
            print(f"🔧 Replaced '{entity}' {count} time(s)")

    print(f"🔍 Modified text with placeholders: '{modified_text}'")
    return modified_text, placeholder_map

def convert_numbers_to_script(text, target_language):
    """Convert Latin digits to the target language's numeral script."""
    digit_map = DIGIT_MAP[target_language]
    def replace_digit(match):
        number = match.group(0)
        converted = ''.join(digit_map[int(d)] if d in LATIN_DIGITS else d for d in number)
        return converted
    pattern = re.compile(r'(?<!__PRESERVE\d{3}__)\b\d+(?:\.\d+)?\b(?![^_]*__)')
    return pattern.sub(replace_digit, text)

def get_dynamic_batch_size(num_texts, fast_mode):
    if DEVICE != "cuda":
        return min(8, num_texts)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    free_memory = total_memory - torch.cuda.memory_allocated()
    tokens_per_text = MAX_LENGTH_DEFAULT
    bytes_per_text = tokens_per_text * 4 * (3 if fast_mode else 1)
    max_batch = max(1, min(free_memory // bytes_per_text, num_texts))
    return min(16 if fast_mode else 4, max_batch)

def translate_batch(texts, target_language, fast_mode=False):
    if not texts:
        return []
    batch_size = get_dynamic_batch_size(len(texts), fast_mode)
    translated_texts = []
    target_lang_code = LANGUAGES[target_language]["code"]
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        max_length = max(MAX_LENGTH_DEFAULT, max(len(t.split()) for t in batch) * 2)
        try:
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(DEVICE)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang_code),
                    max_length=max_length,
                    num_beams=3 if fast_mode else 1,
                    use_cache=True,
                    early_stopping=True
                )
            translated = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            translated_texts.extend([re.sub(r'^\.+|\s*\.+$|^\s*…', '', t.strip()) for t in translated])
            del inputs, outputs
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"⚠️ Memory error: {e}. Reducing batch size and retrying...")
            if batch_size > 1:
                batch_size = max(1, batch_size // 2)
                translated_texts.extend(translate_batch(batch, target_language, fast_mode))
            else:
                raise
    return translated_texts

def reset_gpu_memory():
    global model, tokenizer
    if DEVICE == "cuda":
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("🔄 Refreshing GPU memory...")
        start = time.time()
        tokenizer, model = initialize_model()
        print(f"✅ GPU memory refreshed in {time.time()-start:.2f}s")

def check_memory_and_reset(total_pages):
    if DEVICE != "cuda" or total_pages <= 5:
        return False
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated()
    if allocated_memory / total_memory > MEMORY_THRESHOLD:
        reset_gpu_memory()
        return True
    return False

def join_spans(spans):
    if not spans:
        return ""
    spans = sorted(spans, key=lambda s: s["bbox"][0])
    text_parts = [spans[0]["text"].strip()]
    for i in range(1, len(spans)):
        span1, span2 = spans[i - 1], spans[i]
        d = span2["bbox"][0] - span1["bbox"][2]
        text2 = span2["text"].strip()
        if not text2:
            continue
        if len(span1["text"]) > 0 and len(text2) > 0:
            width1 = span1["bbox"][2] - span1["bbox"][0]
            width2 = span2["bbox"][2] - span2["bbox"][0]
            min_avg_char_width = min(width1 / len(span1["text"]), width2 / len(text2))
            if d < 0.5 * min_avg_char_width or d < 0:
                text_parts.append(text2)
            else:
                text_parts.append(" " + text2)
        else:
            text_parts.append(text2)
    return "".join(text_parts)

# Extract and segment PDF
def extract_pdf_components(pdf_path):
    print(f"\n📄 Extracting components from {pdf_path}...")
    doc = fitz.open(pdf_path)
    components = []
    for page_num, page in enumerate(doc):
        print(f"\n📖 Processing page {page_num+1}")
        blocks = page.get_text("dict")["blocks"]
        text_blocks = []
        for b in blocks:
            if b["type"] == 0:  # Text block
                lines = []
                for line in b["lines"]:
                    if line["spans"]:
                        text = join_spans(line["spans"])
                        if text.strip():
                            lines.append({
                                "text": text,
                                "y_pos": line["spans"][0]["origin"][1],
                                "x_pos": line["spans"][0]["origin"][0],
                                "font_size": line["spans"][0]["size"],
                                "color": line["spans"][0]["color"],
                                "line_bbox": line["bbox"]
                            })
                if lines:
                    text_blocks.append({"bbox": b["bbox"], "lines": lines})
        components.append({"page_num": page_num, "text_blocks": text_blocks, "size": (page.rect.width, page.rect.height)})
    doc.close()
    return components

def split_block_into_subblocks(block):
    lines = block["lines"]
    if not lines:
        return []
    subblocks = []
    current_subblock = None

    for i, line in enumerate(lines):
        text = line["text"].strip()
        if not text:
            continue
        font_size = line["font_size"]
        gap = (lines[i + 1]["y_pos"] - line["y_pos"] - font_size) if i < len(lines) - 1 else font_size
        x_shift = abs(line["x_pos"] - lines[i-1]["x_pos"]) if i > 0 else 0

        if current_subblock is None:
            current_subblock = {"text": text, "lines": [line], "font_size": font_size}
        else:
            # Split only on significant changes to preserve paragraph-like structure
            if (font_size != current_subblock["font_size"] or
                gap > font_size * 1.5 or  # Increased threshold for paragraph breaks
                x_shift > 10):
                subblocks.append(current_subblock)
                current_subblock = {"text": text, "lines": [line], "font_size": font_size}
            else:
                current_subblock["text"] += " " + text
                current_subblock["lines"].append(line)

    if current_subblock:
        subblocks.append(current_subblock)
    return subblocks

# Translate with context awareness
def translate_chunk(chunk, entities, target_language, fast_mode=False):
    all_subblocks = []
    for page in chunk:
        for block in page["text_blocks"]:
            subblocks = split_block_into_subblocks(block)
            block["subblocks"] = subblocks
            all_subblocks.extend(subblocks)

    if not all_subblocks:
        return

    texts = []
    placeholder_maps = []
    for subblock in all_subblocks:
        original_text = subblock["text"]
        if not original_text.strip():
            subblock["translated_text"] = ""
            continue
        modified_text, placeholder_map = replace_with_placeholders(original_text, entities)
        texts.append(modified_text)
        placeholder_maps.append(placeholder_map)

    if texts:
        translated_texts = translate_batch(texts, target_language, fast_mode=fast_mode)
        for subblock, translated_text, placeholder_map in zip([sb for sb in all_subblocks if sb["text"].strip()], translated_texts, placeholder_maps):
            for placeholder, original in placeholder_map.items():
                if placeholder in translated_text:
                    translated_text = translated_text.replace(placeholder, original)
                    print(f"🔄 Restored '{original}' in '{translated_text}'")
                else:
                    translated_text += f" {original}"
                    print(f"🔄 Appended '{original}' to '{translated_text}'")
            translated_text = convert_numbers_to_script(translated_text, target_language)
            print(f"🔢 Converted numbers in '{translated_text}'")
            subblock["translated_text"] = translated_text

# Convert color from integer to hex
def int_to_hex_color(color_int):
    blue = color_int & 0xFF
    green = (color_int >> 8) & 0xFF
    red = (color_int >> 16) & 0xFF
    return f"#{red:02x}{green:02x}{blue:02x}"

# Estimate text height and adjust font size
def estimate_text_height(text, bbox_width, bbox_height, initial_font_size, fontname="helv"):
    if not text.strip():
        return initial_font_size

    min_font_size = 5  # Minimum readable font size
    step = 0.5  # Font size adjustment step

    def calculate_height(font_size):
        lines = []
        current_line = []
        current_width = 0
        words = text.split()

        for word in words:
            word_width = fitz.get_text_length(word, fontname, font_size)
            space_width = fitz.get_text_length(" ", fontname, font_size)
            total_width = word_width + space_width

            if current_width + total_width > bbox_width:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                    current_width = word_width
                else:
                    lines.append(word)
                    current_width = 0
            else:
                current_line.append(word)
                current_width += total_width

        if current_line:
            lines.append(" ".join(current_line))

        line_height = font_size * 1.1
        return len(lines) * line_height

    font_size = initial_font_size
    total_height = calculate_height(font_size)
    if total_height > bbox_height:
        while font_size > min_font_size and total_height > bbox_height:
            font_size -= step
            total_height = calculate_height(font_size)
    else:
        while total_height <= bbox_height and font_size < initial_font_size + step:
            candidate_font_size = font_size + step
            candidate_height = calculate_height(candidate_font_size)
            if candidate_height <= bbox_height:
                font_size = candidate_font_size
                total_height = candidate_height
            else:
                break

    return max(min_font_size, min(initial_font_size, font_size))

# Rebuild PDF with layout preservation and font size adjustment
def rebuild_pdf(components, output_path, original_pdf_path, target_language, use_white_background=True):
    print(f"\n🏗️ Rebuilding {target_language} PDF...")
    doc = fitz.open(original_pdf_path)
    lang_iso = LANGUAGES[target_language]["iso"]
    for page_data in components:
        page = doc[page_data["page_num"]]
        links = list(page.get_links())
        for block in page_data["text_blocks"]:
            original_bbox = fitz.Rect(block["bbox"])
            if use_white_background:
                page.draw_rect(original_bbox, color=(1, 1, 1), fill=(1, 1, 1), fill_opacity=1.0)
            else:
                page.add_redact_annot(original_bbox)
                page.apply_redactions()

            for subblock in block["subblocks"]:
                if "translated_text" not in subblock or not subblock["translated_text"].strip():
                    continue
                subblock_bbox = fitz.Rect(
                    min(line["line_bbox"][0] for line in subblock["lines"]),
                    min(line["line_bbox"][1] for line in subblock["lines"]),
                    max(line["line_bbox"][2] for line in subblock["lines"]),
                    max(line["line_bbox"][3] for line in subblock["lines"])
                )
                first_line = subblock["lines"][0]
                initial_font_size = first_line["font_size"]
                hex_color = int_to_hex_color(first_line["color"])

                adjusted_font_size = estimate_text_height(
                    subblock["translated_text"],
                    subblock_bbox.width,
                    subblock_bbox.height,
                    initial_font_size
                )

                html = f'<div style="width: 100%; height: 100%; padding: 0; margin: 0;"><p lang="{lang_iso}" style="margin: 0; padding: 0; font-size: {adjusted_font_size}pt; color: {hex_color};">{subblock["translated_text"]}</p></div>'
                try:
                    page.insert_htmlbox(subblock_bbox, html, css="", scale_low=0, rotate=0, oc=0, opacity=1, overlay=True)
                    print(f"✓ Inserted subblock at {subblock_bbox.top_left}: '{subblock['translated_text'][:30]}...' with font size {adjusted_font_size}pt")
                except Exception as e:
                    print(f"⚠️ Error inserting subblock at {subblock_bbox.top_left}: {e}")

        for link in links:
            page.insert_link(link)
            print(f"🔗 Restored link to: {link.get('uri', 'unknown destination')}")
    print(f"💾 Saving to {output_path}")
    doc.save(output_path, garbage=4, deflate=True)

    print(f"\n🔍 Verifying text in {output_path}...")
    doc = fitz.open(output_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        print(f"Extracted text from page {page_num+1}:\n{text}\n")
    doc.close()

# Main execution
if __name__ == "__main__":
    pdf_path = "/content/example.pdf"
    print("\n" + "="*40)
    print("🌐 Select target language (Hindi, Tamil, Telugu):")
    while True:
        target_language = input().strip().capitalize()
        if target_language in LANGUAGES:
            break
        print("❌ Invalid choice. Please enter 'Hindi', 'Tamil', or 'Telugu'.")

    print("\n" + "="*40)
    print("📝 Enter entities to preserve (comma-separated, e.g., 'Unni Jacobsen, Torstein Jahr, Suzanne Bolstad') (optional):")
    entities = parse_user_entities(input().strip())

    print("\n" + "="*40)
    print("🎨 Use white background for blocks? (yes/no):")
    use_white = input().strip().lower() in ('yes', 'y', 'true', 't', '1')

    components = extract_pdf_components(pdf_path)
    total_pages = len(components)
    fast_mode = total_pages <= 5

    start_time = time.time()
    print(f"\n🚀 Starting {target_language} translation")
    if fast_mode:
        translate_chunk(components, entities, target_language, fast_mode=True)
        print(f"✅ Translated {total_pages} pages in one pass")
    else:
        chunk_size = 2
        num_chunks = (total_pages + chunk_size - 1) // chunk_size
        for i in range(0, total_pages, chunk_size):
            check_memory_and_reset(total_pages)
            chunk = components[i:i + chunk_size]
            translate_chunk(chunk, entities, target_language, fast_mode=False)
            print(f"✅ Chunk {i // chunk_size + 1}/{num_chunks} translated ({len(chunk)} pages)")

    output_path = f"/content/translated_{target_language.lower()}.pdf"
    rebuild_pdf(components, output_path, pdf_path, target_language, use_white_background=use_white)
    print(f"\n✅ {target_language} translation completed in {time.time()-start_time:.2f}s")