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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

def parse_user_entities(user_input):
    entities = [e.strip() for e in user_input.split(',') if e.strip()]
    print(f"📌 Entities to preserve: {', '.join(entities) if entities else 'None'}")
    return sorted(set(entities), key=len, reverse=True)

def parse_user_languages(user_input):
    selected = [lang.strip().capitalize() for lang in user_input.split(',')]
    valid = [lang for lang in selected if lang in LANGUAGES]
    if not valid:
        print("⚠️ No valid languages selected. Using all available.")
        return list(LANGUAGES.keys())
    print(f"🌍 Selected languages: {', '.join(valid)}")
    return valid

def replace_with_placeholders(text, entities):
    placeholder_map = {}
    modified_text = text
    for idx, entity in enumerate(entities):
        pattern = re.compile(r'\b' + re.escape(entity) + r'\b', re.IGNORECASE)
        def replacer(match):
            original = match.group()
            placeholder = f"__ENT{len(placeholder_map):03d}__"
            placeholder_map[placeholder] = original
            return placeholder
        modified_text, count = pattern.subn(replacer, modified_text)
        if count > 0:
            print(f"🔧 Replaced '{entity}' {count} time(s) in text: '{text}'")
    print(f"🔍 Modified text with placeholders: '{modified_text}'")
    return modified_text, placeholder_map

def get_dynamic_batch_size(num_texts, fast_mode):
    if DEVICE != "cuda":
        return min(8, num_texts)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    free_memory = total_memory - torch.cuda.memory_allocated()
    tokens_per_text = MAX_LENGTH_DEFAULT
    bytes_per_text = tokens_per_text * 4 * (3 if fast_mode else 1)
    max_batch = max(1, min(free_memory // bytes_per_text, num_texts))
    return min(16 if fast_mode else 4, max_batch)

def translate_batch(texts, target_lang_code, fast_mode=False):
    if not texts:
        return []
    batch_size = get_dynamic_batch_size(len(texts), fast_mode)
    translated_texts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH_DEFAULT).to(DEVICE)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang_code),
                    max_length=MAX_LENGTH_DEFAULT,
                    num_beams=3 if fast_mode else 1,
                    use_cache=True,
                    early_stopping=True
                )
            translated = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            # Clean up unwanted dots or ellipses
            translated_texts.extend([re.sub(r'^\.+|\s*\.+$|^\s*…', '', t.strip()) for t in translated])
            del inputs, outputs
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"⚠️ Memory error: {e}. Reducing batch size and retrying...")
            if batch_size > 1:
                batch_size = max(1, batch_size // 2)
                translated_texts.extend(translate_batch(batch, target_lang_code, fast_mode))
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
                text_lines = []
                for line in b["lines"]:
                    if line["spans"]:
                        text = join_spans(line["spans"])
                        if text.strip():  # Only include non-empty text
                            text_lines.append({
                                "text": text,
                                "y_pos": line["spans"][0]["origin"][1],
                                "font_size": line["spans"][0]["size"],
                                "line_bbox": line["bbox"]
                            })
                if text_lines:
                    text_blocks.append({"bbox": b["bbox"], "lines": text_lines})
        components.append({"page_num": page_num, "text_blocks": text_blocks, "size": (page.rect.width, page.rect.height)})
    doc.close()
    return components

def split_block_into_subblocks(block):
    lines = block["lines"]
    if not lines:
        return []
    subblocks = []
    current_subblock = {"text": "", "lines": []}
    for i, line in enumerate(lines):
        text = line["text"].strip()
        if not text:
            continue  # Skip empty lines
        current_subblock["text"] += " " + text if current_subblock["text"] else text
        current_subblock["lines"].append(line)
        if i < len(lines) - 1:
            next_line = lines[i + 1]
            gap = next_line["y_pos"] - line["y_pos"] - line["font_size"]
            if gap > line["font_size"] * 0.5:
                subblocks.append(current_subblock)
                current_subblock = {"text": "", "lines": []}
        else:
            subblocks.append(current_subblock)
    return subblocks

def translate_chunk(chunk, entities, target_lang, fast_mode=False):
    target_lang_code = LANGUAGES[target_lang]["code"]
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
        translated_texts = translate_batch(texts, target_lang_code, fast_mode=fast_mode)
        for subblock, translated_text, placeholder_map in zip([sb for sb in all_subblocks if sb["text"].strip()], translated_texts, placeholder_maps):
            for placeholder, entity in placeholder_map.items():
                if placeholder in translated_text:
                    translated_text = translated_text.replace(placeholder, entity)
                    print(f"🔄 Restored '{placeholder}' to '{entity}' in text: '{translated_text}'")
                else:
                    print(f"⚠️ Placeholder '{placeholder}' not found in: '{translated_text}'")
            subblock["translated_text"] = translated_text

    for page in chunk:
        for block in page["text_blocks"]:
            translated_subblocks = [sb["translated_text"] for sb in block["subblocks"] if sb.get("translated_text", "").strip()]
            block["translated_text"] = " ".join(translated_subblocks)
            block["original_lines"] = block["lines"]

def redistribute_translated_text(translated_text, original_lines):
    if not original_lines or not translated_text.strip():
        return [""] * len(original_lines)
    translated_words = translated_text.split()
    translated_lines = []
    word_idx = 0
    default_font = fitz.Font("helv")
    for line in original_lines:
        max_width = line["line_bbox"][2] - line["line_bbox"][0]
        font_size = line["font_size"]
        current_line = []
        current_width = 0
        while word_idx < len(translated_words):
            word = translated_words[word_idx]
            word_width = default_font.text_length(word + " ", fontsize=font_size)
            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width
                word_idx += 1
            else:
                break
        translated_lines.append(" ".join(current_line) if current_line else "")
    while len(translated_lines) < len(original_lines):
        translated_lines.append("")
    if word_idx < len(translated_words):
        remaining_text = " ".join(translated_words[word_idx:])
        translated_lines[-1] = translated_lines[-1] + " " + remaining_text if translated_lines[-1] else remaining_text
    return translated_lines

def rebuild_pdf(components, target_lang, output_path, original_pdf_path, use_white_background=True):
    print(f"\n🏗️ Rebuilding {target_lang} PDF...")
    doc = fitz.open(original_pdf_path)
    lang_iso = LANGUAGES[target_lang]["iso"]
    for page_data in components:
        page = doc[page_data["page_num"]]
        links = list(page.get_links())
        for block in page_data["text_blocks"]:
            original_bbox = fitz.Rect(block["bbox"])
            translated_text = block.get("translated_text", "")
            if not translated_text.strip():
                continue
            translated_lines = redistribute_translated_text(translated_text, block["original_lines"])
            if use_white_background:
                page.draw_rect(original_bbox, color=(1, 1, 1), fill=(1, 1, 1), fill_opacity=1.0)
            else:
                page.add_redact_annot(original_bbox)
                page.apply_redactions()
            for i, (original_line, translated_line) in enumerate(zip(block["original_lines"], translated_lines)):
                line_rect = fitz.Rect(original_line["line_bbox"])
                font_size = original_line["font_size"]
                if translated_line.strip():
                    html = f'<div style="width: 100%; height: 100%; padding: 0; margin: 0;"><p lang="{lang_iso}" style="margin: 0; padding: 0;">{translated_line}</p></div>'
                    css = f"p {{ font-size: {font_size}pt; }}"
                    try:
                        page.insert_htmlbox(line_rect, html, css=css, scale_low=0, rotate=0, oc=0, opacity=1, overlay=True)
                        print(f"✓ Inserted line {i+1} at {line_rect.top_left}: '{translated_line[:30]}...'")
                    except Exception as e:
                        print(f"⚠️ Error inserting line {i+1} at {line_rect.top_left}: {e}")
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

if __name__ == "__main__":
    pdf_path = "/content/sample-10-page-pdf-a4-size.pdf"
    print("\n" + "="*40)
    print("📝 Enter entities to preserve (comma-separated, e.g., 'Name, Place etc'):")
    entities = parse_user_entities(input().strip())
    print("\n" + "="*40)
    print("🌐 Available languages:", ", ".join(LANGUAGES.keys()))
    print("📢 Enter target languages (comma-separated):")
    languages = parse_user_languages(input().strip())
    print("\n" + "="*40)
    print("🎨 Use white background for blocks? (yes/no):")
    use_white = input().strip().lower() in ('yes', 'y', 'true', 't', '1')

    components = extract_pdf_components(pdf_path)
    total_pages = len(components)
    fast_mode = total_pages <= 5

    for lang in languages:
        start_time = time.time()
        print(f"\n🚀 Starting {lang} translation")
        if fast_mode:
            translate_chunk(components, entities, lang, fast_mode=True)
            print(f"✅ Translated {total_pages} pages in one pass")
        else:
            chunk_size = 2
            num_chunks = (total_pages + chunk_size - 1) // chunk_size
            for i in range(0, total_pages, chunk_size):
                check_memory_and_reset(total_pages)
                chunk = components[i:i + chunk_size]
                translate_chunk(chunk, entities, lang, fast_mode=False)
                print(f"✅ Chunk {i // chunk_size + 1}/{num_chunks} translated ({len(chunk)} pages)")
        output_path = f"/content/translated_{lang}.pdf"
        rebuild_pdf(components, lang, output_path, pdf_path, use_white_background=use_white)
        print(f"\n✅ {lang} translation completed in {time.time()-start_time:.2f}s")