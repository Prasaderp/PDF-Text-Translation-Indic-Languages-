import fitz

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
    print(f"\nExtracting components from {pdf_path}...")
    doc = fitz.open(pdf_path)
    components = []
    for page_num, page in enumerate(doc):
        print(f"\nProcessing page {page_num+1}")
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