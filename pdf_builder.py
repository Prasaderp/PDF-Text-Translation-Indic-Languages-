import fitz
import config # For LANGUAGES

def int_to_hex_color(color_int):
    blue = color_int & 0xFF
    green = (color_int >> 8) & 0xFF
    red = (color_int >> 16) & 0xFF
    return f"#{red:02x}{green:02x}{blue:02x}"

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

def rebuild_pdf(components, output_path, original_pdf_path, target_language, use_white_background=True):
    print(f"\nRebuilding {target_language} PDF...")
    doc = fitz.open(original_pdf_path)
    lang_iso = config.LANGUAGES[target_language]["iso"]
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
                    print(f"Inserted subblock at {subblock_bbox.top_left}: '{subblock['translated_text'][:30]}...' with font size {adjusted_font_size}pt")
                except Exception as e:
                    print(f"Error inserting subblock at {subblock_bbox.top_left}: {e}")

        for link in links:
            page.insert_link(link)
            print(f"Restored link to: {link.get('uri', 'unknown destination')}")
    print(f"Saving to {output_path}")
    doc.save(output_path, garbage=4, deflate=True)

    print(f"\nVerifying text in {output_path}...")
    doc_check = fitz.open(output_path) # Renamed to avoid conflict with 'doc' above
    for page_num in range(len(doc_check)):
        page = doc_check[page_num]
        text = page.get_text("text")
        print(f"Extracted text from page {page_num+1}:\n{text}\n")
    doc_check.close()
