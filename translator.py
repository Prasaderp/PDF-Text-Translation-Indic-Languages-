import torch
import re
import config
from model_loader import tokenizer, model
from text_utils import replace_with_placeholders, convert_numbers_to_script
from pdf_processor import split_block_into_subblocks

def get_dynamic_batch_size(num_texts, fast_mode):
    if config.DEVICE != "cuda":
        return min(8, num_texts)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    free_memory = total_memory - torch.cuda.memory_allocated()
    tokens_per_text = config.MAX_LENGTH_DEFAULT
    bytes_per_text = tokens_per_text * 4 * (3 if fast_mode else 1)
    max_batch = max(1, min(free_memory // bytes_per_text, num_texts))
    return min(16 if fast_mode else 4, max_batch)

def translate_batch(texts, target_language, fast_mode=False):
    if not texts:
        return []
    batch_size = get_dynamic_batch_size(len(texts), fast_mode)
    translated_texts = []
    target_lang_code = config.LANGUAGES[target_language]["code"]
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        max_length = max(config.MAX_LENGTH_DEFAULT, max(len(t.split()) for t in batch) * 2)
        try:
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(config.DEVICE)
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
            if config.DEVICE == "cuda":
                torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"Memory error: {e}. Reducing batch size and retrying...")
            if batch_size > 1:
                batch_size = max(1, batch_size // 2)
                translated_texts.extend(translate_batch(batch, target_language, fast_mode))
            else:
                raise
    return translated_texts

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
                    print(f"Restored '{original}' in '{translated_text}'")
                else:
                    translated_text += f" {original}"
                    print(f"Appended '{original}' to '{translated_text}'")
            translated_text = convert_numbers_to_script(translated_text, target_language)
            print(f"Converted numbers in '{translated_text}'")
            subblock["translated_text"] = translated_text 