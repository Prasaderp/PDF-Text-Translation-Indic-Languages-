import re
import config

def parse_user_entities(user_input):
    entities = [e.strip() for e in user_input.split(',') if e.strip()]
    print(f"Entities to preserve: {', '.join(entities) if entities else 'None'}")
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
            print(f"Replaced '{entity}' {count} time(s)")

    print(f"Modified text with placeholders: '{modified_text}'")
    return modified_text, placeholder_map

def convert_numbers_to_script(text, target_language):
    """Convert Latin digits to the target language's numeral script."""
    digit_map = config.DIGIT_MAP[target_language]
    def replace_digit(match):
        number = match.group(0)
        converted = ''.join(digit_map[int(d)] if d in config.LATIN_DIGITS else d for d in number)
        return converted
    pattern = re.compile(r'(?<!__PRESERVE\d{3}__)\b\d+(?:\.\d+)?\b(?![^_]*__)')
    return pattern.sub(replace_digit, text) 