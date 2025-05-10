import torch

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