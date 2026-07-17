from text_utils import detect_language, normalize_for_match, normalize_persian_text


def test_persian_normalization_unifies_arabic_characters():
    assert normalize_persian_text("يادگيري ماشين و كاربرد") == "یادگیری ماشین و کاربرد"


def test_normalize_for_match_removes_punctuation_and_half_spaces():
    assert normalize_for_match("آسیب‌ شناسی  دهان، فک") == "آسیب شناسی دهان فک"


def test_detect_language():
    assert detect_language("این یک متن فارسی است") == "Persian"
    assert detect_language("This is English") == "English"
