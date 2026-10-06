#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Tuesday, April 30th 2024, 1:56:20 pm
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###


import re
import jiwer

from audio_bench.dataset_src.text_normalizer import whisper_english

all_jiwer_process = jiwer.Compose([
    jiwer.RemoveMultipleSpaces(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemoveKaldiNonWords(),
    jiwer.RemovePunctuation()
])

#EnglishNumberNormalizer   = whisper_english.EnglishNumberNormalizer()
#EnglishSpellingNormalizer = whisper_english.EnglishSpellingNormalizer()
EnglishTextNormalizer     = whisper_english.EnglishTextNormalizer()
IMDAPART4TextNormalizer = whisper_english.IMDAPART4TextNormalizer()


def normalize_text(text):
    """ Normalize text by converting to lowercase and standardizing numbers. """

    # Digit to word conversions
    digits_to_words = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
        '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
        '14': 'fourteen', '15': 'fifteen', '16': 'sixteen',
        '17': 'seventeen', '18': 'eighteen', '19': 'nineteen',
        '20': 'twenty', '30': 'thirty', '40': 'forty', '50': 'fifty',
        '60': 'sixty', '70': 'seventy', '80': 'eighty', '90': 'ninety',
    }
    for digit, word in digits_to_words.items():
        text = re.sub(r'\b' + digit + r'\b', word, text)

    # Expand common contractions
    contractions = {
        "i'm": "i am",
        "you're": "you are",
        "he's": "he is",
        "she's": "she is",
        "it's": "it is",
        "we're": "we are",
        "they're": "they are",
        "i've": "i have",
        "you've": "you have",
        "we've": "we have",
        "they've": "they have",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "doesn't": "does not",
        "don't": "do not",
        "didn't": "did not",
        "that's": "that is",
    }
    for contraction, expanded in contractions.items():
        text = re.sub(r'\b' + contraction + r'\b', expanded, text)

    return text

# def remove_punctuation(text):
#     """ Remove punctuation from text. """
#     text = re.sub(r'[^\w\s]', '', text)  # Remove all except word characters and whitespace
#     return text

def remove_non_speech_elements(text):
    """ Remove common non-speech elements like 'uh', 'um', etc. """
    non_speech_patterns = r'\b(uh|umm|um|er|ah)\b'
    text = re.sub(non_speech_patterns, '', text)
    return text

def remove_parentheses(text):
    return re.sub(r'(\[|\(|\{|\<)[^\(\)\\n\[\]]*(\]|\)|\}|\>)', "", text)

# Non-English number normalization. The English pipeline below already standardizes English
# numbers, but on other languages it turns digits into English words ("12" -> "twelve"), so a
# French "douze" never matches a "12". Spell digits out in the dataset language first instead.
_DECIMAL_WORDS = {
    "fr": "virgule", "de": "komma", "es": "coma", "it": "virgola", "pt": "vírgula",
    "nl": "komma", "ar": "فاصلة",
}
# Thousands groups: "10 000" (fr, with plain / non-breaking / narrow spaces) or "10.000".
_THOUSANDS_RE = re.compile(r"\b\d{1,3}(?:[ .  ]\d{3})+\b")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def spell_out_numbers(text, language):
    """Write the digits of `text` as words in `language` ("FR", or the target of "FR-EN").

    English and unknown languages are returned unchanged.
    """
    lang = (language or "").lower().split("-")[-1]
    if lang not in _DECIMAL_WORDS:
        return text
    from num2words import num2words

    def spell_int(digits):
        return num2words(int(digits), lang=lang)

    def repl(m):
        parts = re.split(r"[.,]", m.group(), maxsplit=1)
        try:
            words = spell_int(parts[0])
            if len(parts) == 2:
                # Keep leading zeros of the decimals: "0,05" -> "zéro virgule zéro cinq".
                decimals = parts[1]
                zeros = len(decimals) - len(decimals.lstrip("0"))
                tail = [spell_int("0")] * zeros + ([spell_int(decimals)] if decimals.lstrip("0") else [])
                words = " ".join([words, _DECIMAL_WORDS[lang]] + tail)
        except (ValueError, NotImplementedError, OverflowError):
            return m.group()
        return f" {words} "

    text = _THOUSANDS_RE.sub(lambda m: re.sub(r"\D", "", m.group()), text)
    return re.sub(r"\s+", " ", _NUMBER_RE.sub(repl, text)).strip()


def preprocess_text_asr(text, language=None):

    # Spell non-English numbers out in their own language (see spell_out_numbers)
    text = spell_out_numbers(text, language)

    # All Adapt to Lower Case
    text = text.lower()

    # Borrow from Whisper
    # text = EnglishNumberNormalizer(text)
    # text = EnglishSpellingNormalizer(text)
    text = EnglishTextNormalizer(text)

    # Should be handled by EnglishNumberNormalizer and EnglishTextNormalizer
    # Does not hurt to have it here
    text = normalize_text(text)

    # Use regex to remove all things between brackets [] () {} <> 
    text = remove_parentheses(text)

    # Add some standard process from jiwer
    text = all_jiwer_process(text)

    # Remove some non-speech elements
    text = remove_non_speech_elements(text).strip()

    return text



def separate_and_space_chinese(text):
    # Separate Chinese characters and non-Chinese parts
    parts = re.split(r'([\u4e00-\u9fff]+)', text)
    
    # Add spaces between Chinese characters and join the parts back together
    processed_parts = []
    for part in parts:
        if re.match(r'[\u4e00-\u9fff]+', part):
            spaced = ' '.join(char for char in part)  # Add space between Chinese characters
            processed_parts.append(spaced)
        else:
            processed_parts.append(part)
    
    return ''.join(processed_parts)

def preprocess_text_asr_code_switch_chinese(text):
    """ Preprocess text for ASR tasks with code-switching between Chinese and English. """

    # All Adapt to Lower Case
    text = text.lower()

    # Borrow from Whisper
    # text = EnglishNumberNormalizer(text)
    # text = EnglishSpellingNormalizer(text)
    text = EnglishTextNormalizer(text)

    # Should be handled by EnglishNumberNormalizer and EnglishTextNormalizer
    # Does not hurt to have it here
    text = normalize_text(text)

    # Use regex to remove all things between brackets [] () {} <> 
    text = remove_parentheses(text)

    # Add some standard process from jiwer
    text = all_jiwer_process(text)

    # Remove some non-speech elements
    text = remove_non_speech_elements(text).strip()

    # Separate Chinese characters
    text = separate_and_space_chinese(text)

    return text



def preprocess_text_asr_code_imda_part4(text):
    """ Post-processing text for WER computation. 3 language code-switching from IMDA PART4"""

    # All adapt to lower case
    text = text.lower()

    # Borrow from Whisper
    text = IMDAPART4TextNormalizer(text)

    # Should be handled by EnglishNumberNormalizer and EnglishTextNormalizer
    # Does not hurt to have it here
    text = normalize_text(text)

    # Use regex to remove all things between brackets [] () {} <> 
    text = remove_parentheses(text)

    # Add some standard process from jiwer
    text = all_jiwer_process(text)

    # Remove some non-speech elements
    text = remove_non_speech_elements(text).strip()

    # Separate Chinese characters
    text = separate_and_space_chinese(text)

    # replace any successive whitespaces with a space
    text = re.sub(r"\s+", " ", text)  

    return text