"""Neutralize "file" framing in ASR/AST instructions for chatty multimodal models.

Some prompt templates name the audio as a file ("the WAV file", "du fichier MP3",
"der MP3-Datei", "questo file", ...). Chat-style models (Qwen2.5-Omni, Qwen3-Omni) then latch
onto that and REFUSE -- "Je ne peux pas transcrire un fichier WAV" / "I can't access the file" --
instead of transcribing/translating. Measured on Qwen2.5-Omni: ~17% refusals on prompts that say
"fichier WAV" vs ~0.6% on prompts that say "texte parlé".

This rewrites ONLY the file-reference phrase to the word "audio", grammatically, per language.
The language is inferred from the determiner/preposition in the phrase itself (du/del/do/der/
het/the...), which is more reliable than any dataset language field (for AST that field is the
translation pair, not the instruction's language). Everything else -- the rest of the
instruction, the shared dataset prompts, and every other model -- is left untouched; it is
applied per-model, to ASR and AST only.

Design choices:
* The output never contains "file"/"recording" (either can trigger the "I can't open it"
  refusal); a format-qualified recording noun ("this FLAC recording", "enregistrement FLAC")
  becomes "audio" too, not "recording".
* A leading "File" is the French verb (filer, "pass/give") in "File le texte ..." -- NOT a file
  noun -- so such instructions are returned entirely untouched.
* Formats are limited to WAV/MP3/FLAC (AAC/OGG/M4A collide with real acronyms like "the AAC").
"""

import re

# Optional trailing type qualifier, e.g. " MP3", " de audio", " de áudio" (Romance word order).
_T = r"(?:\s+(?:de\s+|d')?(?:[aá]udio|WAV|MP3|FLAC))?"
# Optional leading/compound type, e.g. "WAV-" / "Audio" (German/Dutch compounds, contiguous ok).
_Tc = r"(?:[aá]udio|WAV|MP3|FLAC)?[-\s]?"

# Ordered (pattern, replacement). The determiner identifies the language and fixes the grammar.
_RULES = [
    # ---- French (fichier) ----
    (rf"\bdu fichier{_T}\b", "de l'audio"),
    (rf"\bde ce fichier{_T}\b", "de cet audio"),
    (rf"\bce fichier{_T}\b", "cet audio"),
    (rf"\ble fichier{_T}\b", "l'audio"),
    (r"\bde cet enregistrement[-\s](?:WAV|MP3|FLAC)\b", "de cet audio"),
    (r"\bcet enregistrement[-\s](?:WAV|MP3|FLAC)\b", "cet audio"),
    # ---- Spanish (archivo) ----
    (rf"\bdel archivo{_T}\b", "del audio"),
    (rf"\bde este archivo{_T}\b", "de este audio"),
    (rf"\beste archivo{_T}\b", "este audio"),
    (rf"\bel archivo{_T}\b", "el audio"),
    (r"\bde esta grabación[-\s](?:WAV|MP3|FLAC)\b", "de este audio"),
    (r"\besta grabación[-\s](?:WAV|MP3|FLAC)\b", "este audio"),
    # ---- Portuguese (arquivo) -> áudio ----
    (rf"\bdo arquivo{_T}\b", "do áudio"),
    (rf"\bdeste arquivo{_T}\b", "deste áudio"),
    (rf"\beste arquivo{_T}\b", "este áudio"),
    (rf"\bo arquivo{_T}\b", "o áudio"),
    (r"\bdesta gravação[-\s](?:WAV|MP3|FLAC)\b", "deste áudio"),
    # ---- Italian (file) -> elision dell'audio ----
    (rf"\bdel file{_T}\b", "dell'audio"),
    (rf"\bdi questo file{_T}\b", "di questo audio"),
    (rf"\bin questo file{_T}\b", "in questo audio"),
    (rf"\bquesto file{_T}\b", "questo audio"),
    (r"\bda questa registrazione[-\s](?:WAV|MP3|FLAC)\b", "da questo audio"),
    (r"\bquesta registrazione[-\s](?:WAV|MP3|FLAC)\b", "questo audio"),
    # ---- German (Datei / compound) -> capitalized "Audio", case by context ----
    (rf"\bder {_Tc}Datei\b", "des Audios"),            # genitive: "Version der ... Datei"
    (rf"\baus dieser {_Tc}Datei\b", "aus diesem Audio"),
    (rf"\bin dieser {_Tc}Datei\b", "in diesem Audio"),
    (rf"\bdieser {_Tc}Datei\b", "dieses Audios"),       # genitive default (e.g. "Teil dieser Datei")
    (rf"\bdiese {_Tc}Datei\b", "dieses Audio"),         # accusative: "übersetzen Sie diese Datei"
    (r"\bin dieser (?:WAV|MP3|FLAC)[-\s]Aufnahme\b", "in diesem Audio"),
    (r"\bdieser (?:WAV|MP3|FLAC)[-\s]Aufnahme\b", "diesem Audio"),
    # ---- Dutch (bestand) -> "audio" is a 'de'-word ----
    (rf"\bvan het {_Tc}bestand\b", "van de audio"),
    (rf"\bhet {_Tc}bestand\b", "de audio"),
    (r"\buit dit bestand\b", "uit deze audio"),
    (r"\bvan dit bestand\b", "van deze audio"),
    (r"\bdit bestand\b", "deze audio"),
    (r"\bin deze (?:WAV|MP3|FLAC)[-\s]opname\b", "in deze audio"),
    (r"\bdeze (?:WAV|MP3|FLAC)[-\s]opname\b", "deze audio"),
    # ---- English (file / recording) ----
    (rf"\bthis {_Tc}file\b", "this audio"),
    (rf"\bthe {_Tc}file\b", "the audio"),
    (r"\bthis (?:WAV|MP3|FLAC)[-\s]recording\b", "this audio"),
    (r"\bthis audio file\b", "this audio"),
    (r"\bthe audio file\b", "the audio"),
    # ---- generic fallbacks for anything left (keeps the output file-word-free) ----
    (rf"\b(?:fichier|archivo){_T}\b", "audio"),
    (rf"\barquivo{_T}\b", "áudio"),
    (rf"\b{_Tc}(?:datei|bestand)\b", "audio"),
]
_RULES = [(re.compile(p, re.I), r) for p, r in _RULES]


def neutralize_file_references(text):
    """Rewrite file-reference phrases in an ASR/AST instruction to a grammatical 'audio'."""
    # Exception: a leading "File" is the French verb (filer, "pass/give"), e.g.
    # "File le texte que tu as entendu ..." -- never a file noun. Leave it entirely untouched.
    if re.match(r"\s*File\b", text):
        return text
    t = text
    for pat, rep in _RULES:
        t = pat.sub(rep, t)
    # Collapse only a double space left by a removed word; never touch space-before-punctuation
    # (French legitimately writes "... entend ?").
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()
