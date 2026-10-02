"""Post-processing for chatty ASR predictions (e.g. Qwen3-Omni Instruct).

Qwen3-Omni frequently answers an ASR prompt conversationally: a preamble followed by the
real transcript wrapped in quotes ("..." / «...»), sometimes inside markdown. This module
recovers the transcript.

It is deliberately conservative -- on a plain transcript (by far the common case) it is a
no-op, and it NEVER strips on preamble *words* alone: many real transcripts legitimately
begin with "Oui" / "Alors" / "Donc". It only edits on unambiguous structure:

  1. a leading [mm:ss] / (mm:ss) timestamp marker;
  2. the first quoted span (angle «», curly "", or straight "");
  3. a meta "answer intro" that names the task and ends in ':' + newline (no quotes case).

Anything else is returned unchanged -- including outputs where the model described the
audio instead of transcribing it, which are genuine failures and should score as such.
"""

import re

# Quote pairs, tried together; the earliest-starting span in the text wins.
_QUOTE_PAIRS = [("«", "»"), ("„", "“"), ("“", "”"), ('"', '"')]

# Leading "[00:00]" / "(0:00)" / "[00:00:00]" timestamp marker.
_LEADING_TS = re.compile(r"^\s*[\[(]\s*\d{1,2}:\d{2}(?::\d{2})?\s*[\])]\s*")

# A short intro ending in ':' then a line break, capturing the rest.
_COLON_BLOCK = re.compile(r"^(.{0,300}?):[ \t]*\n(.+)$", re.S)

# Meta words that mark a chatty intro (never present in a real transcript's intro).
_META = re.compile(
    r"transcri|audio|enregistrement|\btexte\b|voici|entends|\bmessage\b|fichier|\bclip\b",
    re.I,
)


def _first_quoted(text):
    """Return the content of the earliest-starting quoted span, or None.

    Only fires when the text BEFORE the quote is a chatty lead-in (empty, or containing a
    meta word). This avoids truncating a plain transcript that merely contains an incidental
    quotation, e.g. `... now known as "Dunlap broadsides".` -- there the pre-quote text is
    real content with no meta word, so we leave it alone (the normalizer drops the quote
    marks anyway).
    """
    best = None
    for open_q, close_q in _QUOTE_PAIRS:
        i = text.find(open_q)
        if i < 0:
            continue
        j = text.find(close_q, i + len(open_q))
        if j < 0:
            continue
        preamble = text[:i]
        trailing = text[j + len(close_q):].strip(" \t\r\n*>.")
        # Language-agnostic gate for "the transcript, presented as a quoted block":
        #  * a preamble is only a chatty intro if a blank line separates it from the quote
        #    (`... audio:\n\n"..."`). A same-line lead-in is real speech -- an inline
        #    quotation (`... from his office: "..."`, `known as "..."`) -- so skip it.
        #  * a bare leading quote counts only if it is essentially the whole output; if real
        #    text follows the close-quote (`"...," his heart said.`) it is a quotation inside
        #    a transcript, not the transcript, so skip it.
        pre = preamble.rstrip(" \t\r\n*>")
        if pre:
            if "\n\n" not in preamble:
                continue
        elif len(trailing) > 2:
            continue
        content = text[i + len(open_q):j].strip()
        if content and (best is None or i < best[0]):
            best = (i, content)
    return best[1] if best else None


def _strip_markup(s):
    """Peel surrounding markdown blockquote/bold from a recovered span."""
    return s.strip().lstrip(">").strip().strip("*").strip()


def postprocess_asr_prediction(text):
    """Recover the transcript from a possibly-chatty ASR prediction.

    Conservative: a plain transcript is returned unchanged (only stripped).
    """
    if not text:
        return text

    t = _LEADING_TS.sub("", text.strip()).strip()

    # 1) transcript wrapped in quotes -> take the first quoted span.
    quoted = _first_quoted(t)
    if quoted is not None:
        return _strip_markup(quoted)

    # 2) no quotes, but a meta "here is the transcription:" intro -> take the tail.
    m = _COLON_BLOCK.match(t)
    if m and _META.search(m.group(1)):
        tail = _strip_markup(m.group(2))
        if tail:
            return tail

    # 3) plain transcript (or an unsalvageable description) -> leave as-is.
    return t


# --- Phi-4 -----------------------------------------------------------------------------------
# Phi-4 sometimes labels its ASR output ("Spoken text: ...", "Transcription: ..."). Strip only a
# single leading label; a plain transcript (no label) is returned unchanged.
_LABEL_PREFIX = re.compile(r"^\s*(?:spoken\s+text|spoken\s+words|transcription|transcript)\s*:\s*", re.I)


def strip_label_prefix(text):
    """Remove a leading 'Spoken text:' / 'Transcription:' label from an ASR prediction."""
    if not text:
        return text
    return _LABEL_PREFIX.sub("", text, count=1).strip()


# --- Audio-Flamingo --------------------------------------------------------------------------
# Audio-Flamingo prefixes transcripts with a description ("The spoken content of the audio is ...",
# "The audio contains ...", "The literal translation of the audio is: ..."). Strip the preamble and
# unwrap a quoted transcript. A plain transcript (no such preamble) is returned unchanged. Note:
# this only cleans the framing -- it cannot fix Audio-Flamingo translating instead of transcribing.
_AF_PREAMBLE = re.compile(
    r"^\s*(?:the spoken content of the audio is|the spoken content is|the audio contains|"
    r"the transcription of the audio is|the literal translation of the audio is|the audio says)"
    r"\s*:?\s*",
    re.I,
)
_WRAPPING_QUOTES = re.compile(r"^[\"“‘']\s*(.*?)\s*[\"”’']\s*$", re.S)


def strip_audio_description_preamble(text):
    """Recover the transcript from an Audio-Flamingo description-style ASR prediction."""
    if not text:
        return text
    t = text.strip()
    m = _AF_PREAMBLE.match(t)
    if not m:
        return t
    rest = t[m.end():].strip()
    q = _WRAPPING_QUOTES.match(rest)
    if q and q.group(1).strip():
        rest = q.group(1).strip()
    return rest if rest else t
