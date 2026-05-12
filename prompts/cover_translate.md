---
name: cover_translate
version: "1.0"
description: >
  Reference prompt for AI image models to translate text on a book cover.
  NOTE: This file is NOT loaded at runtime. The prompt is built
  programmatically in cover.py/_build_default_prompt() because image
  models use a different API (not chat completions with system/user split).
  This file serves as documentation of the prompt logic.
  To customize the prompt, use the `prompt_template` parameter in
  translate_cover() or pass a custom string via the Python API.
---

## Default prompt logic (built in cover.py)

This is a book cover image. Your task is to translate ALL text on this cover into {{ target_lang }}.

### If --title is provided:
Translated title to use: '{{ title }}'

### If --title is NOT provided:
Translate the book title yourself — choose a natural, idiomatic translation
that sounds like a real published book title.

### If --author is provided:
Author name on cover: '{{ author }}'

## Rules

- Replace ALL English text with the {{ target_lang }} translation
- Keep the EXACT same visual style, colors, layout, and artwork
- Keep the same font style and weight (bold, italic, etc.)
- Keep the same text positioning and size proportions
- Do NOT change any non-text elements (images, patterns, colors)
- The result should look like a professionally designed {{ target_lang }}-language book cover
- Preserve the original aspect ratio and composition
- If the translated text is longer than the original, slightly reduce font size to fit — do NOT overflow or crop text
- Subtitle, series name, and any other text should also be translated into {{ target_lang }}
