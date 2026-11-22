"""Put back markup of the source book that a translation lost.

A model copies a paragraph's tags by hand and now and then slips: an empty page
marker disappears (and the reader's page list points nowhere), a marker moves to the
next paragraph, a class changes. None of that is a matter of language, so it is taken
from the source when the book is built, without asking a model and without touching
a word of the text:

- a paragraph carries exactly the ids its source paragraph carries;
- its own attributes are the source's, and so are those of an inner element where
  there is no doubt which source element it is;
- attributes that hold words (`alt`, `title`) stay as translated.

An inline element that is gone with its text inside (an `<i>` around one word) is
not restored: which Russian word it belongs on can't be told from markup.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence

from lxml import etree

from .epub_io import TRANSLATABLE_TAGS, ChapterDoc

# Attributes with words for the reader in them: the translated value is the right one.
_TEXT_ATTRIBUTES = frozenset({"alt", "title", "aria-label"})


def restore_book_markup(
    chapters: Sequence[ChapterDoc], source_chapters: Sequence[ChapterDoc]
) -> list[tuple[str, int, list[str]]]:
    """Restore every paragraph of a translated book from a freshly read source.

    Both are lists of `read_book_structured` chapters of the same EPUB. Returns
    (file, paragraph number, what was restored) for each paragraph changed.
    """
    report: list[tuple[str, int, list[str]]] = []
    for chapter, original in zip(chapters, source_chapters, strict=True):
        pairs = zip(original.paragraphs, chapter.paragraphs, strict=True)
        for number, (source, translated) in enumerate(pairs, start=1):
            if done := restore_markup(source, translated):
                report.append((chapter.archive_name, number, done))
    return report


def restore_markup(source: etree._Element, translated: etree._Element) -> list[str]:
    """Restore one paragraph in place; what was done, empty when nothing was wrong."""
    done: list[str] = []
    if _copy_attributes(source, translated):
        done.append("paragraph attributes")
    done += _drop_foreign_ids(source, translated)
    if _restore_inner_attributes(source, translated):
        done.append("inner attributes")
    done += _insert_lost_ids(source, translated)
    return done


def _elements(root: etree._Element) -> list[etree._Element]:
    """The elements inside `root`, without comments and entities."""
    return [el for el in root.iterdescendants() if isinstance(el.tag, str)]


def _local_name(el: etree._Element) -> str:
    return etree.QName(el.tag).localname.lower()


def _is_empty(el: etree._Element) -> bool:
    return not (el.text or "").strip() and len(el) == 0


def _copy_attributes(source: etree._Element, target: etree._Element) -> bool:
    wanted = dict(source.attrib)
    for name in _TEXT_ATTRIBUTES & set(source.attrib) & set(target.attrib):
        wanted[name] = target.attrib[name]
    if wanted == dict(target.attrib):
        return False
    target.attrib.clear()
    target.attrib.update(wanted)
    return True


def _remove_keeping_tail(el: etree._Element) -> None:
    parent = el.getparent()
    tail = el.tail or ""
    previous = el.getprevious()
    if previous is not None:
        previous.tail = (previous.tail or "") + tail
    elif tail:
        parent.text = (parent.text or "") + tail
    el.tail = None
    parent.remove(el)


def _drop_foreign_ids(source: etree._Element, translated: etree._Element) -> list[str]:
    """Ids of another paragraph, made up, or printed twice."""
    allowed = {el.get("id") for el in [source, *_elements(source)] if el.get("id")}
    seen = {translated.get("id")}
    done: list[str] = []
    for el in _elements(translated):
        el_id = el.get("id")
        if not el_id:
            continue
        if el_id in allowed and el_id not in seen:
            seen.add(el_id)
            continue
        if _is_empty(el):
            _remove_keeping_tail(el)
        else:
            del el.attrib["id"]
        done.append(f"-{el_id}")
    return done


def _restore_inner_attributes(source: etree._Element, translated: etree._Element) -> bool:
    """An element is matched by its id, or when every candidate for it is the same."""
    originals = _elements(source)
    copies = _elements(translated)
    by_id = {el.get("id"): el for el in originals if el.get("id")}
    matched = [(by_id[el.get("id")], el) for el in copies if el.get("id") in by_id]
    found = {el.get("id") for _, el in matched}

    left: dict[tuple[str, bool], list[etree._Element]] = defaultdict(list)
    for el in originals:
        # A lost marker is inserted where it belongs, never made out of another element.
        if el.get("id") and (el.get("id") in found or _is_empty(el)):
            continue
        left[_local_name(el), _is_empty(el)].append(el)
    unmatched: dict[tuple[str, bool], list[etree._Element]] = defaultdict(list)
    for el in copies:
        if el.get("id") not in found:
            unmatched[_local_name(el), _is_empty(el)].append(el)

    for kind, group in left.items():
        others = unmatched.get(kind, [])
        if len(group) != len(others):
            continue
        # A translation reorders inline elements, so position proves nothing for them;
        # paragraphs inside a quote keep their order.
        alike = all(dict(el.attrib) == dict(group[0].attrib) for el in group)
        if alike or kind[0] in TRANSLATABLE_TAGS:
            matched += zip(group, others, strict=True)

    changed = [_copy_attributes(original, copy) for original, copy in matched]
    return any(changed)


def _insert_lost_ids(source: etree._Element, translated: etree._Element) -> list[str]:
    present = {el.get("id") for el in [translated, *_elements(translated)]}
    namespace = etree.QName(translated.tag).namespace
    done: list[str] = []
    for el in _elements(source):
        el_id = el.get("id")
        if not el_id or el_id in present:
            continue
        if _is_empty(el):
            marker = etree.Element(el.tag, attrib=dict(el.attrib), nsmap=el.nsmap)
        else:
            # Its text is translated somewhere in the paragraph; links need the target only.
            tag = f"{{{namespace}}}span" if namespace else "span"
            marker = etree.Element(tag, attrib={"id": el_id}, nsmap=el.nsmap)
        _insert_at_share(translated, marker, _share_before(source, el))
        present.add(el_id)
        done.append(f"+{el_id}")
    return done


def _share_before(root: etree._Element, marker: etree._Element) -> float:
    """How much of the paragraph's text stands before `marker`, from 0 to 1."""
    total = len("".join(root.itertext()))
    before = 0
    for event, el in etree.iterwalk(root, events=("start", "end")):
        if event == "start":
            if el is marker:
                break
            before += len(el.text or "")
        elif el is not root:
            before += len(el.tail or "")
    return min(before / total, 1.0) if total else 0.0


def _text_places(
    el: etree._Element, depth: int = 0
) -> Iterator[tuple[etree._Element | None, str, str, int]]:
    """Every run of text as (owner, "text" or "tail", text, depth).

    The owner is None inside a link: nothing is inserted there.
    """
    yield el, "text", el.text or "", depth
    for child in el:
        if isinstance(child.tag, str) and (child.text or len(child)):
            if _local_name(child) == "a":
                yield None, "", "".join(child.itertext()), depth
            else:
                yield from _text_places(child, depth + 1)
        yield child, "tail", child.tail or "", depth


def _insert_at_share(root: etree._Element, marker: etree._Element, share: float) -> None:
    """Insert between two words, as near to `share` of the text as that allows."""
    runs = list(_text_places(root))
    whole = "".join(text for _, _, text, _ in runs)
    wanted = share * len(whole)

    best_rank: tuple[float, int] | None = None
    best: tuple[etree._Element, str, str, int] | None = None
    start = 0
    for owner, where, text, depth in runs:
        if owner is not None:
            for cut in range(len(text) + 1):
                at = start + cut
                between_words = (
                    at in (0, len(whole)) or whole[at - 1].isspace() and not whole[at].isspace()
                )
                rank = (abs(at - wanted), depth)
                # `<=`: of equal places the later one, so that markers keep their order.
                if between_words and (best_rank is None or rank <= best_rank):
                    best_rank, best = rank, (owner, where, text, cut)
        start += len(text)
    if best is None:  # no gap between words outside a link
        marker.tail = root.text
        root.text = None
        root.insert(0, marker)
        return

    owner, where, text, cut = best
    if where == "text":
        owner.text = text[:cut] or None
        marker.tail = text[cut:] or None
        owner.insert(0, marker)
    else:
        parent = owner.getparent()
        position = parent.index(owner) + 1
        owner.tail = text[:cut] or None
        marker.tail = text[cut:] or None
        parent.insert(position, marker)
