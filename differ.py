"""
differ.py — Token-level diff between two Federal Register XML snapshots.

Strategy:
  1. Parse both XML documents and extract <P> / text-bearing elements per document entry.
  2. For each document (keyed by document-number), diff the text at the paragraph level.
  3. Within each changed paragraph, perform a word-token diff to surface what changed.
  4. Return structured diff records ready for classification.
"""

import hashlib
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterator
from xml.etree import ElementTree as ET


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DocEntry:
    document_number: str
    agency: str
    paragraphs: list[str]  # ordered list of paragraph texts


@dataclass
class DiffRecord:
    document_number: str
    agency: str
    diff_type: str          # 'added' | 'removed' | 'modified'
    old_text: str | None
    new_text: str | None
    paragraph_hash: str     # SHA-256 of new_text (or old_text for removals)
    token_diff: list[tuple]  # (op, value) pairs from token-level diff


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

_FRNS = "http://www.gpoinc.com/schema/frdoc-1-0"  # approximate namespace

# Federal Register XML uses inconsistent namespacing across years — strip all namespace prefixes before tag comparison
def _strip_ns(tag: str) -> str:
    return re.sub(r"\{[^}]*\}", "", tag)


def _iter_text(elem: ET.Element) -> str:
    """Concatenate all text within an element."""
    return " ".join((elem.itertext()))


def parse_fedregister_xml(xml_bytes: bytes) -> dict[str, DocEntry]:
    """
    Parse a Federal Register bulk XML file.
    Returns a dict keyed by document-number → DocEntry.
    Falls back gracefully when namespace or structure differs.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        raise ValueError(f"XML parse error: {e}") from e

    entries: dict[str, DocEntry] = {}

    # Walk all elements looking for document containers.
    # The FR bulk XML wraps each notice/rule in <RULE>, <NOTICE>, <PRORULE> etc.
    # PRESDOCU captures presidential documents; CORRECT captures errata/corrections.
    doc_tags = {"RULE", "NOTICE", "PRORULE", "PRESDOCU", "CORRECT", "EXEC"}

    for elem in root.iter():
        tag = _strip_ns(elem.tag).upper()
        if tag not in doc_tags:
            continue

        doc_num = _extract_text(elem, ["FRDOC", "DOCNUM", "FRNUM"]) or f"UNKNOWN-{id(elem)}"
        agency = _extract_text(elem, ["AGENCY", "AGYNAME", "SUBAGY"]) or "Unknown Agency"

        paragraphs: list[str] = []
        for child in elem.iter():
            ctag = _strip_ns(child.tag).upper()
            if ctag in {"P", "FP", "HD", "E", "AMDPAR"}:
                txt = " ".join(child.itertext()).strip()
                if txt:
                    paragraphs.append(txt)

        if paragraphs:
            entries[doc_num] = DocEntry(
                document_number=doc_num,
                agency=agency,
                paragraphs=paragraphs,
            )

    return entries


def _extract_text(elem: ET.Element, tags: list[str]) -> str | None:
    for tag in tags:
        child = elem.find(f".//{tag}")
        if child is None:
            # try stripped namespace
            for c in elem.iter():
                if _strip_ns(c.tag).upper() == tag.upper():
                    child = c
                    break
        if child is not None:
            txt = " ".join(child.itertext()).strip()
            if txt:
                return txt
    return None


# ---------------------------------------------------------------------------
# Token-level diff
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Split on word boundaries keeping punctuation as separate tokens."""
    return re.findall(r"\w+|[^\w\s]", text)


def token_diff(old: str, new: str) -> list[tuple[str, str]]:
    """
    Return a list of (op, token) pairs where op ∈ {equal, insert, delete}.
    """
    old_toks = _tokenise(old)
    new_toks = _tokenise(new)
    sm = SequenceMatcher(None, old_toks, new_toks, autojunk=False)
    result: list[tuple[str, str]] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for tok in old_toks[i1:i2]:
                result.append(("equal", tok))
        elif op in ("replace", "delete"):
            for tok in old_toks[i1:i2]:
                result.append(("delete", tok))
            if op == "replace":
                for tok in new_toks[j1:j2]:
                    result.append(("insert", tok))
        elif op == "insert":
            for tok in new_toks[j1:j2]:
                result.append(("insert", tok))
    return result


# ---------------------------------------------------------------------------
# Document-level diff
# ---------------------------------------------------------------------------

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def diff_publications(
    old_entries: dict[str, DocEntry],
    new_entries: dict[str, DocEntry],
) -> list[DiffRecord]:
    """
    Compare two publication snapshots and return a list of DiffRecords.
    """
    records: list[DiffRecord] = []

    all_doc_nums = set(old_entries) | set(new_entries)

    for doc_num in all_doc_nums:
        old_entry = old_entries.get(doc_num)
        new_entry = new_entries.get(doc_num)
        agency = (new_entry or old_entry).agency  # type: ignore[union-attr]

        if old_entry is None and new_entry is not None:
            # Entirely new document — every paragraph is 'added'
            for para in new_entry.paragraphs:
                records.append(DiffRecord(
                    document_number=doc_num,
                    agency=agency,
                    diff_type="added",
                    old_text=None,
                    new_text=para,
                    paragraph_hash=_sha(para),
                    token_diff=token_diff("", para),
                ))
        elif old_entry is not None and new_entry is None:
            # Document disappeared — every paragraph is 'removed'
            for para in old_entry.paragraphs:
                records.append(DiffRecord(
                    document_number=doc_num,
                    agency=agency,
                    diff_type="removed",
                    old_text=para,
                    new_text=None,
                    paragraph_hash=_sha(para),
                    token_diff=token_diff(para, ""),
                ))
        else:
            # Document exists in both — paragraph-level alignment
            assert old_entry and new_entry
            records.extend(_diff_paragraphs(old_entry, new_entry))

    return records


def _diff_paragraphs(old: DocEntry, new: DocEntry) -> list[DiffRecord]:
    """Align paragraphs by index (simple) and diff each pair."""
    records: list[DiffRecord] = []
    sm = SequenceMatcher(None, old.paragraphs, new.paragraphs, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        if op in ("replace", "delete"):
            for para in old.paragraphs[i1:i2]:
                if op == "delete":
                    records.append(DiffRecord(
                        document_number=old.document_number,
                        agency=old.agency,
                        diff_type="removed",
                        old_text=para,
                        new_text=None,
                        paragraph_hash=_sha(para),
                        token_diff=token_diff(para, ""),
                    ))
        if op in ("replace", "insert"):
            paired_old = old.paragraphs[i1:i2] if op == "replace" else []
            for k, para in enumerate(new.paragraphs[j1:j2]):
                old_para = paired_old[k] if k < len(paired_old) else None
                dtype = "modified" if old_para else "added"
                records.append(DiffRecord(
                    document_number=new.document_number,
                    agency=new.agency,
                    diff_type=dtype,
                    old_text=old_para,
                    new_text=para,
                    paragraph_hash=_sha(para),
                    token_diff=token_diff(old_para or "", para),
                ))
    return records
