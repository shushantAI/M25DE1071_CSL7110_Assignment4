"""
Part 2 - Inverted index and a tiny search engine.

I have kept the class layout faithful to the assignment (MySet, Position,
WordEntry, PageIndex, PageEntry, MyHashTable, InvertedPageIndex,
SearchEngine), but written the whole thing in idiomatic Python.

Processing rules applied exactly as given:
  - every token is lowercased
  - connector words are NOT stored, but their position is counted so that
    the surviving words keep their "original" 1-based index
  - the listed punctuation symbols are each replaced with a single space
  - the given singular/plural pairs collapse to one canonical form
  - the three lists above are treated as exhaustive; no extra cleanup.

Run:
    python -m src.part2_websearch \
        --pages data/q2/webpages \
        --actions data/q2/actions.txt \
        --answers data/q2/answers.txt
"""

from __future__ import annotations
import argparse
import os
import re
from typing import Dict, List, Optional, Set


# -------------------------------------------------------------
# constants from the assignment
# -------------------------------------------------------------
CONNECTOR_WORDS: Set[str] = {
    "a", "an", "the", "they", "these", "this", "for", "is", "are", "was",
    "of", "or", "and", "does", "will", "whose",
}

# punctuation list from the PDF: { } [ ] < > = ( ) . , ; ' " ? # ! - :
PUNCT_CHARS = "{}[]<>=().,;'\"?#!-:"

# (plural -> singular) so we can canonicalise lookups and stored tokens
PLURAL_TO_SINGULAR: Dict[str, str] = {
    "stacks": "stack",
    "structures": "structure",
    "applications": "application",
}


# -------------------------------------------------------------
# MySet
# -------------------------------------------------------------
class MySet:
    """Thin wrapper over a Python set to mirror the required API."""

    def __init__(self, items=None):
        self._data: set = set(items) if items else set()

    def addElement(self, element) -> None:
        self._data.add(element)

    def union(self, otherSet: "MySet") -> "MySet":
        return MySet(self._data | otherSet._data)

    def intersection(self, otherSet: "MySet") -> "MySet":
        return MySet(self._data & otherSet._data)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __contains__(self, x):
        return x in self._data

    def to_sorted_list(self):
        return sorted(self._data)


# -------------------------------------------------------------
# Position  <page, word-index>
# -------------------------------------------------------------
class Position:
    __slots__ = ("_page", "_word_index")

    def __init__(self, p: "PageEntry", wordIndex: int):
        self._page = p
        self._word_index = wordIndex

    def getPageEntry(self) -> "PageEntry":
        return self._page

    def getWordIndex(self) -> int:
        return self._word_index

    def __repr__(self):
        return f"Position({self._page.getName()}, {self._word_index})"


# -------------------------------------------------------------
# WordEntry
# -------------------------------------------------------------
class WordEntry:
    def __init__(self, word: str):
        self._word = word
        self._positions: List[Position] = []

    def getWord(self) -> str:
        return self._word

    def addPosition(self, position: Position) -> None:
        self._positions.append(position)

    def addPositions(self, positions: List[Position]) -> None:
        self._positions.extend(positions)

    def getAllPositionsForThisWord(self) -> List[Position]:
        return list(self._positions)

    def getTermFrequency(self, pageName: str) -> int:
        return sum(
            1 for pos in self._positions
            if pos.getPageEntry().getName() == pageName
        )


# -------------------------------------------------------------
# PageIndex (per-page inverted index: word -> WordEntry)
# -------------------------------------------------------------
class PageIndex:
    def __init__(self):
        self._entries: Dict[str, WordEntry] = {}

    def addPositionForWord(self, word: str, p: Position) -> None:
        if word in self._entries:
            self._entries[word].addPosition(p)
        else:
            we = WordEntry(word)
            we.addPosition(p)
            self._entries[word] = we

    def getWordEntries(self) -> List[WordEntry]:
        return list(self._entries.values())

    def contains(self, word: str) -> bool:
        return word in self._entries

    def getWordEntry(self, word: str) -> Optional[WordEntry]:
        return self._entries.get(word)


# -------------------------------------------------------------
# tokenisation pipeline
# -------------------------------------------------------------
_PUNCT_RE = re.compile("[" + re.escape(PUNCT_CHARS) + "]")


def _tokenise(raw_text: str) -> List[str]:
    """Turn a raw file into a list of tokens using the rules from the PDF."""
    lowered = raw_text.lower()
    cleaned = _PUNCT_RE.sub(" ", lowered)
    return cleaned.split()


def _canonical(tok: str) -> str:
    """Map plurals in the exhaustive list to their singular form."""
    return PLURAL_TO_SINGULAR.get(tok, tok)


# -------------------------------------------------------------
# PageEntry
# -------------------------------------------------------------
class PageEntry:
    """Reads the webpage from disk and builds its PageIndex.

    Word position is 1-based and includes connector words in the count,
    but connector words themselves are never stored.
    """

    def __init__(self, pageName: str, pagesDir: str):
        self._name = pageName
        path = os.path.join(pagesDir, pageName)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

        tokens = _tokenise(raw)
        self._page_index = PageIndex()

        for idx, tok in enumerate(tokens, start=1):
            if tok in CONNECTOR_WORDS:
                # still counts towards the position, but is not stored
                continue
            canon = _canonical(tok)
            pos = Position(self, idx)
            self._page_index.addPositionForWord(canon, pos)

        self._num_tokens = len(tokens)  # used by TF if someone ever wants it

    def getName(self) -> str:
        return self._name

    def getPageIndex(self) -> PageIndex:
        return self._page_index


# -------------------------------------------------------------
# MyHashTable : word -> WordEntry across all pages
# -------------------------------------------------------------
class MyHashTable:
    def __init__(self):
        self._table: Dict[str, WordEntry] = {}

    def getHashIndex(self, s: str) -> int:
        # The assignment only requires "a hash function that maps a string
        # to the index of its word-entry"; Python's built-in hash is fine.
        return hash(s)

    def addPositionsForWord(self, w: WordEntry) -> None:
        key = w.getWord()
        if key in self._table:
            self._table[key].addPositions(w.getAllPositionsForThisWord())
        else:
            # Keep an independent WordEntry so edits to one page's index
            # don't leak into the global table.
            merged = WordEntry(key)
            merged.addPositions(w.getAllPositionsForThisWord())
            self._table[key] = merged

    def getWordEntry(self, word: str) -> Optional[WordEntry]:
        return self._table.get(word)


# -------------------------------------------------------------
# InvertedPageIndex
# -------------------------------------------------------------
class InvertedPageIndex:
    def __init__(self):
        self._hash = MyHashTable()
        self._pages: Dict[str, PageEntry] = {}

    def addPage(self, p: PageEntry) -> None:
        self._pages[p.getName()] = p
        for we in p.getPageIndex().getWordEntries():
            self._hash.addPositionsForWord(we)

    def getPagesWhichContainWord(self, s: str) -> MySet:
        we = self._hash.getWordEntry(s)
        result = MySet()
        if we is None:
            return result
        for pos in we.getAllPositionsForThisWord():
            result.addElement(pos.getPageEntry().getName())
        return result

    def getPage(self, name: str) -> Optional[PageEntry]:
        return self._pages.get(name)


# -------------------------------------------------------------
# SearchEngine  (the tiny driver glueing everything together)
# -------------------------------------------------------------
class SearchEngine:
    def __init__(self, pagesDir: str):
        self._index = InvertedPageIndex()
        self._pages_dir = pagesDir

    # --- actions ---------------------------------------------
    def _action_addPage(self, x: str) -> Optional[str]:
        page = PageEntry(x, self._pages_dir)
        self._index.addPage(page)
        return None  # addPage is silent, matching the expected answers.txt

    def _action_findPagesWithWord(self, x: str) -> str:
        w = _canonical(x.lower())
        hits = self._index.getPagesWhichContainWord(w)
        if len(hits) == 0:
            return f"No webpage contains word {x}"
        return ", ".join(hits.to_sorted_list())

    def _action_findPositions(self, x: str, y: str) -> str:
        page = self._index.getPage(y)
        if page is None:
            return f"No webpage {y} found"
        w = _canonical(x.lower())
        we = page.getPageIndex().getWordEntry(w)
        if we is None:
            return f"Webpage {y} does not contain word {x}"
        positions = sorted({pos.getWordIndex() for pos in we.getAllPositionsForThisWord()})
        return ", ".join(str(p) for p in positions)

    # --- dispatcher ------------------------------------------
    def performAction(self, actionMessage: str) -> Optional[str]:
        parts = actionMessage.strip().split()
        if not parts:
            return None
        cmd, args = parts[0], parts[1:]

        if cmd == "addPage" and len(args) == 1:
            return self._action_addPage(args[0])
        if cmd == "queryFindPagesWhichContainWord" and len(args) == 1:
            return self._action_findPagesWithWord(args[0])
        if cmd == "queryFindPositionsOfWordInAPage" and len(args) == 2:
            return self._action_findPositions(args[0], args[1])
        return f"Unknown action: {actionMessage}"


# -------------------------------------------------------------
# runner + answers-diff
# -------------------------------------------------------------
def run(pagesDir: str, actionsFile: str, answersFile: Optional[str]) -> None:
    engine = SearchEngine(pagesDir)

    with open(actionsFile, "r", encoding="utf-8") as f:
        actions = [line.strip() for line in f if line.strip()]

    produced: List[str] = []
    print("=" * 60)
    print("Running actions:")
    print("=" * 60)
    for act in actions:
        out = engine.performAction(act)
        if out is not None:          # addPage produces no visible output
            produced.append(out)
            print(out)

    if answersFile and os.path.exists(answersFile):
        with open(answersFile, "r", encoding="utf-8") as f:
            # strip trailing CR-LF noise, drop blank lines
            expected = [ln.rstrip("\r\n").strip() for ln in f if ln.strip()]
        print("=" * 60)
        print("Diff vs answers.txt:")
        print("=" * 60)
        all_ok = True
        for i, (got, want) in enumerate(zip(produced, expected), start=1):
            ok = got.strip() == want.strip()
            all_ok &= ok
            flag = "OK" if ok else "MISMATCH"
            print(f"[{flag}] line {i}")
            if not ok:
                print(f"       got : {got!r}")
                print(f"       want: {want!r}")
        if len(produced) != len(expected):
            all_ok = False
            print(f"[MISMATCH] number of outputs: got {len(produced)}, expected {len(expected)}")
        print("=" * 60)
        print("ALL MATCH" if all_ok else "THERE ARE MISMATCHES -- review above")


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="data/q2/webpages")
    ap.add_argument("--actions", default="data/q2/actions.txt")
    ap.add_argument("--answers", default="data/q2/answers.txt")
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse()
    run(a.pages, a.actions, a.answers)
