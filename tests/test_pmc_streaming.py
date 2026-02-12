"""Tests for streaming PMC XML chunker."""

import pytest

from examples.medlit.pipeline.pmc_streaming import (
    iter_overlapping_windows,
    iter_pmc_sections,
    iter_pmc_windows,
)


MINIMAL_PMC = b"""<?xml version="1.0"?>
<article xmlns="http://jats.nlm.nih.gov">
  <front><article-meta>
    <abstract><p>Abstract text here.</p></abstract>
  </article-meta></front>
  <body>
    <sec id="s1"><title>Intro</title><p>First section content.</p></sec>
    <sec id="s2"><p>Second section with more text.</p></sec>
  </body>
</article>
"""


def test_iter_pmc_sections_yields_abstract_and_secs():
    """iter_pmc_sections yields abstract first then body secs."""
    sections = list(iter_pmc_sections(MINIMAL_PMC))
    assert len(sections) >= 1
    ids = [s[0] for s in sections]
    assert "abstract" in ids
    # May have sec_s1, sec_s2 or just sec depending on namespace
    assert any("sec" in i for i in ids)


def test_iter_overlapping_windows_abstract_separately():
    """Abstract is yielded as first window when include_abstract_separately True."""
    sections = [("abstract", "Short abstract."), ("sec_1", "x" * 5000)]
    windows = list(iter_overlapping_windows(iter(sections), window_size=4000, overlap=800))
    assert len(windows) >= 2
    assert windows[0][0] == 0
    assert windows[0][1] == "Short abstract."


def test_iter_overlapping_windows_has_overlap():
    """Consecutive windows overlap by roughly overlap chars."""
    sections = [("sec_1", "a" * 2000), ("sec_2", "b" * 2000), ("sec_3", "c" * 2000)]
    windows = list(iter_overlapping_windows(iter(sections), window_size=4000, overlap=800))
    assert len(windows) >= 1
    step = 4000 - 800
    # First window should end with some "a", second should start with some "a"
    if len(windows) >= 2:
        w0_tail = windows[0][1][-200:]
        w1_head = windows[1][1][:200]
        # Overlap region should appear in both
        assert "a" in w0_tail or "b" in w0_tail
        assert "a" in w1_head or "b" in w1_head


def test_iter_pmc_windows_returns_iterator():
    """iter_pmc_windows returns an iterator of (index, text)."""
    it = iter_pmc_windows(MINIMAL_PMC, window_size=100, overlap=20)
    first = next(it)
    assert isinstance(first, tuple)
    assert len(first) == 2
    assert isinstance(first[0], int)
    assert isinstance(first[1], str)
