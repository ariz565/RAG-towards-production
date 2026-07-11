from pathlib import Path

import pytest

from second_brain.store import SandboxViolation, WikiStore


def _store(tmp_path: Path) -> WikiStore:
    return WikiStore(tmp_path)


def test_write_and_read_page_roundtrip(tmp_path):
    s = _store(tmp_path)
    s.write_page("concepts/foo.md", title="Foo", tldr="A thing.", content="Body text.")
    body = s.read_page_body("concepts/foo.md")
    assert "Body text." in body
    assert "**TLDR**: A thing." in body
    meta = s.page_meta("concepts/foo.md")
    assert meta.title == "Foo"
    assert meta.tldr == "A thing."


@pytest.mark.parametrize("bad_path", [
    "index.md",                  # not under concepts/ or entities/
    "../outside.md",
    "concepts/../../evil.md",    # prefix looks fine, but resolves outside wiki/
    "concepts/" + "../" * 5 + "evil.md",
])
def test_write_page_rejects_paths_outside_sandbox(tmp_path, bad_path):
    s = _store(tmp_path)
    with pytest.raises(SandboxViolation):
        s.write_page(bad_path, title="x", tldr="x", content="x")
    assert not (tmp_path / "evil.md").exists()


def test_write_page_allows_nested_subdirectory_under_concepts(tmp_path):
    s = _store(tmp_path)
    s.write_page("concepts/sub/x.md", title="X", tldr="x", content="x")
    assert s.read_page("concepts/sub/x.md") is not None


def test_ingest_ledger_idempotency(tmp_path):
    s = _store(tmp_path)
    raw = s.raw_dir / "a.md"
    raw.write_text("hello world")
    assert not s.already_ingested(raw)
    s.mark_ingested(raw)
    assert s.already_ingested(raw)

    raw.write_text("hello world, changed")
    assert not s.already_ingested(raw)  # content hash changed -> no longer considered ingested


def test_list_pages(tmp_path):
    s = _store(tmp_path)
    s.write_page("concepts/a.md", title="A", tldr="a", content="a")
    s.write_page("entities/b.md", title="B", tldr="b", content="b")
    assert s.list_pages() == ["concepts/a.md", "entities/b.md"]


def test_broken_links_detected(tmp_path):
    s = _store(tmp_path)
    s.write_page("concepts/a.md", title="A", tldr="a", content="See [[concepts/missing.md]] for more.")
    broken = s.broken_links()
    assert ("concepts/a.md", "concepts/missing.md") in broken


def test_broken_links_empty_when_target_exists(tmp_path):
    s = _store(tmp_path)
    s.write_page("concepts/b.md", title="B", tldr="b", content="b")
    s.write_page("concepts/a.md", title="A", tldr="a", content="See [[concepts/b.md]].")
    assert s.broken_links() == []


def test_stale_pages_detects_changed_source(tmp_path):
    s = _store(tmp_path)
    raw = s.raw_dir / "source.md"
    raw.write_text("original content")
    s.write_page("concepts/a.md", title="A", tldr="a", content="a", sources=["raw/source.md"])
    assert s.stale_pages() == []

    raw.write_text("changed content")
    assert s.stale_pages() == ["concepts/a.md"]


def test_stale_pages_detects_deleted_source(tmp_path):
    s = _store(tmp_path)
    raw = s.raw_dir / "source.md"
    raw.write_text("original content")
    s.write_page("concepts/a.md", title="A", tldr="a", content="a", sources=["raw/source.md"])
    raw.unlink()
    assert s.stale_pages() == ["concepts/a.md"]


def test_append_log_is_grep_parseable(tmp_path):
    s = _store(tmp_path)
    s.append_log("ingest", "some-file.md", "did a thing")
    log_text = (s.wiki_dir / "log.md").read_text(encoding="utf-8")
    assert "## [" in log_text
    assert "ingest | some-file.md" in log_text
