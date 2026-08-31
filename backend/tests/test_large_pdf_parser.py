"""大体积校园手册的低内存解析路径。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.pipeline.parser import DocumentParser


def test_large_pdf_bypasses_pdfplumber_and_uses_pypdf(monkeypatch, tmp_path: Path):
    handbook = tmp_path / "large_demo.pdf"
    handbook.write_bytes(b"%PDF-1.4\n")
    handbook.write_bytes(b"%PDF-1.4\n" + b"0" * (32 * 1024 * 1024))
    parser = DocumentParser()
    pdfplumber_calls: list[Path] = []

    class FakePdfPlumber:
        @staticmethod
        def open(path: str):
            pdfplumber_calls.append(Path(path))
            raise AssertionError("large PDF must not use pdfplumber")

    monkeypatch.setitem(sys.modules, "pdfplumber", FakePdfPlumber)
    monkeypatch.setattr(parser, "_pdf_page_count", lambda _path: 1)
    monkeypatch.setattr(parser, "_pdf_has_text_layer", lambda _path, _count: True)
    monkeypatch.setattr(parser, "_parse_pdf_pypdf", lambda _path: ["可抽取的手册正文"])

    parsed = parser.parse(handbook)

    assert pdfplumber_calls == []
    assert parsed.page_count == 1
    assert parsed.text == "可抽取的手册正文"
    assert parsed.meta["ocr_used"] is False


def test_scanned_pdf_falls_back_to_ocr_when_text_layer_is_empty(monkeypatch, tmp_path: Path):
    handbook = tmp_path / "scanned_demo.pdf"
    handbook.write_bytes(b"%PDF-1.4\n")
    parser = DocumentParser()
    monkeypatch.setattr(parser, "_parse_pdf_pypdf", lambda _path: ["", ""])
    monkeypatch.setattr(parser, "_ocr_pdf", lambda _path, page_count: ["封面", "研究生学籍管理办法"])

    parsed = parser.parse(handbook)

    assert parsed.page_count == 2
    assert parsed.text == "封面\n研究生学籍管理办法"
    assert parsed.meta["ocr_used"] is True


def test_large_scanned_pdf_skips_memory_heavy_pypdf_before_ocr(monkeypatch, tmp_path: Path):
    handbook = tmp_path / "large_scanned_demo.pdf"
    handbook.write_bytes(b"%PDF-1.4\n" + b"0" * (32 * 1024 * 1024))
    parser = DocumentParser()
    monkeypatch.setattr(parser, "_pdf_page_count", lambda _path: 2)
    monkeypatch.setattr(parser, "_pdf_has_text_layer", lambda _path, _count: False)
    monkeypatch.setattr(
        parser,
        "_parse_pdf_pypdf",
        lambda _path: (_ for _ in ()).throw(AssertionError("scanned PDF must bypass pypdf")),
    )
    monkeypatch.setattr(parser, "_ocr_pdf", lambda _path, page_count: ["扫描件正文", "第二页正文"])

    parsed = parser.parse(handbook)

    assert parsed.text == "扫描件正文\n第二页正文"
    assert parsed.page_count == 2
    assert parsed.meta["ocr_used"] is True

