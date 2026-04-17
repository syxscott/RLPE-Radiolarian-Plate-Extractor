from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_tei(xml_text: str) -> ET.Element | None:
    if not xml_text.strip():
        return None
    try:
        return ET.fromstring(xml_text)
    except ET.ParseError:
        return None


def tei_namespace(root: ET.Element | None) -> dict[str, str]:
    if root is None or not root.tag.startswith("{"):
        return {}
    return {"tei": root.tag.split("}")[0].strip("{")}


def normalize_xml_id(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    value = value.replace("#", "").strip()
    value = re.sub(r"[^A-Za-z0-9_\-]+", "_", value)
    return value or fallback


def get_figure_elements(root: ET.Element | None) -> list[ET.Element]:
    if root is None:
        return []
    ns = tei_namespace(root)
    if ns:
        return root.findall(".//tei:figure", ns)
    return root.findall(".//figure")


def get_figure_caption(fig: ET.Element, root: ET.Element | None = None) -> str:
    ns = tei_namespace(root)
    parts: list[str] = []
    for tag in ("figDesc", "head", "note"):
        node = fig.find(f"tei:{tag}", ns) if ns else fig.find(tag)
        if node is not None:
            text = " ".join(t.strip() for t in node.itertext() if t and t.strip())
            if text:
                parts.append(text)
    return " ".join(parts).strip()
