"""Split isolation, export channel order and strict Haar representation checks"""

from pathlib import Path
import numpy as np
import pytest
from tinysr.data.pairs import validate_manifest
from tinysr.conversion.tensorflow import depth_to_space_filter_order
from tinysr.conversion.haar import convert_xml, inspect_binary, UnsupportedCascade


def test_frozen_nested_membership():
    root = Path(__file__).resolve().parents[1]
    small, sc = validate_manifest(root / "data/manifests/ffhq_5000.json")
    large, lc = validate_manifest(root / "data/manifests/ffhq_20000.json")
    assert sc == {"train": 5000, "val": 100, "test": 100}
    assert lc == {"train": 20000, "val": 100, "test": 100}
    a = [r["id"] for r in small["pairs"] if r["split"] == "train"]
    b = [r["id"] for r in large["pairs"] if r["split"] == "train"]
    assert b[:5000] == a
    for split in ("val", "test"):
        assert [r["id"] for r in small["pairs"] if r["split"] == split] == [
            r["id"] for r in large["pairs"] if r["split"] == split
        ]


def test_pixelshuffle_mapping_is_color_and_position_exact():
    order = depth_to_space_filter_order()
    for dy in range(4):
        for dx in range(4):
            for color in range(3):
                assert order[(dy * 4 + dx) * 3 + color] == color * 16 + dy * 4 + dx


def xml(nodes="0 -1 0 0.5", leaves="-1 1"):
    return (
        "<opencv_storage><cascade><featureType>HAAR</featureType><stageType>BOOST</stageType><width>24</width><height>24</height><stageNum>1</stageNum><stages><_><maxWeakCount>1</maxWeakCount><stageThreshold>0</stageThreshold><weakClassifiers><_><internalNodes>"
        + nodes
        + "</internalNodes><leafValues>"
        + leaves
        + "</leafValues></_></weakClassifiers></_></stages><features><_><rects><_>0 0 12 12 -1</_><_>12 0 12 12 1</_></rects></_></features></cascade></opencv_storage>"
    ).encode()


def test_haar_readback_and_unsupported_tree():
    binary, meta = convert_xml(xml())
    assert inspect_binary(binary)["sha256"] == meta["sha256"]
    with pytest.raises(UnsupportedCascade, match="stump"):
        convert_xml(xml("0 1 0 0.5 0 -1 0 0.2", "-1 0 1"))
    with pytest.raises(ValueError):
        inspect_binary(binary[:-1])
