from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.ttLib import TTFont

from compreffor.__main__ import main


def test_generate_cff(tmp_path):
    source = tmp_path / "source.otf"
    output = tmp_path / "compressed.otf"
    builder = FontBuilder(1000, isTTF=False)
    builder.setupGlyphOrder([".notdef"])
    builder.setupCharacterMap({})
    builder.setupHorizontalMetrics({".notdef": (500, 0)})
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable({"familyName": "Test", "styleName": "Regular"})
    builder.setupOS2()
    builder.setupPost()
    pen = T2CharStringPen(500, None)
    builder.setupCFF("Test-Regular", {}, {".notdef": pen.getCharString()}, {})
    builder.save(source)

    main([str(source), str(output), "--generate-cff"])
    with TTFont(output) as font:
        assert output.with_suffix(".cff").read_bytes() == font.getTableData("CFF ")
