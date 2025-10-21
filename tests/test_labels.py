import os, json

SPLIT = "sokoban_core/levels/splits/test.txt"
OUT = "tmp/labels_test.jsonl"


def test_generate_labels_smoke():
    if not os.path.exists(SPLIT):
        import pytest; pytest.skip("no split file yet")

    from scripts.generate_labels import main as gen_main
    import sys
    argv_bak = list(sys.argv)
    sys.argv = ["gen", "--list", SPLIT, "--out", OUT, "--use_dl"]
    try:
        gen_main()
    finally:
        sys.argv = argv_bak
    assert os.path.exists(OUT)

    with open(OUT, "r", encoding="utf-8") as f:
        line = f.readline()
        assert line.strip() != ""
        rec = json.loads(line)
        assert "y" in rec and "boxes" in rec
