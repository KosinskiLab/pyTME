import pytest

napari = pytest.importorskip("napari")

WIDGET_NAMES = (
    "MaskWidget",
    "ImageActionsWidget",
    "AlignmentWidget",
    "ExportWidget",
    "CandidatesWidget",
    "MatchingWidget",
)


def test_widgets_instantiate():
    from tme.scripts import gui

    viewer = napari.Viewer(show=False)
    try:
        for name in WIDGET_NAMES:
            cls = getattr(gui, name)
            cls(viewer)
    finally:
        viewer.close()
