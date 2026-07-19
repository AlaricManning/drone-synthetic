from dronesynth.datagen.annotations import AnnotatedBox, FrameAnnotation
from dronesynth.datagen.qc import compute_qc


def annotation(index, boxes=(), width=640, height=480):
    return FrameAnnotation(
        frame_index=index, normal=f"seq.{index:04d}.png",
        width=width, height=height, boxes=tuple(boxes),
    )


def box(x=100, y=100, w=40, h=30, fill_ratio=0.45):
    return AnnotatedBox(
        class_id=0, x=x, y=y, w=w, h=h,
        mask_area=int(w * h * fill_ratio), fill_ratio=fill_ratio,
    )


def test_clean_run_has_no_flags():
    report = compute_qc("run_0001", [annotation(0, [box()]), annotation(1)])
    assert report.frames == 2
    assert report.empty_frames == 1
    assert report.total_boxes == 1
    assert report.box_area_min == 1200
    assert report.flags == ()


def test_empty_run_stats_are_none():
    report = compute_qc("run_0001", [annotation(0)])
    assert report.total_boxes == 0
    assert report.box_area_min is None
    assert report.fill_ratio_min is None


def test_low_fill_ratio_flagged():
    report = compute_qc("run_0001", [annotation(0, [box(fill_ratio=0.05)])])
    assert any("low fill ratio" in f.reason for f in report.flags)


def test_tiny_box_flagged():
    report = compute_qc("run_0001", [annotation(0, [box(w=7, h=7)])])
    assert any("tiny box" in f.reason for f in report.flags)


def test_edge_touching_box_flagged():
    report = compute_qc("run_0001", [annotation(0, [box(x=0)])])
    assert any("touches frame edge" in f.reason for f in report.flags)
    report = compute_qc("run_0001", [annotation(0, [box(x=600, w=40)])])  # 600+40 == width
    assert any("touches frame edge" in f.reason for f in report.flags)


def test_multiple_boxes_flagged():
    report = compute_qc("run_0001", [annotation(0, [box(), box(x=300)])])
    assert any("2 boxes" in f.reason for f in report.flags)
