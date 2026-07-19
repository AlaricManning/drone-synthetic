"""Conversion: paired renders -> annotated dataset.

Pairs normal/mask frames, thresholds masks into bounding boxes, writes
canonical per-frame JSON annotations, exports the YOLO layout, and emits
a QC report with debug renders.
"""
