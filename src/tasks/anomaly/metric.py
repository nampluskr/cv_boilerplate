# "image_auroc" and "pixel_auroc" are intentionally NOT registered here: src/tasks/toy/metric.py
# already registers METRICS.register("image_auroc")(BinaryAUROC) and
# METRICS.register("pixel_auroc")(BinaryAUROC) in the shared global METRICS namespace (imported
# by every task package, PLAN-P1 SS4/toy fixtures), and BinaryAUROC() with no params already
# matches PLAN-P5 SS7.1's spec exactly. Re-registering the same keys here would raise
# RegistryError -- the same reuse pattern src/tasks/segmentation/metric.py documents for "miou".
