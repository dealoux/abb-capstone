"""Visualization utilities for displaying detection results."""

import streamlit as st


def draw_detection_summary(model, detections, confidence_threshold=0.5):
    """Draw detection summary in Streamlit.

    Args:
        model: Model instance with categories attribute
        detections: Detection results from model.predict()
        confidence_threshold: Minimum confidence to include in summary
    """
    if detections is None or detections["num_detections"] == 0:
        st.info("No objects detected in the image.")
        return

    # Filter by confidence threshold
    valid_detections = [
        i
        for i in range(detections["num_detections"])
        if detections["scores"][i] >= confidence_threshold
    ]

    if not valid_detections:
        st.info(f"No objects detected with confidence ≥ {confidence_threshold:.0%}")
        return

    # Create data for table
    data = []
    for i in valid_detections:
        class_id = detections["classes"][i]
        class_name = model.categories.get(class_id, {}).get("name", f"Class {class_id}")
        score = detections["scores"][i]
        data.append({"Object Type": class_name, "Confidence": f"{score:.2%}"})

    # Display as table
    st.table(data)

    # Count by type
    counts = {}
    for i in valid_detections:
        class_id = detections["classes"][i]
        class_name = model.categories.get(class_id, {}).get("name", f"Class {class_id}")
        counts[class_name] = counts.get(class_name, 0) + 1

    # Display summary
    st.subheader("Detection Summary")
    st.write(f"Total objects detected: {len(valid_detections)}")
    for obj_type, count in counts.items():
        st.write(f"- {obj_type}: {count}")
