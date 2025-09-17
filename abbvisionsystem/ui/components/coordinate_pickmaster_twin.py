"""ABB PickMaster Twin 2 XML export functionality."""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime
import os


class ABBPickMasterExporter:
    """Export detection results to ABB PickMaster Twin 2 XML format."""

    def __init__(self, calibrator=None):
        self.calibrator = calibrator

    def export_detections_to_xml(
        self,
        detections: Dict,
        image_shape: tuple,
        output_path: str,
        origin_point: tuple = None,
        coordinate_system: str = "camera",
    ) -> bool:
        """
        Export detection results to ABB PickMaster XML format.

        Args:
            detections: Detection results dictionary
            image_shape: (height, width) of the image
            output_path: Path to save the XML file
            origin_point: (x, y) origin point in pixels
            coordinate_system: Type of coordinate system ("camera" or "robot")

        Returns:
            bool: True if export successful
        """
        try:
            # Create root XML element
            root = ET.Element("PickMasterData")
            root.set("version", "2.0")
            root.set("timestamp", datetime.now().isoformat())

            # Add header information
            header = ET.SubElement(root, "Header")
            ET.SubElement(header, "CoordinateSystem").text = coordinate_system
            ET.SubElement(header, "Units").text = (
                "mm"
                if self.calibrator and self.calibrator.calibration_result
                else "pixels"
            )
            ET.SubElement(header, "ImageSize").text = (
                f"{image_shape[1]}x{image_shape[0]}"
            )

            # Add calibration info if available
            if self.calibrator and self.calibrator.calibration_result:
                calib_elem = ET.SubElement(header, "Calibration")
                ET.SubElement(calib_elem, "PixelsPerMM").text = str(
                    self.calibrator.calibration_result.pixels_per_mm
                )
                ET.SubElement(calib_elem, "ReprojectionError").text = str(
                    self.calibrator.calibration_result.reprojection_error
                )
                if origin_point:
                    ET.SubElement(calib_elem, "OriginX").text = str(origin_point[0])
                    ET.SubElement(calib_elem, "OriginY").text = str(origin_point[1])

            # Add detection results
            objects_elem = ET.SubElement(root, "Objects")
            objects_elem.set("count", str(detections.get("num_detections", 0)))

            if detections and detections.get("num_detections", 0) > 0:
                for i in range(detections["num_detections"]):
                    obj_data = self._process_single_detection(
                        detections, i, image_shape, origin_point
                    )
                    if obj_data:
                        self._add_object_to_xml(objects_elem, obj_data, i + 1)

            # Write XML file with pretty formatting
            xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")

            # Remove empty lines and fix formatting
            xml_lines = [line for line in xml_str.split("\n") if line.strip()]
            formatted_xml = "\n".join(xml_lines)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(formatted_xml)

            return True

        except Exception as e:
            print(f"Error exporting to XML: {e}")
            return False

    def _process_single_detection(
        self,
        detections: Dict,
        index: int,
        image_shape: tuple,
        origin_point: tuple = None,
    ) -> Optional[Dict]:
        """Process a single detection and convert coordinates."""
        try:
            # Get bounding box
            if "absolute_boxes" in detections and len(detections["absolute_boxes"]) > 0:
                box = detections["absolute_boxes"][index]
            else:
                h, w = image_shape
                box = detections["boxes"][index] * [w, h, w, h]

            x1, y1, x2, y2 = map(int, box)

            # Calculate object center in pixels
            center_x_px = (x1 + x2) // 2
            center_y_px = (y1 + y2) // 2
            width_px = x2 - x1
            height_px = y2 - y1

            # Get origin point (fallback to image center if not provided)
            if origin_point is None:
                origin_point = (image_shape[1] // 4, image_shape[0] // 4)

            ox, oy = origin_point

            # Calculate relative coordinates from origin
            rel_x_px = center_x_px - ox
            rel_y_px = center_y_px - oy

            # Convert to real-world coordinates if calibrated
            if self.calibrator and self.calibrator.calibration_result:
                pixels_per_mm = self.calibrator.calibration_result.pixels_per_mm
                x_mm = rel_x_px / pixels_per_mm
                y_mm = rel_y_px / pixels_per_mm
                width_mm = width_px / pixels_per_mm
                height_mm = height_px / pixels_per_mm
                unit = "mm"
            else:
                x_mm = rel_x_px
                y_mm = rel_y_px
                width_mm = width_px
                height_mm = height_px
                unit = "pixels"

            # Calculate orientation (theta) from object geometry
            # For top-down 2D picking, we estimate orientation from bounding box aspect ratio
            theta_deg = self._calculate_object_orientation(detections, index, box)

            # Get confidence and class
            confidence = (
                detections.get("scores", [1.0])[index]
                if index < len(detections.get("scores", []))
                else 1.0
            )
            class_id = (
                detections.get("classes", [0])[index]
                if index < len(detections.get("classes", []))
                else 0
            )
            class_name = self._get_class_name(class_id)

            return {
                "x": x_mm,
                "y": y_mm,
                "theta": theta_deg,
                "width": width_mm,
                "height": height_mm,
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name,
                "unit": unit,
                "pixel_coords": (center_x_px, center_y_px),
                "bounding_box": (x1, y1, x2, y2),
            }

        except Exception as e:
            print(f"Error processing detection {index}: {e}")
            return None

    def _calculate_object_orientation(
        self, detections: Dict, index: int, box: tuple
    ) -> float:
        """
        Calculate object orientation in degrees for 2D top-down picking.

        For ABB PickMaster Twin 2:
        - 0° = object aligned with X axis (horizontal)
        - 90° = object aligned with Y axis (vertical)
        - Range: 0-180° (since top-down view, 180° symmetry)
        """
        try:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # Simple orientation estimation based on aspect ratio
            # If width > height, object is more horizontal (closer to 0°)
            # If height > width, object is more vertical (closer to 90°)

            if width > height:
                # Horizontal orientation
                aspect_ratio = height / width
                theta = np.arctan(aspect_ratio) * 180 / np.pi
            else:
                # Vertical orientation
                aspect_ratio = width / height
                theta = 90 - (np.arctan(aspect_ratio) * 180 / np.pi)

            # Normalize to 0-180° range
            theta = theta % 180

            return round(theta, 1)

        except Exception:
            # Default to 0° if calculation fails
            return 0.0

    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        class_names = {0: "Normal", 1: "Defect", 2: "Unknown"}
        return class_names.get(class_id, "Unknown")

    def _add_object_to_xml(self, parent_elem: ET.Element, obj_data: Dict, obj_id: int):
        """Add object data to XML element."""
        obj_elem = ET.SubElement(parent_elem, "Object")
        obj_elem.set("id", str(obj_id))
        obj_elem.set("class", obj_data["class_name"])
        obj_elem.set("confidence", f"{obj_data['confidence']:.3f}")

        # Position element (ABB PickMaster format)
        pos_elem = ET.SubElement(obj_elem, "Position")
        ET.SubElement(pos_elem, "X").text = f"{obj_data['x']:.2f}"
        ET.SubElement(pos_elem, "Y").text = f"{obj_data['y']:.2f}"
        ET.SubElement(pos_elem, "Theta").text = f"{obj_data['theta']:.1f}"
        ET.SubElement(pos_elem, "Units").text = obj_data["unit"]

        # Dimensions element
        dim_elem = ET.SubElement(obj_elem, "Dimensions")
        ET.SubElement(dim_elem, "Width").text = f"{obj_data['width']:.2f}"
        ET.SubElement(dim_elem, "Height").text = f"{obj_data['height']:.2f}"
        ET.SubElement(dim_elem, "Units").text = obj_data["unit"]

        # Additional data for debugging/traceability
        meta_elem = ET.SubElement(obj_elem, "Metadata")
        ET.SubElement(meta_elem, "PixelX").text = str(obj_data["pixel_coords"][0])
        ET.SubElement(meta_elem, "PixelY").text = str(obj_data["pixel_coords"][1])
        x1, y1, x2, y2 = obj_data["bounding_box"]
        ET.SubElement(meta_elem, "BoundingBox").text = f"{x1},{y1},{x2},{y2}"


def export_detections_for_abb(
    detections: Dict,
    calibrator,
    image_shape: tuple,
    origin_point: tuple = None,
    output_dir: str = "abb_exports",
) -> Optional[str]:
    """
    Convenience function to export detections for ABB PickMaster Twin 2.

    Returns:
        str: Path to exported XML file, or None if export failed
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pickmaster_objects_{timestamp}.xml"
        output_path = os.path.join(output_dir, filename)

        # Create exporter and export
        exporter = ABBPickMasterExporter(calibrator)

        if exporter.export_detections_to_xml(
            detections, image_shape, output_path, origin_point
        ):
            return output_path
        else:
            return None

    except Exception as e:
        print(f"Error in ABB export: {e}")
        return None
