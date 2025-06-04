"""Advanced PatMax-like features for robust pattern matching in defect detection."""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Search strategies for pattern matching."""

    PATMAX = "patmax"  # Robust geometric matching
    PATQUICK = "patquick"  # Fast template matching
    NORMALIZED_CORRELATION = "normalized_correlation"
    FEATURE_BASED = "feature_based"


class AcceptanceLevel(Enum):
    """Acceptance levels for pattern matching quality."""

    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4


@dataclass
class PatMaxResult:
    """Enhanced pattern match result with defect detection context."""

    x: float
    y: float
    angle: float
    score: float
    scale: float
    template_id: str
    search_strategy: SearchStrategy
    acceptance_level: AcceptanceLevel
    is_expected: bool  # For defect detection - is this pattern expected?
    deviation_from_expected: float  # How much it deviates from expected location
    pattern_quality: float  # Quality of the pattern itself


@dataclass
class PatMaxTemplate:
    """Advanced template with defect detection features."""

    id: str
    template: np.ndarray
    mask: Optional[np.ndarray]
    expected_locations: List[Tuple[float, float]]  # Where patterns should normally be
    tolerance_radius: float  # Acceptable deviation from expected location
    min_scale: float
    max_scale: float
    angle_range: Tuple[float, float]  # (min_angle, max_angle)
    acceptance_threshold: float
    search_strategy: SearchStrategy
    defect_indicators: Dict  # What constitutes a defect for this pattern


class AdvancedPatMax:
    """Advanced PatMax implementation focused on defect detection."""

    def __init__(self):
        self.templates = {}
        self.sift = cv2.SIFT_create(nfeatures=1000)
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def train_patmax_template(
        self,
        image: np.ndarray,
        roi: Tuple[int, int, int, int],
        template_id: str,
        expected_locations: List[Tuple[float, float]] = None,
        search_strategy: SearchStrategy = SearchStrategy.PATMAX,
        acceptance_threshold: float = 0.7,
    ) -> bool:
        """Train a PatMax template with defect detection parameters."""
        try:
            x, y, w, h = roi
            template = image[y : y + h, x : x + w].copy()

            if len(template.shape) == 3:
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = template.copy()

            # Extract robust features for PatMax
            keypoints, descriptors = self.sift.detectAndCompute(template_gray, None)

            if descriptors is None or len(keypoints) < 20:
                logger.warning(
                    f"Insufficient features for robust PatMax template {template_id}"
                )
                return False

            # Calculate template statistics for defect detection
            template_stats = {
                "mean_intensity": np.mean(template_gray),
                "std_intensity": np.std(template_gray),
                "edge_density": self._calculate_edge_density(template_gray),
                "texture_energy": self._calculate_texture_energy(template_gray),
            }

            self.templates[template_id] = PatMaxTemplate(
                id=template_id,
                template=template,
                mask=None,
                expected_locations=expected_locations or [],
                tolerance_radius=50.0,  # Default tolerance
                min_scale=0.8,
                max_scale=1.2,
                angle_range=(-10.0, 10.0),  # Default angle tolerance
                acceptance_threshold=acceptance_threshold,
                search_strategy=search_strategy,
                defect_indicators={
                    "missing_pattern_threshold": 0.3,
                    "distorted_pattern_threshold": 0.5,
                    "intensity_deviation_threshold": 30,
                    "template_stats": template_stats,
                },
            )

            logger.info(
                f"PatMax template {template_id} trained with {len(keypoints)} features"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to train PatMax template {template_id}: {str(e)}")
            return False

    def find_patmax_patterns(
        self,
        image: np.ndarray,
        template_ids: Optional[List[str]] = None,
        detect_defects: bool = True,
    ) -> List[PatMaxResult]:
        """Find patterns using PatMax with defect detection."""
        results = []

        if template_ids is None:
            template_ids = list(self.templates.keys())

        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image.copy()

        for template_id in template_ids:
            if template_id not in self.templates:
                continue

            template = self.templates[template_id]

            if template.search_strategy == SearchStrategy.PATMAX:
                matches = self._patmax_search(image_gray, template)
            elif template.search_strategy == SearchStrategy.PATQUICK:
                matches = self._patquick_search(image_gray, template)
            else:
                matches = self._feature_based_search(image_gray, template)

            # Post-process matches for defect detection
            if detect_defects:
                matches = self._analyze_for_defects(matches, template, image_gray)

            results.extend(matches)

        return results

    def _patmax_search(
        self, image: np.ndarray, template: PatMaxTemplate
    ) -> List[PatMaxResult]:
        """Robust geometric pattern matching (PatMax equivalent)."""
        matches = []

        # Extract features from search image
        keypoints, descriptors = self.sift.detectAndCompute(image, None)

        if descriptors is None:
            return matches

        # Get template features
        template_gray = (
            cv2.cvtColor(template.template, cv2.COLOR_BGR2GRAY)
            if len(template.template.shape) == 3
            else template.template
        )
        template_kp, template_desc = self.sift.detectAndCompute(template_gray, None)

        if template_desc is None:
            return matches

        # Match features using FLANN matcher for better performance
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        try:
            matches_raw = flann.knnMatch(template_desc, descriptors, k=2)
        except:
            return matches

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches_raw:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 15:  # Need more matches for robust PatMax
            return matches

        # Find multiple pattern instances using clustering
        src_pts = np.float32([template_kp[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches])

        # Use RANSAC to find multiple homographies
        patterns_found = self._find_multiple_patterns(src_pts, dst_pts, template)

        for pattern in patterns_found:
            matches.append(
                PatMaxResult(
                    x=pattern["center"][0],
                    y=pattern["center"][1],
                    angle=pattern["angle"],
                    score=pattern["score"],
                    scale=pattern["scale"],
                    template_id=template.id,
                    search_strategy=SearchStrategy.PATMAX,
                    acceptance_level=(
                        AcceptanceLevel.HIGH
                        if pattern["score"] > 0.8
                        else AcceptanceLevel.MEDIUM
                    ),
                    is_expected=True,  # Will be updated in defect analysis
                    deviation_from_expected=0.0,
                    pattern_quality=pattern["quality"],
                )
            )

        return matches

    def _patquick_search(
        self, image: np.ndarray, template: PatMaxTemplate
    ) -> List[PatMaxResult]:
        """Fast template matching (PatQuick equivalent)."""
        matches = []

        template_gray = (
            cv2.cvtColor(template.template, cv2.COLOR_BGR2GRAY)
            if len(template.template.shape) == 3
            else template.template
        )

        # Multi-scale and multi-angle template matching
        scales = np.linspace(template.min_scale, template.max_scale, 5)
        angles = np.linspace(template.angle_range[0], template.angle_range[1], 9)

        for scale in scales:
            for angle in angles:
                # Scale and rotate template
                center = (template_gray.shape[1] // 2, template_gray.shape[0] // 2)
                M = cv2.getRotationMatrix2D(center, angle, scale)
                rotated_template = cv2.warpAffine(
                    template_gray, M, (template_gray.shape[1], template_gray.shape[0])
                )

                # Skip if template becomes too small or large
                if (
                    rotated_template.shape[0] > image.shape[0]
                    or rotated_template.shape[1] > image.shape[1]
                ):
                    continue

                # Perform template matching
                result = cv2.matchTemplate(
                    image, rotated_template, cv2.TM_CCOEFF_NORMED
                )

                # Find peaks above threshold
                locations = np.where(result >= template.acceptance_threshold)

                for pt in zip(*locations[::-1]):
                    score = result[pt[1], pt[0]]
                    center_x = pt[0] + rotated_template.shape[1] // 2
                    center_y = pt[1] + rotated_template.shape[0] // 2

                    # Calculate pattern quality
                    roi = image[
                        pt[1] : pt[1] + rotated_template.shape[0],
                        pt[0] : pt[0] + rotated_template.shape[1],
                    ]
                    quality = self._calculate_pattern_quality(roi, template)

                    matches.append(
                        PatMaxResult(
                            x=center_x,
                            y=center_y,
                            angle=angle,
                            score=score,
                            scale=scale,
                            template_id=template.id,
                            search_strategy=SearchStrategy.PATQUICK,
                            acceptance_level=(
                                AcceptanceLevel.MEDIUM
                                if score > 0.7
                                else AcceptanceLevel.LOW
                            ),
                            is_expected=True,
                            deviation_from_expected=0.0,
                            pattern_quality=quality,
                        )
                    )

        # Remove duplicate matches (non-maximum suppression)
        matches = self._non_maximum_suppression(matches)

        return matches

    def _analyze_for_defects(
        self, matches: List[PatMaxResult], template: PatMaxTemplate, image: np.ndarray
    ) -> List[PatMaxResult]:
        """Analyze pattern matches for defects."""
        analyzed_matches = []

        for match in matches:
            # Check if pattern is in expected location
            deviation = float("inf")
            if template.expected_locations:
                deviations = [
                    np.sqrt((match.x - exp_x) ** 2 + (match.y - exp_y) ** 2)
                    for exp_x, exp_y in template.expected_locations
                ]
                deviation = min(deviations)
                match.deviation_from_expected = deviation
                match.is_expected = deviation <= template.tolerance_radius

            # Check for pattern distortion
            if (
                match.pattern_quality
                < template.defect_indicators["distorted_pattern_threshold"]
            ):
                logger.warning(
                    f"Distorted pattern detected: {template.id} at ({match.x}, {match.y})"
                )

            # Check for unexpected patterns (potential defects)
            if not match.is_expected and deviation > template.tolerance_radius * 2:
                logger.warning(
                    f"Unexpected pattern detected: {template.id} at ({match.x}, {match.y})"
                )

            analyzed_matches.append(match)

        # Check for missing expected patterns
        if template.expected_locations:
            for exp_x, exp_y in template.expected_locations:
                found_nearby = any(
                    np.sqrt((match.x - exp_x) ** 2 + (match.y - exp_y) ** 2)
                    <= template.tolerance_radius
                    for match in analyzed_matches
                )
                if not found_nearby:
                    logger.warning(
                        f"Missing expected pattern: {template.id} at ({exp_x}, {exp_y})"
                    )
                    # Add a "missing pattern" result
                    analyzed_matches.append(
                        PatMaxResult(
                            x=exp_x,
                            y=exp_y,
                            angle=0,
                            score=0,
                            scale=1,
                            template_id=template.id,
                            search_strategy=template.search_strategy,
                            acceptance_level=AcceptanceLevel.LOW,
                            is_expected=True,
                            deviation_from_expected=0,
                            pattern_quality=0,
                        )
                    )

        return analyzed_matches

    def _calculate_edge_density(self, image: np.ndarray) -> float:
        """Calculate edge density for template characterization."""
        edges = cv2.Canny(image, 50, 150)
        return np.sum(edges > 0) / (image.shape[0] * image.shape[1])

    def _calculate_texture_energy(self, image: np.ndarray) -> float:
        """Calculate texture energy using Local Binary Patterns."""
        # Simplified LBP calculation
        lbp = np.zeros_like(image)
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                center = image[i, j]
                code = 0
                neighbors = [
                    image[i - 1, j - 1],
                    image[i - 1, j],
                    image[i - 1, j + 1],
                    image[i, j + 1],
                    image[i + 1, j + 1],
                    image[i + 1, j],
                    image[i + 1, j - 1],
                    image[i, j - 1],
                ]
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= 1 << k
                lbp[i, j] = code

        # Calculate histogram and energy
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= hist.sum()
        energy = np.sum(hist**2)
        return energy

    def _calculate_pattern_quality(
        self, roi: np.ndarray, template: PatMaxTemplate
    ) -> float:
        """Calculate quality score for a detected pattern."""
        if roi.size == 0:
            return 0.0

        # Compare intensity statistics
        roi_mean = np.mean(roi)
        roi_std = np.std(roi)
        template_stats = template.defect_indicators["template_stats"]

        intensity_score = 1.0 - min(
            1.0, abs(roi_mean - template_stats["mean_intensity"]) / 255.0
        )
        std_score = 1.0 - min(
            1.0, abs(roi_std - template_stats["std_intensity"]) / 255.0
        )

        # Calculate edge and texture similarity
        edge_density = self._calculate_edge_density(roi)
        edge_score = 1.0 - min(1.0, abs(edge_density - template_stats["edge_density"]))

        texture_energy = self._calculate_texture_energy(roi)
        texture_score = 1.0 - min(
            1.0, abs(texture_energy - template_stats["texture_energy"])
        )

        # Combined quality score
        quality = (
            0.3 * intensity_score
            + 0.2 * std_score
            + 0.3 * edge_score
            + 0.2 * texture_score
        )
        return quality

    def _find_multiple_patterns(
        self, src_pts: np.ndarray, dst_pts: np.ndarray, template: PatMaxTemplate
    ) -> List[Dict]:
        """Find multiple pattern instances using iterative RANSAC."""
        patterns = []
        remaining_src = src_pts.copy()
        remaining_dst = dst_pts.copy()

        max_iterations = 5  # Maximum number of patterns to find

        for _ in range(max_iterations):
            if len(remaining_src) < 15:
                break

            try:
                # Find homography
                H, mask = cv2.findHomography(
                    remaining_src, remaining_dst, cv2.RANSAC, 5.0, confidence=0.99
                )

                if H is None:
                    break

                # Calculate pattern properties
                template_corners = np.float32(
                    [
                        [0, 0],
                        [template.template.shape[1], 0],
                        [template.template.shape[1], template.template.shape[0]],
                        [0, template.template.shape[0]],
                    ]
                ).reshape(-1, 1, 2)

                transformed_corners = cv2.perspectiveTransform(template_corners, H)

                center_x = np.mean(transformed_corners[:, 0, 0])
                center_y = np.mean(transformed_corners[:, 0, 1])

                # Calculate angle and scale
                angle = np.degrees(np.arctan2(H[1, 0], H[0, 0]))
                scale_x = np.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2)
                scale_y = np.sqrt(H[0, 1] ** 2 + H[1, 1] ** 2)
                scale = (scale_x + scale_y) / 2

                # Calculate match score
                score = np.sum(mask) / len(mask) if mask is not None else 0

                if score >= template.acceptance_threshold:
                    patterns.append(
                        {
                            "center": (center_x, center_y),
                            "angle": angle,
                            "scale": scale,
                            "score": score,
                            "quality": min(1.0, score * 1.2),  # Quality estimate
                        }
                    )

                # Remove inlier points for next iteration
                if mask is not None:
                    outliers = mask.flatten() == 0
                    remaining_src = remaining_src[outliers]
                    remaining_dst = remaining_dst[outliers]
                else:
                    break

            except Exception as e:
                logger.debug(f"Pattern finding iteration failed: {str(e)}")
                break

        return patterns

    def _non_maximum_suppression(
        self, matches: List[PatMaxResult], overlap_threshold: float = 50.0
    ) -> List[PatMaxResult]:
        """Remove overlapping matches keeping the best ones."""
        if not matches:
            return matches

        # Sort by score
        matches.sort(key=lambda x: x.score, reverse=True)

        suppressed = []
        for i, match in enumerate(matches):
            should_keep = True
            for kept_match in suppressed:
                distance = np.sqrt(
                    (match.x - kept_match.x) ** 2 + (match.y - kept_match.y) ** 2
                )
                if distance < overlap_threshold:
                    should_keep = False
                    break

            if should_keep:
                suppressed.append(match)

        return suppressed

    def detect_pattern_defects(self, image: np.ndarray, template_id: str) -> Dict:
        """Comprehensive defect detection for a specific pattern."""
        if template_id not in self.templates:
            return {"error": "Template not found"}

        template = self.templates[template_id]
        matches = self.find_patmax_patterns(image, [template_id], detect_defects=True)

        defects = {
            "missing_patterns": [],
            "unexpected_patterns": [],
            "distorted_patterns": [],
            "displaced_patterns": [],
            "total_defects": 0,
        }

        for match in matches:
            if match.score == 0:  # Missing pattern
                defects["missing_patterns"].append(
                    {"expected_location": (match.x, match.y), "severity": "high"}
                )
            elif not match.is_expected:
                defects["unexpected_patterns"].append(
                    {
                        "location": (match.x, match.y),
                        "score": match.score,
                        "deviation": match.deviation_from_expected,
                    }
                )
            elif (
                match.pattern_quality
                < template.defect_indicators["distorted_pattern_threshold"]
            ):
                defects["distorted_patterns"].append(
                    {
                        "location": (match.x, match.y),
                        "quality": match.pattern_quality,
                        "score": match.score,
                    }
                )
            elif match.deviation_from_expected > template.tolerance_radius:
                defects["displaced_patterns"].append(
                    {
                        "location": (match.x, match.y),
                        "expected_location": self._find_nearest_expected_location(
                            match, template
                        ),
                        "deviation": match.deviation_from_expected,
                    }
                )

        defects["total_defects"] = (
            len(defects["missing_patterns"])
            + len(defects["unexpected_patterns"])
            + len(defects["distorted_patterns"])
            + len(defects["displaced_patterns"])
        )

        return defects

    def _find_nearest_expected_location(
        self, match: PatMaxResult, template: PatMaxTemplate
    ) -> Tuple[float, float]:
        """Find the nearest expected location for a displaced pattern."""
        if not template.expected_locations:
            return (0, 0)

        distances = [
            np.sqrt((match.x - exp_x) ** 2 + (match.y - exp_y) ** 2)
            for exp_x, exp_y in template.expected_locations
        ]
        min_idx = np.argmin(distances)
        return template.expected_locations[min_idx]
