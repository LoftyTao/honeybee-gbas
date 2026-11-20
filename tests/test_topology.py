from __future__ import annotations

import unittest
from dataclasses import replace
from math import hypot
from pathlib import Path

from honeybee.model import Model
from honeybee.room import Room
from ladybug_geometry.geometry3d import Point3D, Vector3D

from honeybee_gbas import layout
from honeybee_gbas import model as hb
from honeybee_gbas.layout import WallSegment


class WallTopologyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.room = Room.from_box("room", 5, 4, 3)
        self.model = Model("model", [self.room])
        self.source = hb.wall_segments(self.model)

    def test_exterior_corners_meet(self) -> None:
        walls = layout.wall_axes(self.source, 1000.0)
        points = [point for wall in walls for point in (wall.start, wall.end)]
        for point in points:
            self.assertEqual(sum(point.distance_to_point(other) < 1e-6 for other in points), 2)

    def test_surface_preserves_source(self) -> None:
        self.assertEqual(layout.wall_axes(self.source, 1000.0, "surface"), self.source)

    def test_interior_wall_meets_shifted_perimeter(self) -> None:
        left = Room.from_box("left", 5, 4, 3)
        right = Room.from_box("right", 5, 4, 3, origin=Point3D(5, 0, 0))
        Room.solve_adjacency([left, right], 0.01)
        model = Model("two_rooms", [left, right])
        walls = layout.wall_axes(hb.wall_segments(model), 1000.0)
        interior = [wall for wall in walls if not wall.exterior]
        self.assertEqual(len(interior), 1)
        for point in (interior[0].start, interior[0].end):
            self.assertTrue(any(
                wall.exterior and layout.point_on_segment(point, wall.start, wall.end, 1e-6)
                for wall in walls
            ))

    def test_tee_extends_perimeter_to_shared_axis_for_both_directions(self) -> None:
        source = [
            WallSegment(Point3D(0, -4, 0), Point3D(0, 0, 0), 0.4, 3, True, "left", "left", Vector3D(1, 0, 0)),
            WallSegment(Point3D(0, 0, 0), Point3D(0, 8, 0), 0.2, 3, False, "right", "shared", Vector3D(-1, 0, 0)),
            WallSegment(Point3D(0, 0, 0), Point3D(6, 0, 0), 0.4, 3, True, "right", "front", Vector3D(0, -1, 0)),
        ]
        for segments in (source, [replace(wall, start=wall.end, end=wall.start) for wall in reversed(source)]):
            with self.subTest(segments=segments):
                walls = layout.wall_axes(segments, 1000)
                self.assertEqual(len(walls), 3)
                self.assertTrue(all(not isinstance(wall, layout.AxisConnector) for wall in walls))
                front = next(wall for wall in walls if wall.face_identifier == "front")
                left = next(wall for wall in walls if wall.face_identifier == "left")
                shared = next(wall for wall in walls if wall.face_identifier == "shared")
                self.assertTrue(any(layout.point_on_segment(point, front.start, front.end, 1e-7) for point in (left.start, left.end)))
                self.assertTrue(any(layout.point_on_segment(point, front.start, front.end, 1e-7) for point in (shared.start, shared.end)))
                self.assertAlmostEqual(front.thickness_m, 0.4)
                self.assertAlmostEqual(shared.thickness_m, 0.2)

    def test_parallel_lines(self) -> None:
        self.assertIsNone(layout.line_intersection(
            Point3D(0, 0, 0), Point3D(5, 0, 0), Point3D(0, 1, 0), Point3D(5, 1, 0)
        ))

    def test_invalid_axis(self) -> None:
        with self.assertRaises(ValueError):
            layout.wall_axes(self.source, 1000.0, "other")

    def test_feet_and_meters_match(self) -> None:
        meters = layout.wall_axes(self.source, 1000.0)
        self.model.convert_to_units("Feet")
        feet = layout.wall_axes(hb.wall_segments(self.model), 304.8)
        for metric, imperial in zip(meters, feet):
            self.assertAlmostEqual(metric.start.x * 1000.0, imperial.start.x * 304.8)
            self.assertAlmostEqual(metric.start.y * 1000.0, imperial.start.y * 304.8)

    def test_sample_floor_section_is_connected(self) -> None:
        root = Path(__file__).resolve().parents[1]
        model = Model.from_file(str(root / "samples/basic_sample_model/basic_sample_model.hbjson"))
        for story in hb.story_names(model):
            walls = layout.floor_section(model, story)
            for index, wall in enumerate(walls):
                for point in (wall.start, wall.end):
                    self.assertTrue(any(
                        other_index != index and layout.point_on_segment(point, other.start, other.end, 1e-5)
                        for other_index, other in enumerate(walls)
                    ), (story, wall.face_identifier, point))

    def test_sample_centerline_sections_have_disjoint_collinear_ranges(self) -> None:
        root = Path(__file__).resolve().parents[1]
        model = Model.from_file(str(root / "samples/basic_sample_model/basic_sample_model.hbjson"))
        tolerance = 0.1 / hb.unit_scale(model.units)
        for story in hb.story_names(model):
            walls = [wall for wall in layout.floor_section(model, story) if not isinstance(wall, layout.AxisConnector)]
            for index, wall in enumerate(walls):
                dx, dy = wall.end.x - wall.start.x, wall.end.y - wall.start.y
                length = hypot(dx, dy)
                for other in walls[index + 1:]:
                    if not layout._collinear(wall, other, tolerance):
                        continue
                    low, high = sorted(
                        ((point.x - wall.start.x) * dx + (point.y - wall.start.y) * dy) / length
                        for point in (other.start, other.end)
                    )
                    overlap = min(length, high) - max(0.0, low)
                    self.assertLessEqual(overlap, tolerance, (story, wall.face_identifier, other.face_identifier))


if __name__ == "__main__":
    unittest.main()
