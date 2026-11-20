from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

from honeybee.aperture import Aperture
from honeybee.face import Face
from honeybee.model import Model
from honeybee.room import Room
from ladybug_geometry.geometry3d import Face3D, Point3D, Vector3D
from shapely.geometry import Polygon

from honeybee_gbas.dwg import build
from honeybee_gbas.dwg.records import new_drawing
from honeybee_gbas.dwg.surfaces import add_parts, wcs_points
from honeybee_gbas.dwg.validation import inspect_native, require_export, validate


def vector_area(points, normal):
    origin = Point3D(*points[0])
    direction = Vector3D(*normal)
    return sum((Point3D(*a) - origin).cross(Point3D(*b) - origin).dot(direction) for a, b in zip(points, (*points[1:], points[0]))) / 2


def surface(points, normal, holes=(), kind="RoofCeiling"):
    drawing, codes = new_drawing()
    report = {"surfaces": []}
    row = add_parts(drawing, codes, report, "Room", "Panel", kind, "Roof", [points], normal, tuple, holes=holes, bind=False)
    return drawing, report, row


class SurfaceOrientationTest(unittest.TestCase):
    def assert_orientation(self, drawing, report):
        readback = inspect_native(drawing.bytes(), include_geometry=True)
        self.assertEqual(readback["surface_orientation"]["checked_surfaces"] + readback["native_roof_meshes"], len(report["surfaces"]))
        for row in report["surfaces"]:
            native = readback["geometry"]["entities"][row["handle"]]
            if native["type"] == "POLYLINE_PFACE":
                areas = [vector_area(ring, row["normal"]) for ring in native["polygons_mm"]]
                self.assertTrue(all(a > 0 for a in areas))
                self.assertAlmostEqual(sum(areas), vector_area(row["boundary_mm"], row["normal"]), places=5)
                continue
            self.assertEqual(native["outline_mm"], row["outline_mm"])
            self.assertEqual(native["holes_mm"], row["holes_mm"])
            self.assertEqual(native["normal"], tuple(row["normal"]))
            self.assertTrue(Polygon(native["outline_mm"]).exterior.is_ccw)
            self.assertGreater(vector_area(row["boundary_mm"], row["normal"]), 0)
            for actual, expected in zip(wcs_points(native), row["boundary_mm"]):
                for a, b in zip(actual, expected):
                    self.assertAlmostEqual(a, b, places=6)
            for hole in native["holes_mm"]:
                self.assertFalse(Polygon(hole).exterior.is_ccw)
        return readback

    def test_punched_flat_roof_and_skylight_keep_area_and_right_handed_rings(self):
        room = Room.from_box("RoofRoom", 5, 4, 3)
        room.story = "Top"
        roof = next(face for face in room.faces if str(face.type) == "RoofCeiling")
        roof.add_aperture(Aperture("RoofWindow", Face3D(tuple(Point3D(x, y, 3) for x, y in ((1, 1), (2, 1), (2, 2), (1, 2))))))
        model = Model("PunchedRoof", [room])
        original = copy.deepcopy(model.to_dict())
        drawing, report = build(model)
        validate(model, drawing.bytes(), report)
        readback = self.assert_orientation(drawing, report)
        roofs = [row for row in report["surfaces"] if row["kind"] == "RoofCeiling"]
        skylights = [row for row in report["surfaces"] if row["kind"] == "Skylight"]
        self.assertEqual(roofs + skylights, [])
        definition, = report["defined_skylights"]
        poly = readback["geometry"]["polylines"][definition["handle"]]
        self.assertTrue(poly["closed"] and Polygon(poly["points"]).exterior.is_ccw)
        self.assertAlmostEqual(Polygon(poly["points"]).area / 1e6, 1)
        self.assertEqual(definition["source_room"], room.identifier)
        source_roof = next(row for row in report["roof_selection"]["faces"] if row["identifier"] == roof.identifier)
        self.assertEqual(source_roof["source_gross_area_m2"], 20)
        self.assertEqual(source_roof["source_opaque_area_m2"], 19)
        self.assertEqual(source_roof["source_skylight_area_m2"], 1)
        self.assertEqual(report["source_geometry_validation"]["summary"]["review_items"], 0)
        self.assertEqual(model.to_dict(), original)
        self.assertTrue(all(row["kind"] != "Floor" for row in report["surfaces"]))

    def test_native_window_wall_retains_source_plane_and_right_handed_surfaces(self):
        box = Room.from_box("InclinedRoom", 5, 4, 3)
        def shear(point):
            return Point3D(point.x + point.z * 0.3, point.y, point.z)
        room = Room("InclinedRoom", [Face(face.identifier, Face3D(tuple(shear(point) for point in face.geometry.boundary)), face.type) for face in box.faces])
        room.story = "Top"
        face = next(face for face in room.faces if face.normal.x > 0.5)
        window = Aperture("InclinedWindow", Face3D(tuple(Point3D(5 + z * 0.3, y, z) for y, z in ((0.5, 0.5), (1.5, 0.5), (1.5, 2), (0.5, 2)))))
        face.add_aperture(window)
        model = Model("PunchedWall", [room])
        original = copy.deepcopy(model.to_dict())
        drawing, report = build(model)
        validate(model, drawing.bytes(), report)
        self.assert_orientation(drawing, report)
        walls = [row for row in report["walls"] if row["identifier"] == face.identifier]
        windows = [row for row in report["openings"] if window.identifier in row["source_identifiers"]]
        self.assertGreaterEqual(len(walls), 1)
        self.assertEqual(len(windows), 1)
        self.assertTrue(all(row.get("source_oblique") for row in walls))
        self.assertEqual(windows[0]["wall"], walls[0]["handle"])
        self.assertEqual(report.get("projected_walls", []), [])
        self.assertEqual(report["counts"]["native_panel_openings"], 0)
        self.assertEqual(report["source_geometry_validation"]["summary"]["passed_openings"], 1)
        self.assertEqual(report["source_geometry_validation"]["summary"]["review_items"], 0)
        require_export(report)
        self.assertEqual(model.to_dict(), original)

    def test_outer_and_inner_rings_follow_flat_tilted_vertical_and_reverse_normals(self):
        cases = (
            ((0, 0, 1), lambda x, y: Point3D(x, y, 3000)),
            ((0, 0, -1), lambda x, y: Point3D(x, y, 3000)),
            ((-1, 0, 2), lambda x, y: Point3D(x, y, 3000 + x / 2)),
            ((1, 0, -2), lambda x, y: Point3D(x, y, 3000 + x / 2)),
            ((0, -1, 0), lambda x, y: Point3D(x, 4000, y)),
            ((0, 1, 0), lambda x, y: Point3D(x, 4000, y)),
        )
        for normal, point in cases:
            for reverse_outer, reverse_hole in ((False, False), (False, True), (True, False), (True, True)):
                with self.subTest(normal=normal, reverse_outer=reverse_outer, reverse_hole=reverse_hole):
                    outer = tuple(point(x, y) for x, y in ((0, 0), (5000, 0), (5000, 4000), (0, 4000)))
                    hole = tuple(point(x, y) for x, y in ((1000, 1000), (2000, 1000), (2000, 2000), (1000, 2000)))
                    outer = outer[::-1] if reverse_outer else outer
                    hole = hole[::-1] if reverse_hole else hole
                    original = copy.deepcopy((outer, hole))
                    drawing, report, row = surface(outer, normal, (hole,))
                    readback = self.assert_orientation(drawing, report)
                    self.assertEqual(readback["surface_orientation"]["checked_inner_rings"], 1)
                    self.assertEqual(len(row["boundary_mm"]), len(outer))
                    self.assertTrue(all(min(Point3D(*actual).distance_to_point(expected) for expected in outer) < 1e-8 for actual in row["boundary_mm"]))
                    self.assertAlmostEqual(Vector3D(*row["normal"]).dot(Vector3D(*normal).normalize()), 1)
                    actual = Polygon(row["outline_mm"], row["holes_mm"]).area
                    self.assertAlmostEqual(actual, Face3D(outer, holes=(hole,)).area)
                    self.assertEqual((outer, hole), original)

    def test_equivalent_reverse_normals_preserve_the_world_plane(self):
        points = tuple(Point3D(x, y, 3000 + x / 2) for x, y in ((0, 0), (5000, 0), (5000, 4000), (0, 4000)))
        first, report, forward = surface(points, (-1, 0, 2))
        second, reverse_report, reverse = surface(points, (1, 0, -2))
        self.assert_orientation(first, report)
        self.assert_orientation(second, reverse_report)
        self.assertEqual(set(forward["boundary_mm"]), set(reverse["boundary_mm"]))
        self.assertAlmostEqual(forward["elevation_mm"], -reverse["elevation_mm"])
        self.assertAlmostEqual(Polygon(forward["outline_mm"]).area, Polygon(reverse["outline_mm"]).area)

    def test_small_rings_at_large_ocs_offsets_pass_independent_readback(self):
        points = tuple(Point3D(1e9 + x, -1e9 + y, 3000) for x, y in ((0, 0), (0, 4), (4, 4), (4, 0)))
        hole = tuple(Point3D(1e9 + x, -1e9 + y, 3000) for x, y in ((1, 1), (2, 1), (2, 2), (1, 2)))
        drawing, report, row = surface(points, (0, 0, 1), (hole,))
        self.assert_orientation(drawing, report)
        self.assertAlmostEqual(Polygon(row["outline_mm"], row["holes_mm"]).area, 15)

    def test_binary_gate_checks_reversed_outer_rings_without_a_manifest(self):
        points = tuple(Point3D(x, y, 3000) for x, y in ((0, 0), (5000, 0), (5000, 4000), (0, 4000)))
        drawing, report, row = surface(points, (0, 0, 1))
        row["outline_mm"].reverse()
        with self.assertRaisesRegex(ValueError, "right-handed outer ring"):
            inspect_native(drawing.bytes())

    def test_binary_gate_checks_inner_ring_direction_without_a_manifest(self):
        points = tuple(Point3D(x, y, 3000) for x, y in ((0, 0), (5000, 0), (5000, 4000), (0, 4000)))
        hole = tuple(Point3D(x, y, 3000) for x, y in ((1000, 1000), (2000, 1000), (2000, 2000), (1000, 2000)))
        drawing, report, row = surface(points, (0, 0, 1), (hole,))
        row["holes_mm"][0].reverse()
        with self.assertRaisesRegex(ValueError, "oppositely oriented inner rings"):
            inspect_native(drawing.bytes())

    def test_binary_gate_requires_positive_outer_area_and_negative_hole_area(self):
        points = tuple(Point3D(x, y, 3000) for x, y in ((0, 0), (5000, 0), (5000, 4000), (0, 4000)))
        for field, message in (("outline_mm", "right-handed outer ring"), ("holes_mm", "oppositely oriented inner rings")):
            with self.subTest(field=field):
                drawing, report, row = surface(points, (0, 0, 1))
                line = [(0, 0), (1, 0), (2, 0)]
                row[field] = line if field == "outline_mm" else [line]
                with self.assertRaisesRegex(ValueError, message):
                    inspect_native(drawing.bytes())

    def test_plane_projection_keeps_boundary_and_outline_in_the_same_order(self):
        points = [Point3D(0, 0, 0), Point3D(0, 1000, 0), Point3D(1000, 1000, 0.2), Point3D(1000, 0, 0)]
        drawing, codes = new_drawing()
        report = {"surfaces": []}
        row = add_parts(drawing, codes, report, "Room", "Projected", "RoofCeiling", "Roof", [points], (0, 0, 1), tuple, bind=False, planar_tolerance_mm=1)
        self.assert_orientation(drawing, report)
        self.assertAlmostEqual(row["maximum_projection_mm"], 0.15)
        self.assertTrue(all(abs(point[2] - 0.05) < 1e-8 for point in row["boundary_mm"]))

    def test_floor_policy_retains_the_existing_upward_normal(self):
        points = tuple(Point3D(x, y, 3000) for x, y in ((0, 0), (0, 4000), (5000, 4000), (5000, 0)))
        drawing, report, row = surface(points, (0, 0, -1), kind="Floor")
        self.assert_orientation(drawing, report)
        self.assertEqual(row["normal"], (0, 0, 1))
        self.assertEqual(row["face_usage"], 8)


if __name__ == "__main__":
    unittest.main()
