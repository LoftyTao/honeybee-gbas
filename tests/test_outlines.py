from __future__ import annotations

import copy
import unittest
from pathlib import Path

from honeybee.model import Model
from honeybee.room import Room
from ladybug_geometry.geometry3d import Face3D, Point3D, Polyface3D
from shapely import set_precision
from shapely.affinity import affine_transform, translate
from shapely.geometry import Polygon
from shapely.ops import unary_union

from honeybee_gbas import model as hb
from honeybee_gbas.dwg import build, building_area_regions
from honeybee_gbas.dwg.validation import compare_source, inspect_export, inspect_native, validate
from honeybee_gbas.layout import plan_native_levels


ROOT = Path(__file__).resolve().parents[1]


class BuildingOutlinePartitionTest(unittest.TestCase):
    def assert_exact(self, source, parts):
        self.assertTrue(parts)
        self.assertTrue(all(part.is_valid and part.area > 0 and not part.interiors for part in parts))
        self.assertTrue(all(part.convex_hull.difference(part).area <= max(1e-6, part.area * 1e-12) for part in parts))
        union = unary_union(parts)
        tolerance = max(1.0, source.area * 1e-10)
        self.assertLessEqual(source.symmetric_difference(union).area, tolerance)
        self.assertLessEqual(abs(sum(part.area for part in parts) - source.area), tolerance)
        self.assertLessEqual(sum(part.area for part in parts) - union.area, tolerance)

    def test_aligned_courtyards_use_compact_convex_regions_and_preserve_the_holes(self):
        source = Polygon(((0, 0), (10000, 0), (10000, 8000), (0, 8000)),
                         [((1000, 1000), (3000, 1000), (3000, 3000), (1000, 3000)),
                          ((5000, 1000), (8000, 1000), (8000, 6000), (5000, 6000))])
        parts = building_area_regions(source)
        self.assert_exact(source, parts)
        self.assertLessEqual(len(parts), 6)

    def test_fixed_precision_oblique_courtyards_preserve_area_at_large_offsets(self):
        source = Polygon(((0, 0), (10000, 0), (10000, 8000), (6000, 8000),
                          (6000, 5000), (4000, 5000), (4000, 8000), (0, 8000)),
                         [((1000, 1000), (3000, 1000), (2500, 3500), (1200, 3000)),
                          ((6500, 1000), (8500, 1200), (8000, 4000), (7000, 3500))])
        for offset in (0, 1e9):
            with self.subTest(offset=offset):
                shape = set_precision(translate(affine_transform(source, (0.83, -0.37, 0.42, 0.91, 0, 0)), offset, -offset), 0.01)
                parts = building_area_regions(shape)
                self.assert_exact(shape, parts)
                self.assertLessEqual(len(parts), 10)

    def test_sydney_floor_regions_keep_courtyards_with_compact_boundaries(self):
        model = Model.from_file(str(ROOT / "samples/Sydney_Opera_House/Sydney_Opera_House.hbjson"))
        scale = hb.unit_scale(model.units)
        count = 0
        vertices = 0
        for floor in plan_native_levels(model, include_orphan_shades=False):
            shapes = [Polygon([(point.x * scale, point.y * scale) for point in face.boundary],
                              [[(point.x * scale, point.y * scale) for point in ring] for ring in face.holes or ()])
                      for room in model.rooms if room.identifier in floor.room_identifiers
                      for face in hb.room_plan_boundaries(room)]
            union = unary_union([set_precision(shape, 0.01) for shape in shapes])
            components = list(union.geoms) if union.geom_type == "MultiPolygon" else [union]
            for component in components:
                parts = building_area_regions(component)
                self.assert_exact(component, parts)
                count += len(parts)
                vertices += len(component.exterior.coords) + sum(len(ring.coords) for ring in component.interiors)
        self.assertLess(count, vertices)


class BuildingOutlineReadbackTest(unittest.TestCase):
    def fixture(self):
        boundary = tuple(Point3D(x, y, 0) for x, y in ((0, 0), (10, 0), (10, 8), (0, 8)))
        hole = tuple(Point3D(x, y, 0) for x, y in ((3, 3), (7, 3), (7, 5), (3, 5)))
        room = Room.from_polyface3d("Courtyard", Polyface3D.from_offset_face(Face3D(boundary, holes=(hole,)), 3))
        model = Model("Courtyard", [room])
        drawing, report = build(model)
        validate(model, drawing.bytes(), report)
        readback = inspect_native(drawing.bytes(), report, include_geometry=True)
        return model, report, readback

    def test_decoded_regions_cover_source_area_and_preserve_the_courtyard(self):
        _, report, _ = self.fixture()
        result = report["source_geometry_validation"]["building_outlines"]
        self.assertTrue(result["passed"])
        self.assertTrue(inspect_export(report)["checks"]["native_building_outline_regions"])
        row = result["records"][0]
        self.assertEqual(row["native_regions"], 4)
        self.assertAlmostEqual(row["native_union_area_m2"], 72)
        self.assertAlmostEqual(row["symmetric_difference_m2"], 0)
        self.assertGreater(row["internal_seam_length_mm"], 0)

    def test_equal_area_displacement_fails_the_building_region_gate(self):
        model, report, readback = self.fixture()
        for outline in report["building_outlines"]:
            native = readback["geometry"]["entities"][outline["handle"]]
            native["outline_mm"] = [(x + 500, y) for x, y in native["outline_mm"]]
            x, y, z = native["position_mm"]
            native["position_mm"] = (x + 500, y, z)
        report["source_geometry_validation"] = compare_source(model, report, readback)
        result = report["source_geometry_validation"]["building_outlines"]
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["records"][0]["native_union_area_m2"], 72)
        self.assertFalse(inspect_export(report)["checks"]["native_building_outline_regions"])

    def test_duplicate_region_fails_even_when_union_coverage_matches(self):
        model, report, readback = self.fixture()
        original = report["building_outlines"][0]
        report["building_outlines"].append({**original, "handle": 999999})
        readback["geometry"]["entities"][999999] = copy.deepcopy(readback["geometry"]["entities"][original["handle"]])
        result = compare_source(model, report, readback)["building_outlines"]
        self.assertFalse(result["passed"])
        self.assertTrue(result["records"][0]["coverage_passed"])
        self.assertFalse(result["records"][0]["uniqueness_passed"])
        self.assertGreater(result["records"][0]["overlap_area_m2"], 0)

    def test_envelope_regions_use_the_preceding_occupied_storey_union(self):
        lower = Room.from_box("Tall", 5, 4, 6)
        upper = Room.from_box("Upper", 5, 4, 3, origin=Point3D(10, 0, 3))
        lower.story, upper.story = "First", "Second"
        model = Model("Envelope", [lower, upper])
        drawing, report = build(model)
        validate(model, drawing.bytes(), report)
        result = report["source_geometry_validation"]["building_outlines"]
        self.assertTrue(result["passed"])
        helper = next(row for row in result["records"] if row["story"] == "Envelope:First")
        self.assertEqual(helper["source_stories"], ["First"])
        self.assertAlmostEqual(helper["native_union_area_m2"], 20)


class RoomOutlineDisplayTest(unittest.TestCase):
    def fixture(self):
        points = ((0, 0), (10, 0), (10, 10), (7, 10), (7, 3), (3, 3), (3, 10), (0, 10))
        floor = Face3D(tuple(Point3D(x, y, 0) for x, y in points))
        room = Room.from_polyface3d("ConcaveRoom", Polyface3D.from_offset_face(floor, 3))
        model = Model("ConcaveRoom", [room])
        drawing, report = build(model)
        validate(model, drawing.bytes(), report)
        return model, drawing, report

    def test_source_room_contour_is_visible_and_computational_seams_have_a_separate_layer(self):
        _, drawing, report = self.fixture()
        readback = inspect_native(drawing.bytes(), report, include_geometry=True)
        self.assertTrue(readback["plan_display"]["passed"])
        self.assertTrue(inspect_export(report)["checks"]["native_room_outline_display"])
        self.assertEqual(readback["plan_display"]["visible_source_room_outlines"], 1)
        entities = readback["geometry"]["entities"]
        room = entities[report["rooms"][0]["handle"]]
        self.assertEqual(room["layer"], "建-房间")
        self.assertTrue(room["layer_visible"])
        self.assertEqual(room["flags"] & 0x400, 0x400)
        self.assertEqual(room["outline_mm"], report["rooms"][0]["outline_mm"])
        self.assertGreater(len(report["building_outlines"]), 1)
        for row in report["building_outlines"]:
            native = entities[row["handle"]]
            self.assertEqual(native["layer"], "建-建筑区域")
            self.assertFalse(native["layer_visible"])
            self.assertEqual(native["category"], 1)

    def test_visible_computational_partition_receives_a_display_review(self):
        model, drawing, report = self.fixture()
        helpers = {row["handle"] for row in report["building_outlines"]}
        for entity in drawing.entities:
            if entity.handle in helpers:
                entity.layer = "建-房间"
        validate(model, drawing.bytes(), report)
        self.assertTrue(report["source_geometry_validation"]["building_outlines"]["passed"])
        self.assertFalse(inspect_export(report)["checks"]["native_room_outline_display"])

    def test_enabled_computational_layer_receives_a_display_review(self):
        _, drawing, report = self.fixture()
        handle, color, _ = drawing.layers["建-建筑区域"]
        drawing.layers["建-建筑区域"] = (handle, color, True)
        readback = inspect_native(drawing.bytes(), report)
        self.assertFalse(readback["plan_display"]["passed"])
        self.assertEqual(len(readback["plan_display"]["reviews"]), len(report["building_outlines"]))

    def test_source_contour_display_flag_has_an_independent_gate(self):
        model, drawing, report = self.fixture()
        report["rooms"][0]["show_outline"] = False
        validate(model, drawing.bytes(), report)
        self.assertFalse(inspect_export(report)["checks"]["native_room_outline_display"])

    def test_small_area_deep_room_notch_receives_a_boundary_review(self):
        room = Room.from_box("LargeRoom", 100, 100, 3)
        model = Model("LargeRoom", [room])
        drawing, report = build(model)
        readback = inspect_native(drawing.bytes(), report, include_geometry=True)
        native = readback["geometry"]["entities"][report["rooms"][0]["handle"]]
        x0 = min(point[0] for point in native["outline_mm"])
        y0 = min(point[1] for point in native["outline_mm"])
        x1 = max(point[0] for point in native["outline_mm"])
        y1 = max(point[1] for point in native["outline_mm"])
        native["outline_mm"] = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0 + 1000), (x0 + 5, y0 + 1000), (x0 + 5, y0)]
        result = compare_source(model, report, readback)["room_plans"][0]
        self.assertLess(result["missing_area_m2"], 0.01)
        self.assertEqual(result["boundary_hausdorff_mm"], 5)
        self.assertFalse(result["passed"])

    def test_envelope_calculation_regions_keep_the_separate_display_layer(self):
        lower = Room.from_box("Tall", 5, 4, 6)
        upper = Room.from_box("Upper", 5, 4, 3, origin=Point3D(10, 0, 3))
        lower.story, upper.story = "First", "Second"
        drawing, report = build(Model("Envelope", [lower, upper]))
        readback = inspect_native(drawing.bytes(), report, include_geometry=True)
        helpers = [row for row in report["building_outlines"] if row["story"].startswith("Envelope:")]
        self.assertTrue(helpers)
        self.assertTrue(readback["plan_display"]["passed"])
        for row in helpers:
            self.assertFalse(readback["geometry"]["entities"][row["handle"]]["layer_visible"])


if __name__ == "__main__":
    unittest.main()
