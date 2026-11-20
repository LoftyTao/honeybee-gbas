from __future__ import annotations

import unittest
from pathlib import Path

from honeybee.model import Model
from shapely import set_precision
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from honeybee_gbas import model as hb
from honeybee_gbas.dwg import build_plan, building_area_regions
from honeybee_gbas.dwg.records import add_native, attach_room_properties, new_drawing, oriented, space
from honeybee_gbas.dwg.validation import compare_building_outlines, inspect_native
from honeybee_gbas.layout import plan_native_levels


ROOT = Path(__file__).resolve().parents[1]


def source_plans(model, floor):
    scale = hb.unit_scale(model.units)
    return [Polygon([(point.x * scale, point.y * scale) for point in face.boundary],
                    [[(point.x * scale, point.y * scale) for point in ring] for ring in face.holes or ()])
            for room in model.rooms if room.identifier in floor.room_identifiers
            for face in hb.room_plan_boundaries(room)]


def fixture(source, parts):
    manifest = {"floors": [{"story": "Floor", "offset_mm": (0, 0)}],
                "building_outline_precision_mm": 0.01, "building_outlines": []}
    entities = {}
    for handle, part in enumerate(parts, 1):
        seed = part.representative_point()
        row = {"identifier": f"Floor:{handle}", "handle": handle, "story": "Floor"}
        manifest["building_outlines"].append(row)
        entities[handle] = {"outline_mm": oriented(list(part.exterior.coords), False),
                            "holes_mm": [list(ring.coords)[:-1] for ring in part.interiors],
                            "position_mm": (seed.x, seed.y, 0), "category": 1,
                            "area_m2": part.area / 1e6, "statistics": [part.area / 1e6] * 2}
    return manifest, entities, {"Floor": [source]}


class BuildingRegionGeometryTest(unittest.TestCase):
    def assert_partition(self, source, parts):
        self.assertTrue(parts)
        union = unary_union(parts)
        limit = max(1.0, source.area * 1e-10)
        self.assertLessEqual(source.symmetric_difference(union).area, limit)
        self.assertLessEqual(abs(source.area - sum(part.area for part in parts)), limit)
        self.assertLessEqual(sum(part.area for part in parts) - union.area, limit)
        for part in parts:
            self.assertTrue(part.is_valid)
            self.assertGreater(part.area, 0)
            self.assertEqual(len(part.interiors), 0)
            self.assertLessEqual(part.convex_hull.difference(part).area, max(1e-6, part.area * 1e-12))
            seed = part.representative_point()
            self.assertTrue(part.contains(seed))
            self.assertEqual(sum(other.covers(seed) for other in parts), 1)

    def test_deep_concavities_remain_outside_every_native_region(self):
        source = Polygon(((0, 0), (10000, 0), (10000, 10000), (7000, 10000),
                          (7000, 3000), (3000, 3000), (3000, 10000), (0, 10000)))
        parts = building_area_regions(source)
        self.assert_partition(source, parts)
        self.assertGreater(len(parts), 1)
        for point in (Point(5000, 4000), Point(5000, 9000), Point(15000, 5000)):
            self.assertTrue(all(not part.covers(point) for part in parts))
        self.assertTrue(compare_building_outlines(*fixture(source, parts))["passed"])

    def test_multiple_courtyards_keep_their_full_boundaries(self):
        source = Polygon(box(0, 0, 20000, 12000).exterior,
                         [box(1000, 1000, 4000, 9000).exterior, box(9000, 4000, 19000, 8000).exterior])
        parts = building_area_regions(source)
        self.assert_partition(source, parts)
        result = compare_building_outlines(*fixture(source, parts))
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["records"][0]["boundary_hausdorff_mm"], 0)

    def test_small_area_deep_notch_keeps_its_geometry_at_large_scales(self):
        source = box(0, 0, 1e7, 1e7).difference(box(4999999.999, 0, 5000000.001, 10000))
        self.assertLess(source.convex_hull.difference(source).area, source.area * 1e-12)
        parts = building_area_regions(source)
        self.assert_partition(source, parts)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(not part.covers(Point(5000000, 5000)) for part in parts))
        self.assertFalse(compare_building_outlines(*fixture(source, [source]))["passed"])

    def test_non_sydney_sdk_floors_preserve_each_source_boundary(self):
        paths = [path for path in (ROOT / "samples").glob("*/*.hbjson")
                 if path.stem != "Sydney_Opera_House"]
        for path in paths:
            model = Model.from_file(str(path))
            for floor in plan_native_levels(model, include_orphan_shades=False):
                with self.subTest(model=path.stem, story=floor.identifier):
                    source = unary_union(source_plans(model, floor))
                    union = unary_union([set_precision(shape, 0.01) for shape in source_plans(model, floor)])
                    components = list(union.geoms) if union.geom_type == "MultiPolygon" else [union]
                    parts = [part for component in components for part in building_area_regions(component)]
                    self.assert_partition(union, parts)
                    result = compare_building_outlines(*fixture(source, parts))
                    self.assertTrue(result["passed"], result["records"])

    def test_dieting_native_3f_preserves_the_bridge_and_courtyard_exclusions(self):
        model = Model.from_file(str(ROOT / "samples/Dieting_Spatial_Building/Dieting_Spatial_Building.hbjson"))
        floor = next(row for row in plan_native_levels(model, include_orphan_shades=False) if row.story.startswith("2"))
        plans = source_plans(model, floor)
        source = unary_union(plans)
        self.assertAlmostEqual(source.area / 1e6, 1709.998416, places=6)
        self.assertEqual(len(source.interiors), 1)
        selected = Model("Dieting_regions", [room.duplicate() for room in model.rooms if room.identifier in floor.room_identifiers],
                         units=model.units, tolerance=model.tolerance, angle_tolerance=model.angle_tolerance)
        drawing, report = build_plan(selected, surface_mode="deferred", floor_slabs=False, primary_levels=True)
        entities = inspect_native(drawing.bytes(), report, include_geometry=True)["geometry"]["entities"]
        result = compare_building_outlines(report, entities, {floor.identifier: plans})
        self.assertTrue(result["passed"], result["records"])
        row = result["records"][0]
        self.assertGreater(row["native_regions"], 2)
        self.assertAlmostEqual(row["boundary_hausdorff_mm"], 0)
        dx, dy = report["floors"][0]["offset_mm"]
        parts = [Polygon([(x - dx, y - dy) for x, y in entities[entry["handle"]]["outline_mm"]])
                 for entry in report["building_outlines"]]
        self.assert_partition(source, parts)
        for point in (Point(27000, 2000), Point(27000, 24000)):
            self.assertFalse(source.covers(point))
            self.assertTrue(all(not part.covers(point) for part in parts))
        bridge = Point(27000, 15500)
        self.assertTrue(source.covers(bridge))
        self.assertEqual(sum(part.covers(bridge) for part in parts), 1)


class BuildingRegionValidationTest(unittest.TestCase):
    def test_small_translated_region_keeps_winding_and_native_area_caches(self):
        offset = (1e9, -1e9)
        shape = Polygon([(0, 0), (3, 0), (1, 2)])
        manifest, _, plans = fixture(shape, [shape])
        floor = manifest["floors"][0]
        floor["offset_mm"] = offset
        drawing, codes = new_drawing()
        points = [(x + offset[0], y + offset[1]) for x, y in shape.exterior.coords]
        row = manifest["building_outlines"][0]
        row.update(name="Region", outline_mm=oriented(points, False), category=1, walls=[],
                   position_mm=(offset[0] + 1, offset[1] + 1, 0), height_mm=3000,
                   area_m2=shape.area / 1e6, perimeter_mm=shape.length)
        entity = add_native(drawing, codes["SWR_SPACE"], "建-建筑区域", row, space)
        attach_room_properties(drawing, codes, entity, [])
        native = inspect_native(drawing.bytes(), include_geometry=True)["geometry"]["entities"]
        self.assertTrue(compare_building_outlines(manifest, native, plans)["passed"])

    def test_single_concave_face_requires_a_rendering_safe_partition(self):
        source = Polygon(((0, 0), (10000, 0), (10000, 10000), (7000, 10000),
                          (7000, 3000), (3000, 3000), (3000, 10000), (0, 10000)))
        result = compare_building_outlines(*fixture(source, [source]))
        row = result["records"][0]
        self.assertTrue(row["coverage_passed"])
        self.assertTrue(row["boundary_passed"])
        self.assertFalse(row["native_fill_passed"])
        self.assertFalse(result["passed"])

    def test_filled_deep_narrow_notch_fails_the_boundary_gate(self):
        actual = box(0, 0, 100000, 100000)
        source = actual.difference(box(49999.99, 0, 50000.01, 50000))
        result = compare_building_outlines(*fixture(source, [actual]))
        row = result["records"][0]
        self.assertTrue(row["coverage_passed"])
        self.assertTrue(row["native_fill_passed"])
        self.assertFalse(row["boundary_passed"])
        self.assertGreater(row["boundary_hausdorff_mm"], 1000)
        self.assertFalse(result["passed"])

    def test_seed_on_the_shared_boundary_requires_strict_interior_placement(self):
        source = box(0, 0, 10000, 10000)
        manifest, entities, plans = fixture(source, [box(0, 0, 5000, 10000), box(5000, 0, 10000, 10000)])
        entities[1]["position_mm"] = (5000, 5000, 0)
        result = compare_building_outlines(manifest, entities, plans)
        self.assertFalse(result["passed"])
        row = result["records"][0]
        self.assertFalse(row["regions"][0]["seed_strictly_inside"])
        self.assertEqual(row["regions"][0]["seed_region_count"], 2)
        self.assertFalse(row["uniqueness_passed"])

    def test_equal_summed_area_keeps_overlap_and_missing_coverage_separate(self):
        source = box(0, 0, 10000, 10000)
        parts = [box(0, 0, 6000, 10000), box(4000, 0, 8000, 10000)]
        self.assertEqual(sum(part.area for part in parts), source.area)
        result = compare_building_outlines(*fixture(source, parts))
        row = result["records"][0]
        self.assertFalse(result["passed"])
        self.assertFalse(row["coverage_passed"])
        self.assertFalse(row["uniqueness_passed"])
        self.assertAlmostEqual(row["overlap_area_m2"], 20)

    def test_stale_native_area_cache_fails_with_exact_boundary_coverage(self):
        source = box(0, 0, 10000, 10000)
        for index in (0, 1):
            with self.subTest(cache=index):
                manifest, entities, plans = fixture(source, [source])
                entities[1]["statistics"][index] += 10
                result = compare_building_outlines(manifest, entities, plans)
                self.assertFalse(result["passed"])
                self.assertAlmostEqual(result["records"][0]["symmetric_difference_m2"], 0)
                self.assertFalse(result["records"][0]["regions"][0]["fields_valid"])


if __name__ == "__main__":
    unittest.main()
