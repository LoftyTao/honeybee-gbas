from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from honeybee.model import Model
from honeybee.room import Room
from honeybee.shade import Shade
from ladybug_geometry.geometry3d import Face3D, Point3D
from shapely.geometry import Polygon
from shapely.ops import unary_union

from honeybee_gbas import model as hb
from honeybee_gbas.dwg import building_area_regions

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "samples" / "basic_sample_model" / "basic_sample_model.hbjson"


class ModelHelperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = Model.from_file(str(MODEL))

    def test_story_order(self) -> None:
        self.assertEqual(hb.story_names(self.model), ["Level 1", "Level 2"])

    def test_room_metrics(self) -> None:
        room = self.model.rooms[0]
        self.assertGreater(hb.room_floor_area(room), 0)
        self.assertGreater(hb.room_height(room), 0)
        self.assertEqual(len(hb.room_floor_boundary(room)), 10)
        self.assertEqual(len(hb.room_plan_boundaries(room)), 1)
        self.assertIsNotNone(hb.room_floor_face(room))

    def test_wall_segments(self) -> None:
        segments = hb.wall_segments(self.model, "Level 1")
        self.assertGreater(len(segments), 10)
        keys = {hb.segment_key(segment.start, segment.end) for segment in segments}
        self.assertEqual(len(keys), len(segments))
        self.assertTrue(all(segment.length > 0 for segment in segments))
        self.assertTrue(all(segment.thickness_m > 0 for segment in segments))

    def test_openings(self) -> None:
        openings = hb.openings(self.model)
        self.assertEqual(len(openings), 66)
        self.assertTrue(all(opening.width > 0 for opening in openings))
        self.assertTrue(all(opening.height > 0 for opening in openings))
        doors = [opening for opening in openings if opening.kind == "Door"]
        windows = [opening for opening in openings if opening.kind == "Window"]
        self.assertGreater(len(doors), 0)
        self.assertGreater(len(windows), 0)
        self.assertTrue(all(abs(opening.sill) < 0.01 for opening in doors))
        self.assertTrue(all(opening.sill >= 0 for opening in windows))

    def test_opening_thermal_values_follow_model_units(self) -> None:
        model = self.model.duplicate()
        feet = {opening.identifier: opening for opening in hb.openings(model)}
        model.convert_to_units("Meters")
        for opening in hb.openings(model):
            self.assertAlmostEqual(opening.u_factor, feet[opening.identifier].u_factor)
            self.assertAlmostEqual(opening.shgc, feet[opening.identifier].shgc)

    def test_inclined_opening_keeps_surface_height(self) -> None:
        inclined = [opening for opening in hb.openings(self.model) if 1e-4 < abs(opening.normal.z) < 0.9999]
        self.assertGreater(len(inclined), 0)
        self.assertTrue(all(opening.height > opening.vertical_height for opening in inclined))


class CalculationCodeTest(unittest.TestCase):
    def test_long_shared_prefixes_have_unique_bounded_symbols(self):
        identifiers = ["Room_e6ac360b-aaed-4c3b-a130-36b4c2ac9d13-" + suffix for suffix in ("000d15d0", "000d14bf")]
        codes = hb.room_codes(identifiers)
        self.assertEqual(len(set(codes.values())), 2)
        self.assertTrue(all(len(("Rm-" + code).encode("ascii")) == 16 for code in codes.values()))
        self.assertEqual(codes[identifiers[0]], "H87f97d8d124d")

    def test_source_order_preserves_room_identity(self):
        self.assertEqual(hb.room_codes(["a", "b"]), hb.room_codes(["b", "a"]))
        self.assertEqual(len(hb.room_code("房间")), 13)

    def test_duplicate_source_identities_require_review(self):
        with self.assertRaises(ValueError):
            hb.room_codes(["room", "room"])

    def test_hash_collision_requires_review(self):
        with patch("honeybee_gbas.model.room_code", return_value="H000000000000"):
            with self.assertRaises(ValueError):
                hb.room_codes(["a", "b"])

    def test_empty_source_identity_requires_review(self):
        for identifier in ("", None, 3):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                hb.room_code(identifier)

    def test_native_mapping_retains_manifest_support(self):
        self.assertEqual(hb.source_room_codes({"rooms": [{"identifier": "source", "native_code": "short"}, {"identifier": "old"}]}), {"short": "source", "old": "old"})

    def test_duplicate_native_mapping_requires_review(self):
        with self.assertRaises(ValueError):
            hb.source_room_codes({"rooms": [{"identifier": "a", "native_code": "short"}, {"identifier": "b", "native_code": "short"}]})


class ModelInspectionTest(unittest.TestCase):
    def test_box_candidate(self) -> None:
        report = hb.inspect_model(Model("box", [Room.from_box("room")]))
        self.assertEqual(report["route"], "native_plan")
        self.assertEqual(report["wall_faces"], 4)
        self.assertEqual(report["issues"], [])

    def test_roof_openings_are_identified(self) -> None:
        model = Model.from_file(str(ROOT / "samples/two_rooms/two_rooms.hbjson"))
        report = hb.inspect_model(model)
        self.assertEqual(report["issue_counts"]["roof_or_floor_opening"], 3)
        self.assertEqual(report["route"], "plan_adaptation")

    def test_shades_require_calculation_adaptation(self) -> None:
        shade = Shade("Shade", Face3D(tuple(Point3D(x, y, 4) for x, y in ((0, 0), (5, 0), (5, 4), (0, 4)))))
        model = Model("Shaded", [Room.from_box("Room", 5, 4, 3)], orphaned_shades=[shade])
        report = hb.inspect_model(model)
        self.assertEqual(len(model.shades), 1)
        self.assertEqual(report["issue_counts"]["shade_geometry"], len(model.shades))
        self.assertEqual(report["route"], "plan_adaptation")

    def test_building_area_regions_preserve_courtyards_with_sdk_mesh_fallback(self):
        polygon = Polygon(((0, 0), (10000, 0), (10000, 8000), (0, 8000)),
                          [((1000, 1000), (3000, 1000), (3000, 3000), (1000, 3000)),
                           ((5000, 1000), (8000, 1000), (8000, 6000), (5000, 6000))])
        for fallback in (False, True):
            with self.subTest(fallback=fallback):
                if fallback:
                    with patch.object(Face3D, "split_through_holes", lambda face: [Face3D(face.boundary)]):
                        parts = building_area_regions(polygon)
                else:
                    parts = building_area_regions(polygon)
                self.assertTrue(all(part.is_valid and not part.interiors for part in parts))
                self.assertAlmostEqual(sum(part.area for part in parts), polygon.area)
                self.assertLessEqual(polygon.symmetric_difference(unary_union(parts)).area, 1.0)

    def test_inspection_is_json_safe_and_preserves_model(self):
        room = Room.from_box("Room", 5, 4, 3)
        model = Model("Codes", [room])
        original = copy.deepcopy(model.to_dict())
        report = hb.inspect_model(model)
        json.dumps(report, allow_nan=False)
        self.assertEqual(model.to_dict(), original)

    def test_model_floor_elevation_is_absolute(self):
        room = Room.from_box("Raised", 5, 4, 3, origin=Point3D(0, 0, 2.5))
        self.assertAlmostEqual(hb.room_floor_elevation(room), 2.5)
        self.assertAlmostEqual(hb.room_height(room), 3)


if __name__ == "__main__":
    unittest.main()
