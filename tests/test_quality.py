from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from honeybee.model import Model
from honeybee.aperture import Aperture
from honeybee.room import Room
from ladybug_geometry.geometry3d import Face3D, Point3D
from shapely.geometry import box as rectangle

from honeybee_gbas.dwg import assign_wall_sides, build
from honeybee_gbas.dwg.validation import (
    ExportValidationError,
    compare_host_bindings,
    inspect_export,
    inspect_native,
    inspect_quality,
    require_export,
    validate,
)
from honeybee_gbas.dwg.walls import merge_coincident_walls


class NativeQualityTest(unittest.TestCase):
    def test_wall_inside_sloping_room_footprint_uses_source_outward_side(self):
        for forward in (True, False):
            first, last = (0, 0, 0), (3000, 0, 0)
            wall = {"handle": 20, "start_mm": first if forward else last, "end_mm": last if forward else first,
                    "left_space": 0, "right_space": 0, "thickness_mm": 200}
            owner = {"handle": 10, "identifier": "Overhang", "walls": [], "position_mm": (0, 0, 0)}
            section = SimpleNamespace(normal=SimpleNamespace(x=0, y=-1), room_identifier="Overhang")
            assign_wall_sides([wall], [owner], {"Overhang": rectangle(-100, -1000, 3100, 1000)}, [section])
            self.assertEqual((wall["left_space"], wall["right_space"]), (10, 0) if forward else (0, 10))
            self.assertEqual(owner["walls"], [20])

    def test_regular_top_flat_roof_uses_native_ceiling_generation(self):
        room = Room.from_box("Room", 5, 4, 3)
        room.story = "Occupied"
        model = Model("FlatRoof", [room])
        drawing, manifest = build(model)
        validate(model, drawing.bytes(), manifest)
        self.assertFalse(manifest["counts"]["native_roof_faces"])
        self.assertEqual(manifest["roof_selection"]["native_room_ceiling_faces"], 1)
        self.assertTrue(manifest["native_quality"]["passed"])

    def test_flat_skylight_uses_an_automatic_roof_and_one_definition(self):
        room = Room.from_box("Room", 5, 4, 3)
        room.story = "Occupied"
        roof = next(face for face in room.faces if str(face.type) == "RoofCeiling")
        roof.aperture_by_width_height(2, 1, 1)
        model = Model("SkylightException", [room])
        drawing, manifest = build(model)
        validate(model, drawing.bytes(), manifest)
        self.assertEqual(manifest["counts"]["native_roof_faces"], 0)
        self.assertEqual(manifest["roof_selection"]["native_room_ceiling_faces"], 1)
        self.assertEqual(manifest["validation"]["native_skylights"], 1)

    def test_shared_wall_uses_common_height_and_separate_upper_native_span(self):
        lower = Room.from_box("Lower", 5, 4, 3)
        tall = Room.from_box("Tall", 5, 4, 6, origin=Point3D(5, 0, 0))
        for room in (lower, tall):
            room.story = "Occupied"
        model = Model("SharedHeight", [tall, lower])
        drawing, manifest = build(model)
        validate(model, drawing.bytes(), manifest)
        self.assertEqual(manifest["wall_panels"], [])
        self.assertTrue(manifest["native_quality"]["passed"])
        shared = [row for row in manifest["walls"] if row["left_space"] and row["right_space"]]
        self.assertEqual(len(shared), 1)
        self.assertAlmostEqual(shared[0]["height_mm"], 3000)
        self.assertTrue(any(row["source_room"] == "Tall" for row in manifest["native_wall_completion"]["walls"]))

    def test_roof_selection_uses_the_final_primary_frame_ceiling(self):
        lower = Room.from_box("Lower", 5, 4, 3)
        raised = Room.from_box("Raised", 5, 4, 3, origin=Point3D(10, 0, 1))
        for room in (lower, raised):
            room.story = "Occupied"
        drawing, manifest = build(Model("FinalCeiling", [lower, raised]))
        policies = {row["source_room"]: row["policy"] for row in manifest["roof_selection"]["faces"]}
        self.assertEqual(policies["Lower"], "explicit_source_roof")
        self.assertEqual(policies["Raised"], "native_room_ceiling")

    def test_coincident_native_completion_reuses_a_two_sided_same_storey_wall(self):
        first = {"handle": 1, "identifier": "First", "story": "Occupied", "source_room": "Room1", "start_mm": (0, 0, 0), "end_mm": (4000, 0, 0), "height_mm": 3000, "left_space": 11, "right_space": 0, "openings": []}
        second = {**first, "handle": 2, "identifier": "Second", "source_room": "Room2", "start_mm": (4000, 0, 0), "end_mm": (0, 0, 0), "left_space": 12}
        report = {"floors": [{"story": "Occupied", "offset_mm": (0, 0), "reference_elevation_mm": 0}], "walls": [first, second], "rooms": [{"handle": 11, "walls": [1]}, {"handle": 12, "walls": [2]}], "openings": [], "wall_profile_geometry": [{"handle": 1}, {"handle": 2}], "elevated_walls": []}
        drawing = SimpleNamespace(entities=[SimpleNamespace(handle=1), SimpleNamespace(handle=2)])
        reused = merge_coincident_walls(drawing, report, {"First": "Physical", "Second": "Physical"})
        self.assertEqual(len(reused), 1)
        self.assertEqual(len(report["walls"]), 1)
        self.assertEqual((first["left_space"], first["right_space"]), (11, 12))
        self.assertEqual(first["source_identifiers"], ["First", "Second"])
        self.assertEqual(report["rooms"][1]["walls"], [1])

    def test_live_host_binding_requires_resolved_ids_and_wall_reverse_membership(self):
        room = Room.from_box("Room", 5, 4, 3)
        room.story = "Occupied"
        room.faces[1].apertures_by_ratio(0.2)
        drawing, manifest = build(Model("Bindings", [room]))
        opening = manifest["openings"][0]
        report = {"status": "completed", "opening_bindings": [{"handle": format(opening["handle"], "X"), "host_handle": format(opening["wall"], "X"), "host_class": "SWR_WALL", "wall_thickness_mm": 200, "reciprocal": True}]}
        self.assertTrue(compare_host_bindings(manifest, report)["passed"])
        report["opening_bindings"][0]["reciprocal"] = False
        self.assertFalse(compare_host_bindings(manifest, report)["passed"])

    def test_independent_quality_detects_a_window_moved_off_its_wall(self):
        room = Room.from_box("Room", 5, 4, 3)
        room.story = "Occupied"
        room.faces[1].apertures_by_ratio(0.2)
        drawing, manifest = build(Model("WindowSupport", [room]))
        readback = inspect_native(drawing.bytes(), manifest, include_geometry=True)
        self.assertTrue(inspect_quality(manifest, readback)["passed"])
        changed = copy.deepcopy(readback)
        row = changed["geometry"]["entities"][manifest["openings"][0]["handle"]]
        x, y, z = row["position_mm"]
        row["position_mm"] = (x + 1000, y + 1000, z + 1000)
        self.assertFalse(inspect_quality(manifest, changed)["passed"])
        changed = copy.deepcopy(readback)
        row = changed["geometry"]["entities"][manifest["openings"][0]["handle"]]
        row["angle_rad"] += 0.25
        self.assertFalse(inspect_quality(manifest, changed)["passed"])


    def test_lower_envelope_storey_keeps_upper_room_absolute_elevation(self):
        tall = Room.from_box("Tall", 5, 4, 6)
        upper = Room.from_box("Upper", 5, 4, 3, origin=Point3D(10, 0, 3))
        for room in (tall, upper):
            room.story = "First" if room is tall else "Second"
        model = Model("TallLower", [tall, upper])
        drawing, manifest = build(model, envelope_storeys=True)
        validate(model, drawing.bytes(), manifest)
        self.assertEqual(len(manifest["envelope_storeys"]), 1)
        helper = manifest["envelope_storeys"][0]
        self.assertEqual(helper["role"], "envelope")
        self.assertEqual(helper["height_mm"], 0.5)
        self.assertEqual(helper["reference_elevation_mm"], 2999.5)
        floor = next(row for row in manifest["floors"] if row["story"] == "Second")
        self.assertEqual(floor["floor_number"], 3)
        self.assertEqual(floor["reference_elevation_mm"], 3000)
        self.assertEqual(manifest["source_geometry_validation"]["summary"]["passed_room_plans"], 2)

    def test_frame_sequence_is_checked_against_actual_binary_labels(self):
        drawing, manifest = build(Model("Sequence", [Room.from_box("Room", 5, 4, 3)]))
        revised = copy.deepcopy(manifest)
        revised["roof_frame"]["floor_number"] += 1
        with self.assertRaisesRegex(ValueError, "frame sequence"):
            inspect_native(drawing.bytes(), revised)

    def test_lower_roof_skylight_is_native_and_binary_compared(self):
        lower = Room.from_box("Lower", 5, 4, 3)
        roof = next(face for face in lower.faces if str(face.type) == "RoofCeiling")
        roof.add_aperture(Aperture("LowerSkylight", Face3D(tuple(Point3D(x, y, 3) for x, y in ((1, 1), (2, 1), (2, 2), (1, 2))))))
        upper = Room.from_box("Upper", 5, 4, 3, origin=Point3D(10, 0, 3))
        lower.story, upper.story = "First", "Second"
        model = Model("LowerRoof", [lower, upper])
        drawing, manifest = build(model)
        validate(model, drawing.bytes(), manifest)
        self.assertEqual(manifest["counts"]["native_skylights"], 1)
        self.assertEqual(manifest["counts"]["reference_openings"], 0)
        sky, = manifest["defined_skylights"]
        self.assertEqual(sky["source_room"], "Lower")
        self.assertEqual(sky["story"], "First")
        self.assertEqual(manifest["source_geometry_validation"]["summary"]["passed_openings"], 1)


    def test_preimport_gate_requires_source_wall_coverage(self):
        model = Model("Coverage", [Room.from_box("Room", 5, 4, 3)])
        drawing, report = build(model)
        validate(model, drawing.bytes(), report)
        report["source_geometry_validation"]["summary"]["passed_room_wall_unions"] = 0
        result = inspect_export(report)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["source_wall_coverage"])
        with self.assertRaisesRegex(ExportValidationError, "source_wall_coverage"):
            require_export(report)

    def test_preimport_gate_requires_native_opening_hosts(self):
        room = Room.from_box("Window", 5, 4, 3)
        room.faces[1].apertures_by_ratio(0.2)
        model = Model("OpeningGate", [room])
        drawing, report = build(model)
        validate(model, drawing.bytes(), report)
        readback = inspect_native(drawing.bytes(), report, include_geometry=True)
        changed = copy.deepcopy(readback)
        row = changed["geometry"]["entities"][report["openings"][0]["handle"]]
        x, y, z = row["position_mm"]
        row["position_mm"] = (x + 1000, y + 1000, z)
        report["native_quality"] = inspect_quality(report, changed)
        result = inspect_export(report)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["native_opening_hosts"])
        with self.assertRaisesRegex(ExportValidationError, "native_opening_hosts"):
            require_export(report)


if __name__ == "__main__":
    unittest.main()
