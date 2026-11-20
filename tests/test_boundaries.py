from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

from honeybee.face import Face
from honeybee.facetype import face_types
from honeybee.model import Model
from honeybee.room import Room
from honeybee_energy.boundarycondition import Adiabatic
from ladybug_geometry.geometry3d import Face3D, Point3D

from honeybee_gbas import model as hb
from honeybee_gbas.dwg import build
from honeybee_gbas.dwg.metadata import compare_assembled_boundaries, compare_boundary_conditions
from honeybee_gbas.dwg.validation import inspect_native, validate


def box(identifier="Room", x=0):
    room = Room.from_box(identifier, 5, 4, 3, origin=Point3D(x, 0, 0))
    room.story = "Occupied"
    return room


class NativeBoundaryTest(unittest.TestCase):
    def test_adiabatic_wall_uses_native_boundary_property(self):
        room = box()
        face = next(face for face in room.faces if face.normal.y < -0.5)
        face.boundary_condition = Adiabatic()
        model = Model("InsulatedWall", [room])
        original = model.to_dict()
        drawing, manifest = build(model)
        validate(model, drawing.bytes(), manifest)
        row = next(row for row in manifest["walls"] if face.identifier in row.get("source_identifiers", []))
        native = inspect_native(drawing.bytes(), manifest, include_geometry=True)["geometry"]["entities"][row["handle"]]
        self.assertEqual(native["catalog"], 1)
        self.assertEqual(native["property_records"]["THSWRAE_WALL_XREC"][0], (301, "6"))
        self.assertEqual(manifest["counts"]["native_insulated_walls"], 1)
        self.assertTrue(manifest["source_geometry_validation"]["boundary_conditions"]["passed"])
        self.assertEqual(model.to_dict(), original)

    def test_air_boundary_is_a_positive_height_native_virtual_wall(self):
        room = box()
        face = next(face for face in room.faces if face.normal.x > 0.5)
        face.type = face_types.air_boundary
        model = Model("VirtualWall", [room])
        original = model.to_dict()
        drawing, manifest = build(model)
        validate(model, drawing.bytes(), manifest)
        row = next(row for row in manifest["walls"] if row.get("catalog") == 4)
        native = inspect_native(drawing.bytes(), manifest, include_geometry=True)["geometry"]["entities"][row["handle"]]
        self.assertEqual(native["base_mode"], 0)
        self.assertEqual(native["catalog"], 4)
        self.assertEqual(native["height_mm"], 3000)
        self.assertEqual(manifest["source_geometry_validation"]["summary"]["passed_room_wall_unions"], 1)
        self.assertTrue(manifest["source_geometry_validation"]["boundary_conditions"]["passed"])
        self.assertEqual(model.to_dict(), original)

    def test_paired_air_faces_share_one_native_virtual_wall(self):
        first, second = box("First"), box("Second", 5)
        Room.solve_adjacency([first, second], 0.01)
        source = []
        for room in (first, second):
            for face in room.faces:
                if str(face.boundary_condition) == "Surface":
                    face.type = face_types.air_boundary
                    source.append(face.identifier)
        model = Model("SharedVirtualWall", [first, second])
        drawing, manifest = build(model)
        validate(model, drawing.bytes(), manifest)
        walls = [row for row in manifest["walls"] if row.get("catalog") == 4]
        self.assertEqual(len(walls), 1)
        self.assertTrue(walls[0]["left_space"] and walls[0]["right_space"])
        records = manifest["source_geometry_validation"]["boundary_conditions"]["records"]
        self.assertEqual({row["identifier"] for row in records}, set(source))
        self.assertTrue(all(row["passed"] for row in records))

    def test_collinear_adiabatic_seam_remains_a_native_partition(self):
        room = box()
        original = next(face for face in room.faces if face.normal.y > 0.5)
        def part(identifier, low, high):
            return Face(identifier, Face3D(tuple(Point3D(x, 4, z) for x, z in ((low, 0), (low, 3), (high, 3), (high, 0)))))
        first, second = part("InsulatedPart", 0, 2.5), part("OutdoorPart", 2.5, 5)
        first.boundary_condition = Adiabatic()
        room = Room("Room", [face for face in room.faces if face is not original] + [first, second])
        room.story = "Occupied"
        model = Model("BoundarySeam", [room])
        drawing, manifest = build(model)
        validate(model, drawing.bytes(), manifest)
        self.assertEqual(manifest["counts"]["walls"], 5)
        self.assertEqual(manifest["counts"]["native_insulated_walls"], 1)
        self.assertTrue(all(set(row.get("source_identifiers", [])) != {first.identifier, second.identifier} for row in manifest["walls"]))
        self.assertTrue(manifest["source_geometry_validation"]["boundary_conditions"]["passed"])

    def test_adiabatic_floor_and_roof_keep_explicit_native_properties(self):
        room = box()
        for face in room.faces:
            if str(face.type) in {"Floor", "RoofCeiling"}:
                face.boundary_condition = Adiabatic()
        model = Model("InsulatedHorizontal", [room])
        drawing, manifest = build(model)
        validate(model, drawing.bytes(), manifest)
        native = inspect_native(drawing.bytes(), manifest, include_geometry=True)["geometry"]["entities"]
        self.assertEqual(native[manifest["rooms"][0]["handle"]]["property_records"]["BECS_ROOM_EXT_FLOOR_INSULATE"], [(280, 1), (301, "1")])
        roofs = [row for row in manifest["surfaces"] if row["kind"] == "RoofCeiling"]
        self.assertTrue(roofs)
        self.assertTrue(all(native[row["handle"]]["board_type"] == 1 for row in roofs))
        self.assertTrue(manifest["source_geometry_validation"]["boundary_conditions"]["passed"])

    def test_full_air_floor_uses_native_room_has_floor_attribute(self):
        room = box()
        next(face for face in room.faces if str(face.type) == "Floor").type = face_types.air_boundary
        model = Model("VirtualFloor", [room])
        drawing, manifest = build(model)
        validate(model, drawing.bytes(), manifest)
        native = inspect_native(drawing.bytes(), manifest, include_geometry=True)["geometry"]["entities"]
        space = native[manifest["rooms"][0]["handle"]]
        self.assertEqual(space["property_records"]["THSWARE_ROOM_XREC"][8], (301, "0"))
        self.assertEqual(space["flags"] & 8, 8)
        self.assertEqual(manifest["counts"]["virtual_floor_rooms"], 1)
        self.assertEqual(manifest["rooms"][0]["area_m2"], 20)
        self.assertEqual(manifest["rooms"][0]["source_area_m2"], 0)
        changed = copy.deepcopy(native)
        changed[space["handle"]]["property_records"]["THSWARE_ROOM_XREC"][8] = (301, "1")
        self.assertFalse(compare_boundary_conditions(model, manifest, changed)["passed"])
        changed = copy.deepcopy(native)
        changed[space["handle"]]["flags"] &= ~8
        self.assertFalse(compare_boundary_conditions(model, manifest, changed)["passed"])

    def test_adjacent_air_ceiling_uses_the_shared_upper_virtual_floor(self):
        lower = box("Lower")
        upper = Room.from_box("Upper", 5, 4, 3, origin=Point3D(0, 0, 3))
        upper.story = "Upper"
        Room.solve_adjacency([lower, upper], 0.01)
        source = []
        for room in (lower, upper):
            for face in room.faces:
                if str(face.boundary_condition) == "Surface":
                    face.type = face_types.air_boundary
                    source.append(face.identifier)
        model = Model("SharedVirtualFloor", [lower, upper])
        drawing, manifest = build(model)
        validate(model, drawing.bytes(), manifest)
        records = manifest["source_geometry_validation"]["boundary_conditions"]
        self.assertTrue(records["passed"])
        self.assertEqual({row["identifier"] for row in records["records"]}, set(source))
        self.assertEqual(manifest["boundary_conditions"]["issues"], [])
        self.assertEqual(manifest["source_geometry_validation"]["summary"]["native_face_review"], 0)

    def test_partial_air_floor_keeps_full_plan_and_physical_floor_area(self):
        room = box()
        original = next(face for face in room.faces if str(face.type) == "Floor")
        def floor(identifier, low, high, kind):
            return Face(identifier, Face3D(tuple(Point3D(x, y, 0) for x, y in ((low, 0), (low, 4), (high, 4), (high, 0)))), kind)
        physical = floor("PhysicalFloor", 0, 2.5, face_types.floor)
        air = floor("AirFloor", 2.5, 5, face_types.air_boundary)
        room = Room("Room", [face for face in room.faces if face is not original] + [physical, air])
        room.story = "Occupied"
        model = Model("PartialVirtualFloor", [room])
        self.assertEqual(hb.room_floor_area(room), 10)
        self.assertAlmostEqual(hb.room_plan_boundaries(room)[0].area, 20)
        drawing, manifest = build(model)
        validate(model, drawing.bytes(), manifest)
        self.assertEqual(manifest["rooms"][0]["area_m2"], 20)
        self.assertEqual(manifest["rooms"][0]["source_area_m2"], 10)
        self.assertEqual(manifest["rooms"][0]["plan_boundary_basis"], "source_floor_and_downward_air_boundary")
        self.assertTrue(any(row["code"] == "partial_air_floor_partition_pending" for row in manifest["boundary_conditions"]["issues"]))
        self.assertEqual(manifest["source_geometry_validation"]["summary"]["passed_room_plans"], 1)

    def test_host_boundary_comparison_checks_flags_and_shared_floor_identity(self):
        lower = box("Lower")
        upper = Room.from_box("Upper", 5, 4, 3, origin=Point3D(0, 0, 3))
        upper.story = "Upper"
        Room.solve_adjacency([lower, upper], 0.01)
        for room in (lower, upper):
            for face in room.faces:
                if str(face.boundary_condition) == "Surface":
                    face.type = face_types.air_boundary
        insulated = next(face for face in lower.faces if face.normal.x < -0.5)
        insulated.boundary_condition = Adiabatic()
        model = Model("NativeFlags", [lower, upper])
        drawing, manifest = build(model)
        validate(model, drawing.bytes(), manifest)
        wall = next(row for row in manifest["walls"] if row.get("border_type") == 6)
        floor = {"native_id": "100", "flags_hex": "00000040"}
        assembly = {"status": "completed", "result": {"assembled": True, "floors": [{"room_details": [{"code": "Lower", "flags_hex": "00000000", "ceiling_object_ids": ["100"]}, {"code": "Upper", "flags_hex": "00000002", "floor_object_ids": ["100"], "floor_details": [floor]}], "wall_details": [{"handle": format(wall["handle"], "X"), "flags_hex": "00000002"}]}]}}
        self.assertTrue(compare_assembled_boundaries(model, manifest, assembly)["passed"])
        changed = copy.deepcopy(assembly)
        changed["result"]["floors"][0]["room_details"][1]["floor_details"][0]["flags_hex"] = "00000000"
        self.assertFalse(compare_assembled_boundaries(model, manifest, changed)["passed"])
        changed = copy.deepcopy(assembly)
        changed["result"]["floors"][0]["room_details"][0]["ceiling_object_ids"] = ["200"]
        self.assertFalse(compare_assembled_boundaries(model, manifest, changed)["passed"])


if __name__ == "__main__":
    unittest.main()
