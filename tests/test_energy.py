from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from honeybee.face import Face
from honeybee.boundarycondition import Outdoors, Surface
from honeybee.model import Model
from honeybee.room import Room
from honeybee_energy.lib.constructions import generic_exterior_wall, generic_interior_wall
from honeybee_energy.lib.programtypes import program_type_by_identifier
from honeybee_energy.load.setpoint import Setpoint
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.material.opaque import EnergyMaterialNoMass
from honeybee_energy.schedule.ruleset import ScheduleRuleset
from ladybug_geometry.geometry3d import Face3D, Point3D

from honeybee_gbas import energy as native_energy
from honeybee_gbas.dwg import build
from honeybee_gbas.dwg.validation import inspect_export, validate


class NativeEnergyTest(unittest.TestCase):
    def test_native_room_usage_uses_display_names_and_preserves_source_programs(self):
        rooms = [Room.from_box(identifier, 5, 4, 3, origin=Point3D(index * 5, 0, 0)) for index, identifier in enumerate(("First", "Second"))]
        rooms[0].display_name, rooms[1].display_name = "首层办公区", "二层办公区"
        model = Model("ReadableRooms", rooms)
        original = model.to_dict()
        library, records, assignments = native_energy.plan_room_types(model)
        self.assertEqual(assignments, {room.identifier: room.display_name for room in rooms})
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["program_key"], records[1]["program_key"])
        self.assertEqual(records[0]["sections"]["能耗分析"], records[1]["sections"]["能耗分析"])
        self.assertEqual(model.to_dict(), original)
        for item, room in zip(library["children"][0]["children"], rooms):
            values = {name: value for name, kind, value in item["properties"]}
            self.assertEqual(values["Name"], room.display_name)

    def test_carrier_construction_roles_preserve_source_layers_and_glazing(self):
        room = Room.from_box("CarrierRoles", 5, 4, 3)
        wall = room[1]
        wall.aperture_by_width_height(1, 1, 1)
        aperture = wall.apertures[0]
        model = Model("CarrierRoles", [room])
        original = model.to_dict()
        plan = native_energy.plan_constructions(model, native_roles={"屋顶": {wall.identifier}, "天窗": {aperture.identifier}})
        values = {row["id"]: row for row in plan.constructions}
        for role, component in (("屋顶", wall), ("天窗", aperture)):
            source = values[plan.assignments[component.identifier]]
            native = values[plan.role_assignments[role][component.identifier]]
            self.assertEqual(native["category"], role)
            self.assertEqual(native["source_construction_id"], source["id"])
            self.assertEqual(native["layers"], source["layers"])
            self.assertEqual(native["u_factor"], source["u_factor"])
            self.assertEqual(native["shgc"], source["shgc"])
            self.assertEqual(native["sources"], [component.identifier])
        self.assertEqual(model.to_dict(), original)

    def test_room_setpoints_override_official_program_through_sdk_objects(self):
        program = program_type_by_identifier("2019::MediumOffice::OpenOffice")
        rooms = [Room.from_box(name, 5, 4, 3) for name in ("Inherited", "Override")]
        for room in rooms:
            room.properties.energy.program_type = program
        rooms[1].properties.energy.setpoint = Setpoint("RoomSetpoint", ScheduleRuleset.from_constant_value("Heating", 18), ScheduleRuleset.from_constant_value("Cooling", 27), ScheduleRuleset.from_constant_value("Humidity", 55), ScheduleRuleset.from_constant_value("Dehumidity", 55))
        model = Model("RoomOverrides", rooms)
        original = model.to_dict()
        library, records, assignments = native_energy.plan_room_types(model)
        values = {row["name"]: row for row in records}
        inherited = values[assignments["Inherited"]]
        override = values[assignments["Override"]]
        self.assertNotEqual(assignments["Inherited"], assignments["Override"])
        self.assertEqual(override["program_identifier"], program.identifier)
        self.assertEqual(assignments["Override"], "Override")
        self.assertEqual(assignments["Inherited"], "Inherited")
        self.assertEqual(override["room_load_overrides"], ["setpoint"])
        self.assertEqual(override["sections"]["能耗分析"]["冬季温度"], "18")
        self.assertEqual(override["sections"]["能耗分析"]["夏季温度"], "27")
        self.assertEqual(override["sections"]["能耗分析"]["冬季湿度"], "55")
        self.assertEqual(inherited["sections"]["能耗分析"]["冬季温度"], "21")
        self.assertEqual(model.to_dict(), original)

    def test_opaque_layers_keep_sdk_si_properties_and_native_millimeters(self):
        room = Room.from_box("Room", 5, 4, 3)
        model = Model("Model", [room])
        plan = native_energy.plan_constructions(model)
        record = next(row for row in plan.constructions if row["identifier"] == generic_exterior_wall.identifier)
        self.assertAlmostEqual(record["u_factor"], generic_exterior_wall.u_factor)
        self.assertAlmostEqual(record["r_value"], generic_exterior_wall.r_value)
        self.assertAlmostEqual(sum(row["thickness_mm"] for row in record["layers"]), generic_exterior_wall.thickness * 1000)
        materials = {row["id"]: row for row in plan.materials}
        for source, layer in zip(generic_exterior_wall.materials, record["layers"]):
            native = materials[layer["material_id"]]
            self.assertEqual(native["conductivity"], source.conductivity)
            self.assertEqual(native["density"], source.density)
            self.assertEqual(native["specific_heat"], source.specific_heat)
            self.assertGreater(native["heat_storage_coefficient"], 0)
            values = {name: value for name, kind, value in native_energy.material_node(native)["properties"]}
            self.assertEqual(values["HeatStore"], native["heat_storage_coefficient"])
            extension = native_energy.material_node(native)["children"][0]
            self.assertEqual(extension["key"], "DE")
            values = {name: value for name, kind, value in extension["properties"]}
            self.assertAlmostEqual(values["Thick"], source.thickness * 1000)
            self.assertEqual(values["RouVal"], source.solar_absorptance)

    def test_window_shgc_uses_native_sc_conversion(self):
        room = Room.from_box("Room", 5, 4, 3)
        room[1].aperture_by_width_height(1, 1, 1)
        plan = native_energy.plan_constructions(Model("Model", [room]))
        window = next(row for row in plan.constructions if row["category"] == "外窗")
        self.assertGreater(window["shgc"], 0)
        properties = {name: value for name, kind, value in native_energy.construction_node(window)["properties"]}
        self.assertAlmostEqual(properties["SC"] * 0.87, window["shgc"])

    def test_massless_layer_uses_native_direct_resistance_material(self):
        room = Room.from_box("Massless", 5, 4, 3)
        material = EnergyMaterialNoMass("SourceResistance", r_value=1.25)
        room[0].properties.energy.construction = OpaqueConstruction("MasslessFloor", [material])
        plan = native_energy.plan_constructions(Model("Massless", [room]))
        source = next(row for row in plan.materials if row["identifier"] == material.identifier)
        construction = next(row for row in plan.constructions if row["identifier"] == "MasslessFloor")
        self.assertEqual(source["native_material_type"], 2)
        self.assertEqual(source["r_value"], material.r_value)
        self.assertEqual(source["thickness_mm"], 0)
        self.assertEqual(len(construction["layers"]), 1)
        self.assertEqual(construction["r_value"], material.r_value)
        values = dict((name, value) for name, kind, value in native_energy.material_node(source)["children"][0]["properties"])
        self.assertEqual(values["MatType"], 2)
        self.assertEqual(values["HeRVal"], material.r_value)

    def test_program_type_is_embedded_as_custom_native_room_function(self):
        room = Room.from_box("Room", 5, 4, 3)
        program = program_type_by_identifier("2019::MediumOffice::OpenOffice")
        room.properties.energy.program_type = program
        library, records, assignments = native_energy.plan_room_types(Model("Model", [room]))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["program_identifier"], program.identifier)
        self.assertEqual(assignments[room.identifier], room.display_name)
        item = library["children"][0]["children"][0]
        properties = {name: value for name, kind, value in item["properties"]}
        self.assertEqual(properties["Comment"], program.identifier)
        values = {name: value for name, kind, value in item["children"][0]["properties"]}
        self.assertEqual(values["PersonUnit"], 1)
        self.assertEqual(values["LightUnit"], 2)
        self.assertAlmostEqual(values["Light"], program.lighting.watts_per_area, places=3)
        self.assertAlmostEqual(values["People"], program.people.area_per_person, places=3)

    def test_different_wall_constructions_keep_distinct_native_parts(self):
        room = Room.from_box("Room", 5, 4, 3)
        source = next(face for face in room.faces if face.normal.y < -0.5)
        first, last = source.geometry.lower_left_counter_clockwise_vertices[:2]
        def point(fraction, z):
            return Point3D(first.x + (last.x - first.x) * fraction, first.y + (last.y - first.y) * fraction, z)
        faces = [face.duplicate() for face in room.faces if face is not source]
        for identifier, low, high, construction in (("Lower", 0, 1.5, generic_interior_wall), ("Upper", 1.5, 3, generic_exterior_wall)):
            face = Face(identifier, Face3D((point(0, low), point(1, low), point(1, high), point(0, high))))
            face.properties.energy.construction = construction
            faces.append(face)
        model = Model("Partition", [Room("Room", faces)])
        drawing, report = build(model)
        validate(model, drawing.bytes(), report)
        self.assertEqual(report["source_energy"]["binding_conflicts"], [])
        assignments = report["source_energy"]["assignments"]
        self.assertNotEqual(assignments["Lower"], assignments["Upper"])
        self.assertEqual(report["source_geometry_validation"]["summary"]["review_items"], 0)
        self.assertEqual(report["counts"]["native_wall_panels"], 0)

    def test_exterior_roof_binding_uses_its_source_construction_beside_an_interior_ceiling(self):
        lower = Room.from_box("Lower", 5, 4, 3)
        upper = Room.from_box("Upper", 2.5, 4, 3, origin=Point3D(0, 0, 3))
        original = next(face for face in lower.faces if str(face.type) == "RoofCeiling")
        inner = Face("InnerCeiling", Face3D(tuple(Point3D(x, y, 3) for x, y in ((0, 0), (2.5, 0), (2.5, 4), (0, 4)))))
        outer = Face("OuterRoof", Face3D(tuple(Point3D(x, y, 3) for x, y in ((2.5, 0), (5, 0), (5, 4), (2.5, 4)))))
        upper_floor = next(face for face in upper.faces if str(face.type) == "Floor")
        inner.boundary_condition = Surface((upper_floor.identifier, upper.identifier))
        upper_floor.boundary_condition = Surface((inner.identifier, lower.identifier))
        lower = Room(lower.identifier, [face.duplicate() for face in lower.faces if face is not original] + [inner, outer])
        lower.story, upper.story = "Lower", "Upper"
        model = Model("MixedRoof", [lower, upper])
        drawing, report = build(model)
        validate(model, drawing.bytes(), report)
        energy = report["source_energy"]
        native = next(row for row in report["rooms"] if row["identifier"] == lower.identifier)
        self.assertEqual(energy["binding_conflicts"], [])
        self.assertEqual(native["construction_ids"]["BECS_ROOM_ROOF_ID"], energy["assignments"][outer.identifier])
        self.assertNotEqual(energy["assignments"][inner.identifier], energy["assignments"][outer.identifier])
        self.assertTrue(inspect_export(report)["checks"]["source_thermal_bindings"])
        energy["binding_conflicts"].append({"property": "BECS_ROOM_ROOF_ID"})
        self.assertFalse(inspect_export(report)["checks"]["source_thermal_bindings"])

    def test_below_datum_exterior_floor_keeps_source_layers_in_native_ground_roles(self):
        room = Room.from_box("BelowDatum", 5, 4, 3, origin=Point3D(0, 0, -0.55))
        floor = next(face for face in room.faces if str(face.type) == "Floor")
        floor.boundary_condition = Outdoors()
        model = Model("BelowDatum", [room])
        original = model.to_dict()
        drawing, report = build(model)
        validate(model, drawing.bytes(), report)
        native = report["rooms"][0]
        values = {row["id"]: row for row in report["source_energy"]["constructions"]}
        source = values[report["source_energy"]["assignments"][floor.identifier]]
        for key in ("BECS_ROOM_EX_GROUND_ID", "BECS_ROOM_INSIDE_GROUND_ID"):
            role = values[native["construction_ids"][key]]
            self.assertEqual(role["layers"], source["layers"])
            self.assertEqual(role["r_value"], source["r_value"])
        self.assertEqual(native["native_ground_classification"]["source_boundaries"][floor.identifier], "Outdoors")
        self.assertEqual(model.to_dict(), original)


if __name__ == "__main__":
    unittest.main()
