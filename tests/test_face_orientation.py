import unittest

from honeybee.facetype import face_types
from honeybee.model import Model
from honeybee.room import Room
from ladybug_geometry.geometry3d import Point3D

from honeybee_gbas.dwg import build, validation
from honeybee_gbas.dwg.orientation import orientation_model


class FaceOrientationTest(unittest.TestCase):
    def test_vertical_exterior_roof_routes_as_wall_with_source_glazing_and_construction(self):
        box = Room.from_box("Box", 5, 4, 3)
        face = next(f for f in box.faces if f.normal.x > 0.99)
        face.type = face_types.roof_ceiling
        face.apertures_by_ratio(0.25)
        glazing = face.apertures[0]
        model = Model("VerticalRoof", [box])
        source = model.to_dict()
        drawing, report = build(model, inclined_carriers=True)
        validation.validate(model, drawing.bytes(), report)
        self.assertTrue(validation.inspect_export(report)["passed"])
        self.assertEqual(model.to_dict(), source)
        self.assertEqual(report["face_orientation"]["reclassified"][face.identifier]["effective"], "Wall")
        wall = next(w for w in report["walls"] if face.identifier in w.get("source_identifiers", []))
        window = next(w for w in report["openings"] if glazing.identifier in w["source_identifiers"])
        self.assertEqual(window["wall"], wall["handle"])
        self.assertIn(window["handle"], wall["openings"])
        for row, item, role in ((wall, face, "外墙"), (window, glazing, "外窗")):
            record = next(c for c in report["source_energy"]["constructions"] if c["id"] == row["construction_id"])
            self.assertEqual((record["category"], record["identifier"]), (role, item.properties.energy.construction.identifier))

    def test_reciprocal_surface_keeps_authored_role(self):
        rooms = [Room.from_box("A", 5, 4, 3), Room.from_box("B", 5, 4, 3, origin=Point3D(5, 0, 0))]
        Room.solve_adjacency(rooms, 0.001)
        face = next(f for f in rooms[0].faces if str(f.boundary_condition) == "Surface")
        face.type = face_types.roof_ceiling
        model = Model("Shared", rooms)
        matched, audit = orientation_model(model)
        self.assertIs(matched, model)
        self.assertEqual(audit["reclassified"], {})
        self.assertEqual(face.boundary_condition.boundary_condition_objects[-1], "B")
