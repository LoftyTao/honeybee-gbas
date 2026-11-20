from __future__ import annotations

import unittest

from honeybee.aperture import Aperture
from honeybee.face import Face
from honeybee.model import Model
from honeybee.room import Room
from ladybug_geometry.geometry3d import Face3D, Point3D

from honeybee_gbas.dwg import build
from honeybee_gbas.dwg.binary import BitReader, inspect_container
from honeybee_gbas.dwg.surfaces import wcs_points
from honeybee_gbas.dwg.validation import common, inspect_native


class DefinedSkylights(unittest.TestCase):
    def test_overhanging_glazing_retains_visible_definition_and_hidden_thermal_geometry(self):
        box = Room.from_box("Porch", 6, 5, 3)
        def shift(p):
            return Point3D(p.x + (2 if p.z else 0), p.y, p.z)
        room = Room("Porch", [Face(f.identifier, Face3D(tuple(shift(v) for v in f.geometry.boundary)), f.type) for f in box.faces])
        host = next(f for f in room.faces if str(f.type) == "RoofCeiling")
        host.add_aperture(Aperture("OverhangWindow", Face3D(tuple(Point3D(x, y, 3) for x, y in ((6.3, 1), (7.3, 1), (7.3, 2), (6.3, 2))))))
        drawing, report = build(Model("Overhang", [room]), inclined_carriers=True)
        row, = report["defined_skylights"]
        self.assertTrue(row["thermal_surface_handles"])
        geometry = inspect_native(drawing.bytes(), report, include_geometry=True)["geometry"]
        self.assertTrue(geometry["polylines"][row["handle"]]["layer_visible"])
        self.assertTrue(all(not geometry["entities"][handle]["layer_visible"] for handle in row["thermal_surface_handles"]))

    def test_concave_sloped_polyline_is_one_numbered_glazing_definition_in_the_room_storey(self):
        box = Room.from_box("Room", 6, 5, 4)
        def raise_roof(point):
            return Point3D(point.x, point.y, point.z + (point.x * 0.2 if point.z else 0))
        room = Room("Atrium", [Face(f.identifier, Face3D(tuple(raise_roof(p) for p in f.geometry.boundary)), f.type) for f in box.faces])
        roof = next(face for face in room.faces if str(face.type) == "RoofCeiling")
        corners = ((1, 1), (4, 1), (4, 3), (2, 3), (2, 2), (1, 2), (0.998, 1.999))
        source = Aperture("Glazing", Face3D(tuple(raise_roof(Point3D(x, y, 4)) for x, y in corners)))
        roof.add_aperture(source)
        source_model = Model("Test", [room])
        drawing, report = build(source_model)
        definition, = report["defined_skylights"]
        self.assertEqual(definition["thermal_surface_handles"], [])
        host, = [s for s in report["surfaces"] if s["identifier"] == roof.identifier]
        self.assertFalse(host["room_bound"])
        self.assertTrue(definition["projected_to_plan"])
        data = drawing.bytes()
        objects = {row["handle"]: row for row in inspect_container(data)["objects"]}
        record = objects[definition["handle"]]
        self.assertEqual(record["type_code"], 0x4D)
        apps = {}
        _, refs, _ = common(record, True, apps)
        layer = common(objects[refs[4][1]], False)[0].text()
        self.assertEqual(layer, "建-门窗-天窗")
        named_apps = {common(objects[key], False)[0].text(): values for key, values in apps.items()}
        self.assertEqual(named_apps["_LOUVER_LBL_"], [(1000, "TC001")])
        self.assertIn((1000, "Glazing"), named_apps["HB_BESI"])
        reader, pointers, _ = common(objects[refs[1][1]], False)
        size, _, _ = reader.bl(), reader.bs(), reader.rc()
        entries = dict(zip((reader.text() for _ in range(size)), pointers[2:]))
        xrecord = objects[entries["THSWARE_STRUCTLIB_XREC"][1]]
        reader = common(xrecord, False)[0]
        fields = BitReader(reader.raw(reader.bl()))
        def text():
            code, size, charset = fields.rs(), fields.rs(), fields.rc()
            self.assertEqual(charset, 39)
            return code, fields.raw(size).rstrip(b"\0").decode("gbk")
        self.assertEqual(text(), (301, ""))
        identity = report["source_energy"]["assignments"][source.identifier]
        self.assertEqual(text(), (302, str(identity)))
        self.assertEqual(fields.position, fields.limit)
        poly = inspect_native(data, include_geometry=True)["geometry"]["polylines"][definition["handle"]]
        self.assertTrue(poly["closed"])
        self.assertEqual(len(poly["points"]), len(corners))
        frame = next(row for row in report["floors"] if row["story"] == definition["story"])
        points = {(round(x - frame["offset_mm"][0], 4), round(y - frame["offset_mm"][1], 4)) for x, y, z in wcs_points(dict(poly, outline_mm=poly["points"]))}
        self.assertEqual(points, {tuple(round(value * 1000, 4) for value in tuple(point)[:2]) for point in source.geometry.boundary})
        self.assertAlmostEqual(definition["area_m2"], source.area)


if __name__ == "__main__":
    unittest.main()
