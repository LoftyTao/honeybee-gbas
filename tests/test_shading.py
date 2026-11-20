from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

from honeybee.model import Model
from honeybee.room import Room
from honeybee.shade import Shade
from honeybee.shademesh import ShadeMesh
from ladybug_geometry.geometry3d import Face3D, Mesh3D, Point3D

from honeybee_gbas import model as hb
from honeybee_gbas.dwg import build
from honeybee_gbas.dwg.records import new_drawing
from honeybee_gbas.dwg.surfaces import compare_assembled_shades
from honeybee_gbas.dwg.validation import compare_source, inspect_native, validate


def source_model():
    room = Room.from_box("ElevatedRoom", 5, 4, 3, origin=Point3D(-10, 20, 2.5))
    room.story = "Occupied"
    model = Model("EditableShades", [room])
    outer = tuple(Point3D(x, y, 6 + x * 0.1) for x, y in ((-12, 18), (-3, 18), (-3, 25), (-12, 25)))
    hole = tuple(Point3D(x, y, 6 + x * 0.1) for x, y in ((-9, 21), (-7, 21), (-7, 23), (-9, 23)))
    model.add_shade(Shade("PerforatedShade", Face3D(outer, holes=(hole,))))
    mesh = Mesh3D(tuple(Point3D(*point) for point in ((-15, 15, 4), (-13, 15, 4), (-13, 17, 5), (-15, 17, 5))), ((0, 1, 2, 3),))
    model.add_shade_mesh(ShadeMesh("EnvironmentMesh", mesh))
    return model


class OfflineShadeTest(unittest.TestCase):
    def test_host_shade_comparison_uses_handles_and_world_geometry(self):
        model = source_model()
        drawing, manifest = build(model)
        decoded = inspect_native(drawing.bytes(), manifest, include_geometry=True)["geometry"]["polyfaces"]
        floor = manifest["site_layout"]
        meshes, handles = [], []
        for record in manifest["shading"]["records"]:
            for handle in record["handles"]:
                faces = []
                for triangle in decoded[handle]["triangles_mm"]:
                    points = [((point[0] - floor["frame"]["alignment_mm"][0]) / 1000, (point[1] - floor["frame"]["alignment_mm"][1]) / 1000, (point[2] + floor["reference_elevation_mm"]) / 1000) for point in triangle]
                    faces.append([*points, points[-1]])
                meshes.append(faces)
                handles.append(format(handle, "X"))
        assembly = {"status": "completed", "result": {"assembled": True, "floors": [], "site_shading": {"story": floor["story"], "shade_objects": len(meshes), "shade_faces": meshes, "shade_handles": handles}}}
        self.assertTrue(compare_assembled_shades(model, manifest, assembly)["passed"])
        moved = copy.deepcopy(assembly)
        moved["result"]["site_shading"]["shade_faces"][0][0][0] = (0, 0, 0)
        self.assertFalse(compare_assembled_shades(model, manifest, moved)["passed"])
        changed = copy.deepcopy(assembly)
        changed["result"]["site_shading"]["shade_handles"][0] = "DEADBEEF"
        self.assertFalse(compare_assembled_shades(model, manifest, changed)["passed"])
        self.assertTrue(compare_assembled_shades(model, manifest, changed)["issues"])

    def test_shades_follow_source_hierarchy_and_orphans_use_the_site_frame(self):
        lower = Room.from_box("Lower", 5, 4, 3, origin=Point3D(0, 0, 2.5))
        lower.story = "Lower"
        upper = Room.from_box("Upper", 5, 4, 3, origin=Point3D(0, 0, 5.5))
        upper.story = "Upper"
        face = next(face for face in upper.faces if face.normal.y < -0.5)
        face.aperture_by_width_height(2, 1, 1)
        face.apertures[0].overhang(0.5)
        room_shade = Shade.from_vertices("RoomShade", ((0, 4, 8), (5, 4, 8), (5, 4.5, 8), (0, 4.5, 8)))
        upper.add_outdoor_shade(room_shade)
        model = Model("ShadeOwners", [lower, upper])
        orphan = Shade.from_vertices("DetachedShade", ((50, 50, 10), (55, 50, 10), (55, 55, 10), (50, 55, 10)))
        model.add_shade(orphan)
        drawing, manifest = build(model)
        validate(model, drawing.bytes(), manifest)
        rooms = {row["identifier"]: row for row in manifest["rooms"]}
        rows = {row["identifier"]: row for row in manifest["shading"]["records"]}
        self.assertEqual(rows[room_shade.identifier]["story"], rooms[upper.identifier]["story"])
        self.assertEqual(rows[face.apertures[0].outdoor_shades[0].identifier]["story"], rooms[upper.identifier]["story"])
        self.assertEqual(rows[orphan.identifier]["story"], "Site")
        self.assertEqual(rows[orphan.identifier]["routing"], "site")
        self.assertEqual(manifest["site_layout"]["floor_number"], 0)
        self.assertEqual(manifest["site_layout"]["reference_elevation_mm"], 0)
        self.assertEqual(manifest["ground_frame"]["height_mm"], 2500)
        self.assertGreater(manifest["site_layout"]["frame"]["width_mm"], 2 * manifest["floors"][0]["frame"]["width_mm"])
        self.assertGreater(manifest["roof_layout"]["frame_bounds_mm"][0], manifest["site_layout"]["frame_bounds_mm"][2])
        self.assertEqual(manifest["source_geometry_validation"]["summary"]["passed_shades"], 3)
        changed = copy.deepcopy(manifest)
        next(row for row in changed["shading"]["records"] if row["identifier"] == room_shade.identifier)["routing"] = "site"
        readback = inspect_native(drawing.bytes(), manifest, include_geometry=True)
        self.assertFalse(compare_source(model, changed, readback)["shading"]["passed"])
        self.assertIs(hb.owner_room(room_shade), upper)
        self.assertIsNone(hb.owner_room(orphan))

    def test_polyface_codec_preserves_vertices_indices_and_ownership(self):
        drawing, codes = new_drawing()
        triangles = [((0, 0, 1), (2, 0, 1), (2, 3, 2)), ((0, 0, 1), (2, 3, 2), (0, 3, 2))]
        entity = drawing.add_polyface(triangles, xdata=[(1000, "Mesh")])
        result = inspect_native(drawing.bytes(), include_geometry=True)
        mesh = result["geometry"]["polyfaces"][entity.handle]
        self.assertEqual(mesh["vertex_count"], 4)
        self.assertEqual(mesh["face_count"], 2)
        self.assertEqual(mesh["triangles_mm"], triangles)
        self.assertEqual(mesh["xdata"], [(1000, "Mesh")])
        self.assertEqual(len(drawing.entities), 1)
        self.assertTrue(all(child.owner == entity.handle for child in drawing.owned_entities[entity.handle]))

    def test_perforated_tilted_shades_and_meshes_keep_absolute_geometry(self):
        model = source_model()
        original = model.to_dict()
        drawing, manifest = build(model)
        validate(model, drawing.bytes(), manifest)
        result = manifest["source_geometry_validation"]["shading"]
        self.assertTrue(result["passed"])
        self.assertEqual(result["source_objects"], 2)
        self.assertEqual(manifest["shading"]["native_shades"], 2)
        self.assertEqual(manifest["shading"]["environment_shades"], 2)
        self.assertTrue(all(row["layer"] == "日-建筑" for row in manifest["shading"]["records"]))
        self.assertAlmostEqual(manifest["shading"]["records"][0]["area_m2"], model.shades[0].area)
        self.assertEqual(model.to_dict(), original)

    def test_attached_and_indoor_shades_have_explicit_geometry_and_semantics(self):
        room = Room.from_box("Room", 5, 4, 3)
        room.story = "Occupied"
        wall = next(face for face in room.faces if face.normal.y < -0.5)
        wall.aperture_by_width_height(2, 1, 1)
        aperture = wall.apertures[0]
        aperture.overhang(0.5)
        indoor = Shade.from_vertices("IndoorShade", ((1, 0.1, 2.2), (3, 0.1, 2.2), (3, 0.6, 2.2), (1, 0.6, 2.2)))
        aperture.add_indoor_shade(indoor)
        model = Model("AttachedShades", [room])
        drawing, manifest = build(model)
        validate(model, drawing.bytes(), manifest)
        self.assertEqual(manifest["source_geometry_validation"]["summary"]["passed_shades"], 2)
        self.assertEqual(manifest["shading"]["environment_shades"], 1)
        self.assertEqual(manifest["shading"]["indoor_shades"], 1)
        record = next(row for row in manifest["shading"]["records"] if row["identifier"] == indoor.identifier)
        self.assertEqual(record["source_parent"], aperture.identifier)
        self.assertEqual(record["layer"], "HB-Indoor-Shade")
        self.assertEqual(record["calculation_status"], "indoor_shading_semantics_pending")

    def test_decoded_shade_translation_and_missing_mesh_are_detected(self):
        model = source_model()
        drawing, manifest = build(model)
        readback = inspect_native(drawing.bytes(), manifest, include_geometry=True)
        handle = manifest["shading"]["records"][0]["handle"]
        changed = copy.deepcopy(readback)
        mesh = changed["geometry"]["polyfaces"][handle]
        mesh["triangles_mm"] = [tuple((point[0], point[1], point[2] + 100) for point in triangle) for triangle in mesh["triangles_mm"]]
        self.assertFalse(compare_source(model, manifest, changed)["shading"]["passed"])
        changed = copy.deepcopy(readback)
        del changed["geometry"]["polyfaces"][handle]
        self.assertFalse(compare_source(model, manifest, changed)["shading"]["passed"])
        manifest["shading"]["records"] = []
        self.assertFalse(compare_source(model, manifest, readback)["shading"]["complete"])

    def test_polyface_requires_triangles_and_finite_coordinates(self):
        drawing, codes = new_drawing()
        with self.assertRaisesRegex(ValueError, "triangular"):
            drawing.add_polyface([((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            drawing.add_polyface([])
        drawing.add_polyface([((float("inf"), 0, 0), (1, 0, 0), (1, 1, 0))])
        with self.assertRaisesRegex(ValueError, "finite"):
            drawing.bytes()


if __name__ == "__main__":
    unittest.main()
