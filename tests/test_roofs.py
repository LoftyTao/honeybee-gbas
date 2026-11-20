from __future__ import annotations

import hashlib
import sys
import unittest
from collections import Counter
from math import dist
from pathlib import Path
from unittest.mock import patch

from honeybee.aperture import Aperture
from honeybee.door import Door
from honeybee.face import Face
from honeybee.model import Model
from honeybee.room import Room
from honeybee.shade import Shade
from ladybug_geometry.geometry3d import Face3D, Point3D, Polyface3D
from shapely.geometry import Polygon
from shapely.ops import unary_union

from honeybee_gbas import model as hb
from honeybee_gbas.dwg import build
from honeybee_gbas.dwg.surfaces import planar_roof_parts, roof_mesh_indices, skylight_parts
from honeybee_gbas.dwg.validation import compare_source, inspect_export, inspect_mesh_glazing, inspect_native, inspect_skylight_properties, validate


ROOT = Path(__file__).resolve().parents[1]


def build_roof(model):
    return build(model, envelope_storeys=False, primary_levels=False)


def roof_face(room):
    return next(face for face in room.faces if str(face.type) == "RoofCeiling")


def synthetic_model():
    lower = Room.from_box("Sunken", 5, 4, 2.75, origin=Point3D(10, 20, -0.55))
    lower.story = "Lower"
    regular = Room.from_box("Regular", 3, 3, 2.7, origin=Point3D(16, 20, 0))
    regular.story = "Lower"
    box = Room.from_box("Upper", 5, 4, 3, origin=Point3D(10, 20, 3))

    def slope(point):
        return Point3D(point.x, point.y, point.z + (point.x - 10) * 0.25 if point.z > 3 else point.z)

    faces = [Face(face.identifier, Face3D(tuple(slope(point) for point in face.geometry.boundary)), face.type) for face in box.faces]
    upper = Room("Upper", faces)
    upper.story = "Upper"
    roof_face(upper).add_aperture(Aperture("RoofWindow", Face3D(tuple(Point3D(x, y, 6 + (x - 10) * 0.25) for x, y in ((11, 21), (12, 21), (12, 22), (11, 22))))))
    return Model("RoofRouting", [lower, regular, upper], units="Meters")


def source_polygon(geometry, plane):
    return Polygon([tuple(plane.xyz_to_xy(point)) for point in geometry.boundary], [[tuple(plane.xyz_to_xy(point)) for point in hole] for hole in geometry.holes or ()])


def mixed_shared_ceiling():
    section = ((0, 0), (10, 0), (10, 2), (5, 3), (0, 3))
    front, back = [tuple(Point3D(x, y, z) for x, z in section) for y in (0, 4)]
    faces = [Face3D(front), Face3D(tuple(reversed(back)))]
    for i in range(len(section)):
        faces.append(Face3D((front[i - 1], back[i - 1], back[i], front[i])))
    lower = Room.from_polyface3d("Lower", Polyface3D.from_faces(faces, 1e-6))
    upper = Room.from_box("Upper", 5, 4, 3, origin=Point3D(0, 0, 3))
    lower.story, upper.story = "Lower storey", "Upper storey"
    Room.solve_adjacency([lower, upper], 1e-6)
    return Model("MixedSharedCeiling", [lower, upper])


def source_points(row, report, scale):
    roof = report["roof_layout"]
    return [Point3D((point[0] - roof["offset_mm"][0]) / scale, (point[1] - roof["offset_mm"][1]) / scale, (point[2] + roof["reference_elevation_mm"]) / scale) for point in row["boundary_mm"]]


def observed_polygons(rows, report, scale, plane):
    return [Polygon([tuple(plane.xyz_to_xy(point)) for point in source_points(row, report, scale)]) for row in rows]


def frame_bounds(frame):
    x, y = frame["center_mm"]
    return (x - frame["width_mm"] / 2, y - frame["depth_mm"] / 2, x + frame["width_mm"] / 2, y + frame["depth_mm"] / 2)


class RoofGeometryAssertions:
    def assert_roof_geometry(self, model, report):
        scale = hb.unit_scale(model.units)
        selected = {row["identifier"] for row in report["roof_selection"]["faces"] if row["policy"] == "explicit_source_roof"}
        expected = {face.identifier: (room, face) for room in model.rooms for face in room.faces if face.identifier in selected}
        roofs = [row for row in report["surfaces"] if row["kind"] == "RoofCeiling"]
        self.assertEqual({row["identifier"] for row in roofs}, set(expected))
        for identifier, (room, face) in expected.items():
            rows = [row for row in roofs if row["identifier"] == identifier]
            plane = face.geometry.plane
            polygons = observed_polygons(rows, report, scale, plane)
            expected_polygon = source_polygon(face.geometry, plane)
            if face.apertures:
                expected_polygon = expected_polygon.difference(unary_union([source_polygon(item.geometry, plane) for item in (*face.apertures, *face.doors)]))
            observed = unary_union(polygons)
            with self.subTest(face=identifier):
                self.assertLess(observed.symmetric_difference(expected_polygon).area, max(1e-9, expected_polygon.area * 1e-9))
                self.assertAlmostEqual(sum(polygon.area for polygon in polygons), observed.area, places=8)
                for row in rows:
                    self.assertEqual(row["story"], "Roof")
                    self.assertEqual(row["source_room"], room.identifier)
                    self.assertEqual(row["source_story"], room.story or "Unassigned")
                    self.assertEqual(row["source_identifiers"], [identifier])
                    self.assertEqual(row["source_face_identifier"], identifier)
                    self.assertEqual(row["holes_mm"], [])
                    self.assertTrue(row["room_bound"])
                    self.assertTrue(all(plane.distance_to_point(point) < 1e-8 for point in source_points(row, report, scale)))


class OfflineRoofRoutingTest(RoofGeometryAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = synthetic_model()
        cls.drawing, cls.report = build_roof(cls.model)

    def test_mesh_conditions_a_nearly_collinear_chain_without_losing_boundary_nodes(self):
        ring = ((0, 0), (0, 10000), (-250, 9500), (-250, 125), (-1750, -925),
                (-2250, -1275), (-3750, -2325 - 1e-8), (-4250, -2680), (-6250, -4300))
        faces = roof_mesh_indices(ring)
        shapes = [Polygon([ring[i] for i in face]) for face in faces]
        edges = Counter((a, b) for face in faces for a, b in zip(face, (*face[1:], face[0])))
        self.assertEqual({e for e, count in edges.items() if count == 1 and e[::-1] not in edges},
                         {(i, (i + 1) % len(ring)) for i in range(len(ring))})
        self.assertEqual(len(faces), len(ring) - 2)
        self.assertTrue(all(p.exterior.is_ccw and 2 * p.area > max(dist(a, b) for a, b in
                            zip(p.exterior.coords, p.exterior.coords[1:])) for p in shapes))
        self.assertAlmostEqual(Polygon(ring).symmetric_difference(unary_union(shapes)).area, 0)

    def test_mixed_roof_leaves_shared_patch_to_the_upper_physical_floor(self):
        model = mixed_shared_ceiling()
        original = model.to_dict()
        drawing, report = build(model, inclined_carriers=True)
        validate(model, drawing.bytes(), report)
        self.assertTrue(inspect_export(report)["passed"])
        self.assertFalse(report["envelope_storeys"])
        shared = report["roof_selection"]["shared_ceiling_faces"]
        self.assertEqual(len(shared), 1)
        self.assertEqual(next(iter(shared.values()))["upper_room"], "Upper")
        roofs = [r for r in report["surfaces"] if r["source_room"] == "Lower"]
        self.assertEqual(len(roofs), 1)
        self.assertEqual(roofs[0]["story"], "Upper storey")
        self.assertEqual(roofs[0]["boundary_condition"], "Outdoors")
        self.assertNotIn(roofs[0]["identifier"], shared)
        self.assertEqual(model.to_dict(), original)

    def test_flat_glazed_roof_merges_seams_while_preserving_every_cutout(self):
        from honeybee_gbas.dwg.geometry import convex_regions
        roof = roof_face(Room.from_box("Hall", 15, 15, 3))
        for x in (1, 5, 9):
            for y in (1, 5, 9):
                roof.add_aperture(Aperture(f"Sky{x}_{y}", Face3D(tuple(
                    Point3D(x + a, y + b, 3) for a, b in ((0, 0), (2, 0), (2, 2), (0, 2))))))
        plane = roof.geometry.plane
        opaque = source_polygon(roof.geometry, plane).difference(unary_union([source_polygon(a.geometry, plane) for a in roof.apertures]))
        pieces = [Polygon([tuple(plane.xyz_to_xy(v)) for v in ring]) for ring in planar_roof_parts(roof)]
        self.assertLess(len(pieces), len(convex_regions(opaque)))
        self.assertTrue(all(p.is_valid and not p.interiors for p in pieces))
        self.assertAlmostEqual(sum(p.area for p in pieces), opaque.area)
        self.assertAlmostEqual(unary_union(pieces).symmetric_difference(opaque).area, 0)

    def test_automatic_skylight_host_keeps_adjacent_turn_and_straight_ceiling_automatic(self):
        lower = Face3D(tuple(Point3D(x, y, 0) for x, y in ((0, 0), (10, 0), (10, 2), (4, 2), (4, 6), (0, 6))))
        atrium = Room.from_polyface3d("Atrium", Polyface3D.from_offset_face(lower, 3))
        corner = Room.from_box("Corner", 6, 4, 3, origin=Point3D(4, 2, 0))
        straight = Room.from_box("Straight", 4, 2, 3, origin=Point3D(0, 6, 0))
        model = Model("RoofCorner", [atrium, corner, straight])
        Room.solve_adjacency(model.rooms, 1e-6)
        for cut in (False, True):
            with self.subTest(cut=cut):
                if cut:
                    roof_face(atrium).add_aperture(Aperture("Sky", Face3D(tuple(
                        Point3D(x, y, 3) for x, y in ((1, 1), (2, 1), (2, 2), (1, 2))))))
                drawing, report = build(model, inclined_carriers=True)
                decisions = {r["source_room"]: r for r in report["roof_selection"]["faces"]}
                self.assertEqual(decisions["Straight"]["policy"], "native_room_ceiling")
                self.assertEqual(decisions["Corner"]["policy"], "native_room_ceiling")
                if cut:
                    self.assertEqual(decisions["Atrium"]["policy"], "native_room_ceiling")
                    self.assertFalse(report["surfaces"])
                    validate(model, drawing.bytes(), report)
                    self.assertTrue(report["native_skylight_properties"]["definitions"]["passed"])

    def test_shared_datum_has_a_millimetre_limit_in_source_units(self):
        for rise_mm in (0.8, 1.2):
            with self.subTest(rise_mm=rise_mm):
                rooms = []
                for box in (Room.from_box("Lower", 5, 4, 3), Room.from_box("Upper", 5, 4, 3, origin=Point3D(0, 0, 3))):
                    def transform(p):
                        return Point3D(p.x, p.y, p.z + p.x * rise_mm / 5000 if p.z == 3 else p.z)
                    faces = [Face(f.identifier, Face3D(tuple(transform(p) for p in f.geometry.boundary)), f.type) for f in box.faces]
                    room = Room(box.identifier, faces)
                    room.story = box.identifier
                    rooms.append(room)
                Room.solve_adjacency(rooms, 1e-7)
                model = Model("SharedDatum", rooms)
                model.convert_to_units("Feet")
                drawing, report = build(model, inclined_carriers=True)
                validate(model, drawing.bytes(), report)
                candidates = report["roof_selection"]["shared_ceiling_faces"]
                if rise_mm < 1:
                    self.assertAlmostEqual(candidates["Lower_Top"]["maximum_datum_deviation_mm"], rise_mm)
                    self.assertTrue(inspect_export(report)["passed"])
                else:
                    self.assertEqual(candidates, {})
                    self.assertTrue(report["floor_meshes"])
                    self.assertFalse(inspect_export(report)["checks"]["native_profiled_floor_boundary_conditions"])

    def test_profiled_outdoor_floor_meshes_keep_all_slope_and_landing_faces(self):
        section = ((0, 3), (2, 3), (4, 4), (6, 4), (6, 8), (0, 8))
        front, back = [tuple(Point3D(x, y, z) for x, z in section) for y in (0, 5)]
        faces = [Face3D(front), Face3D(tuple(reversed(back)))]
        faces.extend(Face3D((front[i - 1], back[i - 1], back[i], front[i])) for i in range(len(section)))
        room = Room.from_polyface3d("Stepped", Polyface3D.from_faces(faces, 1e-6))
        model = Model("RaisedDeck", [room])
        model.convert_to_units("Feet")
        before = model.to_dict()
        drawing, report = build(model, inclined_carriers=True)
        data = drawing.bytes()
        validate(model, data, report)
        self.assertTrue(inspect_export(report)["passed"])
        floors = {f.identifier for f in room.faces if str(f.type) == "Floor"}
        self.assertEqual(len(floors), 3)
        self.assertEqual({row["identifier"] for row in report["floor_meshes"]}, floors)
        records = [r for r in report["source_geometry_validation"]["faces"] if r["identifier"] in floors]
        self.assertTrue(all(r["passed"] and r["outward_triangles"] for r in records))
        self.assertTrue(all(abs(r["native_triangle_area_m2"] - r["source_area_m2"]) < 1e-8 for r in records))
        self.assertTrue(all(row["policy"] == "explicit_source_roof" for row in report["roof_selection"]["faces"]))
        self.assertEqual(model.to_dict(), before)
        decoded = inspect_native(data, report, include_geometry=True)
        owner, = report["rooms"]
        self.assertEqual({r["story"] for r in report["floor_meshes"]}, {owner["story"]})
        self.assertEqual(decoded["physical_floor_frame"]["status"], "in_source_room_storey")
        for row in report["floor_meshes"]:
            face = decoded["geometry"]["polyfaces"][row["handle"]]
            self.assertEqual(face["layer"], "建-屋顶")
            self.assertEqual(face["application_xdata"]["BECS_FACE_USAGE"], [(1071, 8)])
            self.assertEqual(face["property_records"]["BECS_ROOF_SPACE_CODE"][-1], (301, owner["native_code"]))
        mesh = decoded["geometry"]["polyfaces"][report["floor_meshes"][0]["handle"]]["triangles_mm"]
        mesh.append(mesh[0])
        self.assertGreater(compare_source(model, report, decoded)["summary"]["native_face_review"], 0)

    def test_roof_follows_ground_and_preserves_occupied_series(self):
        report = self.report
        occupied = report["floors"]
        self.assertEqual({row["role"] for row in occupied}, {"occupied"})
        self.assertEqual([(row["floor_number"], row["series"]) for row in occupied], [(1, ""), (1, "A"), (2, "")])
        self.assertEqual(report["roof_frame"]["floor_number"], 3)
        self.assertEqual(report["roof_frame"]["series"], "")
        self.assertEqual(report["roof_frame"]["height_mm"], 0.0)
        self.assertEqual(report["roof_layout"]["height_mm"], 0.0)
        self.assertEqual(report["ground_frame"]["height_mm"], -550.0)
        ground = frame_bounds(report["ground_frame"])
        roof = frame_bounds(report["roof_frame"])
        self.assertAlmostEqual(roof[0] - ground[2], 5000.0)
        self.assertEqual(report["counts"]["floor_frames"], len(occupied) + 2)
        for floor in occupied:
            self.assertLess(frame_bounds(floor["frame"])[2], roof[0])

    def test_roof_and_skylight_share_absolute_source_coordinates(self):
        report = self.report
        scale = hb.unit_scale(self.model.units)
        anchor = max(room.max.z for room in self.model.rooms) * scale
        roof = report["roof_layout"]
        self.assertEqual(roof["reference_elevation_mm"], anchor)
        self.assertEqual(roof["source_elevation_mm"], anchor)
        self.assertEqual(roof["native_elevation_mm"], anchor)
        primary = sorted((row for row in report["floors"] if row["series"] == ""), key=lambda row: row["floor_number"])
        self.assertAlmostEqual(report["ground_frame"]["height_mm"] + sum(row["height_mm"] for row in primary), anchor)
        self.assert_roof_geometry(self.model, report)
        face = roof_face(self.model.rooms[-1])
        aperture = face.apertures[0]
        skylights = [row for row in report["surfaces"] if row["kind"] == "Skylight"]
        self.assertEqual(len(skylights), 1)
        sky = skylights[0]
        self.assertEqual(sky["host_identifier"], face.identifier)
        self.assertEqual(sky["identifier"], aperture.identifier)
        self.assertEqual(sky["source_room"], self.model.rooms[-1].identifier)
        self.assertEqual(sky["story"], "Roof")
        self.assertEqual(sky["board_type"], 3)
        self.assertTrue(sky["room_bound"])
        decoded = inspect_native(self.drawing.bytes(), report, include_geometry=True)["geometry"]
        self.assertFalse(decoded["entities"][sky["handle"]]["layer_visible"])
        definition = next(v for v in report["defined_skylights"] if v["identifier"] == aperture.identifier)
        self.assertTrue(decoded["polylines"][definition["handle"]]["layer_visible"])
        self.assertTrue(all(row["room_bound"] for row in report["surfaces"] if row["identifier"] == face.identifier))
        self.assertEqual(next(v["roof_binding"] for v in report["roof_selection"]["faces"] if v["identifier"] == face.identifier), "explicit_room")
        actual = observed_polygons(skylights, report, scale, face.geometry.plane)[0]
        expected = source_polygon(aperture.geometry, face.geometry.plane)
        self.assertLess(actual.symmetric_difference(expected).area, 1e-9)
        roofs = [row for row in report["surfaces"] if row["kind"] == "RoofCeiling"]
        roof_union = unary_union(observed_polygons(roofs, report, scale, face.geometry.plane))
        self.assertAlmostEqual(roof_union.intersection(expected).area, 0, places=9)
        self.assertLess(roof_union.union(actual).symmetric_difference(source_polygon(face.geometry, face.geometry.plane)).area, 1e-9)

    def test_occupied_frames_retain_their_own_reference_transforms(self):
        floors = {row["story"]: row for row in self.report["floors"]}
        rooms = {room.identifier: room for room in self.model.rooms}
        for row in self.report["rooms"]:
            expected = hb.room_floor_elevation(rooms[row["identifier"]]) * hb.unit_scale(self.model.units)
            self.assertAlmostEqual(row["position_mm"][2] + floors[row["story"]]["reference_elevation_mm"], expected)
        self.assertEqual({row["story"] for row in self.report["surfaces"]}, {"Roof"})
        self.assertEqual({row["kind"] for row in self.report["surfaces"]}, {"RoofCeiling", "Skylight"})

    def test_extents_include_negative_roof_z_and_roof_frame(self):
        low, high = self.drawing.extents
        self.assertLess(low[2], 0)
        roof_bounds = frame_bounds(self.report["roof_frame"])
        self.assertLessEqual(low[0], roof_bounds[0])
        self.assertGreaterEqual(high[0], roof_bounds[2])
        for row in self.report["surfaces"]:
            for point in row["boundary_mm"]:
                self.assertTrue(all(low[index] - 1e-6 <= point[index] <= high[index] + 1e-6 for index in range(3)))

    def test_geometry_acceptance_requires_full_host_geometry(self):
        acceptance = self.report["geometry_acceptance"]
        self.assertEqual(acceptance["status"], "host_geometry_pending")
        self.assertTrue({"closed_room_shells", "absolute_elevation", "source_floor_and_roof_coverage", "skylight_geometry_and_host_relations", "per_room_volume"}.issubset(acceptance["mandatory"]))
        self.assertEqual(acceptance["absolute_tolerance_mm"], 1.0)

    def test_highest_top_series_keeps_one_roof_number_and_anchor(self):
        model = synthetic_model()
        higher = Room.from_box("UpperRaised", 3, 3, 5, origin=Point3D(16, 25, 3.5))
        higher.story = "Upper"
        model.add_room(higher)
        _, report = build_roof(model)
        self.assertEqual(len(report["floors"]), 4)
        self.assertEqual(report["roof_frame"]["floor_number"], 3)
        self.assertEqual(report["roof_layout"]["reference_elevation_mm"], 8500.0)
        top = next(row for row in report["floors"] if row["floor_number"] == 2 and row["series"] == "")
        self.assertEqual(top["frame"]["height_mm"], 6300.0)
        self.assertEqual(top["height_mm"], 6300.0)
        self.assertEqual(top["reference_elevation_mm"], 2200.0)
        self.assertEqual(report["counts"]["native_roof_faces"], 1)
        self.assertEqual(next(row for row in report["roof_selection"]["faces"] if row["source_room"] == "UpperRaised")["policy"], "native_room_ceiling")
        self.assert_roof_geometry(model, report)

    def test_roof_terminal_frame_preserves_source_building_height(self):
        primary = [row for row in self.report["floors"] if row["series"] == ""]
        native_height = sum(row["height_mm"] for row in primary) + self.report["roof_frame"]["height_mm"]
        scale = hb.unit_scale(self.model.units)
        source_height = (max(room.max.z for room in self.model.rooms) - min(hb.room_floor_elevation(room) for room in self.model.rooms)) * scale
        self.assertAlmostEqual(native_height, source_height)

    def test_source_holes_are_exact_hole_free_native_pieces(self):
        boundary = tuple(Point3D(x, y, 3) for x, y in ((10, 20), (20, 20), (20, 30), (10, 30)))
        hole = tuple(Point3D(x, y, 3) for x, y in ((13, 23), (17, 23), (17, 27), (13, 27)))
        polyface = Polyface3D.from_offset_face(Face3D(boundary, holes=(hole,)), 3)
        room = Room.from_polyface3d("Courtyard", polyface)
        room.story = "Top"
        model = Model("RoofHole", [room], units="Meters")
        _, report = build_roof(model)
        face = roof_face(room)
        self.assertTrue(face.geometry.holes)
        self.assertGreater(len(planar_roof_parts(face)), 1)
        self.assert_roof_geometry(model, report)
        self.assertEqual(report["counts"]["native_roof_faces"], 1)
        self.assertTrue(all(row["holes_mm"] == [] for row in report["surfaces"]))

    def test_real_roof_hole_and_skylight_cutout_keep_distinct_geometry(self):
        boundary = tuple(Point3D(x, y, 3) for x, y in ((10, 20), (20, 20), (20, 30), (10, 30)))
        hole = tuple(Point3D(x, y, 3) for x, y in ((13, 23), (17, 23), (17, 27), (13, 27)))
        room = Room.from_polyface3d("CourtyardSkylight", Polyface3D.from_offset_face(Face3D(boundary, holes=(hole,)), 3))
        room.story = "Top"
        face = roof_face(room)
        aperture = Aperture("CourtyardWindow", Face3D(tuple(Point3D(x, y, 3) for x, y in ((11, 21), (12, 21), (12, 22), (11, 22)))))
        face.add_aperture(aperture)
        model = Model("RoofHoleAndSkylight", [room])
        original = model.to_dict()
        drawing, report = build_roof(model)
        self.assert_roof_geometry(model, report)
        roofs = [row for row in report["surfaces"] if row["kind"] == "RoofCeiling"]
        actual = unary_union(observed_polygons(roofs, report, 1000, face.geometry.plane))
        self.assertAlmostEqual(actual.area, 83)
        self.assertAlmostEqual(actual.intersection(source_polygon(aperture.geometry, face.geometry.plane)).area, 0)
        self.assertEqual(len(roofs), len(planar_roof_parts(face)))
        self.assertEqual(inspect_native(drawing.bytes(), report)["native_skylights"], 1)
        self.assertEqual(model.to_dict(), original)

    def test_roof_door_and_skylight_cutouts_keep_distinct_source_areas(self):
        room = Room.from_box("RoofDoor", 5, 4, 3)
        face = roof_face(room)
        face.add_aperture(Aperture("RoofWindow", Face3D(tuple(Point3D(x, y, 3) for x, y in ((1, 1), (2, 1), (2, 2), (1, 2))))))
        door = Door("RoofHatch", Face3D(tuple(Point3D(x, y, 3) for x, y in ((3, 1), (4, 1), (4, 3), (3, 3)))))
        face.add_door(door)
        parts = [source_polygon(Face3D(part), face.geometry.plane) for part in planar_roof_parts(face)]
        actual = unary_union(parts)
        self.assertAlmostEqual(actual.area, 17)
        self.assertAlmostEqual(actual.intersection(source_polygon(face.apertures[0].geometry, face.geometry.plane)).area, 0)
        self.assertLess(actual.intersection(source_polygon(door.geometry, face.geometry.plane)).area, 1e-9)

    def test_build_preserves_source_and_binary_output_is_deterministic(self):
        model = synthetic_model()
        source = model.to_dict()
        drawing, report = build_roof(model)
        second, _ = build_roof(model)
        self.assertEqual(model.to_dict(), source)
        self.assertEqual(hashlib.sha256(drawing.bytes()).digest(), hashlib.sha256(second.bytes()).digest())
        inspection = inspect_native(drawing.bytes(), report)
        self.assertEqual(inspection["class_counts"]["SWR_FLOOR_FRAME"], 5)
        self.assertEqual(list(inspection["floor_series"].values()), ["A"])

    def test_skylights_keep_visible_native_identity_and_source_thermal_properties(self):
        model = synthetic_model()
        drawing, report = build_roof(model)
        validate(model, drawing.bytes(), report)
        decoded = inspect_native(drawing.bytes(), report, include_geometry=True)
        source = roof_face(model.rooms[-1]).apertures[0]
        sky = next(row for row in report["surfaces"] if row["kind"] == "Skylight")
        native = decoded["geometry"]["entities"][sky["handle"]]
        self.assertEqual(native["type"], "POLYLINE_PFACE")
        self.assertEqual([len(p) for p in native["polygons_mm"]], [4])
        self.assertEqual(native["face_usage"], 2)
        self.assertEqual(native["board_type"], 3)
        self.assertEqual(native["source_room"], report["rooms"][-1]["native_code"])
        self.assertAlmostEqual(sky["source_thickness_mm"], source.properties.energy.construction.thickness * 1000)
        self.assertIn((1000, source.display_name), native["xdata"])
        self.assertEqual(native["application_xdata"]["_ARCH_SLAB_TEXT"], [(1000, "天窗")])
        self.assertTrue(inspect_export(report)["required_checks"]["native_skylight_properties"])
        native["board_type"] = 0
        self.assertFalse(inspect_skylight_properties(model, report, decoded)["passed"])
        native["board_type"] = 3
        native["face_usage"] = 1
        self.assertFalse(inspect_skylight_properties(model, report, decoded)["passed"])
        native["face_usage"] = 2
        native["property_records"].pop("THSWARE_STRUCTLIB_XREC")
        self.assertFalse(inspect_skylight_properties(model, report, decoded)["passed"])

    def test_skylight_release_gate_separates_full_source_names_from_compact_native_labels(self):
        model = synthetic_model()
        source = roof_face(model.rooms[-1]).apertures[0]
        source.display_name = "SourceRoofWindow" * 10
        original = model.to_dict()
        drawing, report = build_roof(model)
        validate(model, drawing.bytes(), report)
        self.assertTrue(inspect_export(report)["required_checks"]["native_skylight_properties"])
        decoded = inspect_native(drawing.bytes(), report, include_geometry=True)
        sky = next(row for row in report["surfaces"] if row["kind"] == "Skylight")
        native = decoded["geometry"]["entities"][sky["handle"]]
        self.assertIn((1000, source.display_name), native["application_xdata"]["HB_BESI"])
        self.assertEqual(model.to_dict(), original)
        native["application_xdata"]["_ARCH_SLAB_TEXT"] = [(1000, source.display_name)]
        review = inspect_skylight_properties(model, report, decoded)
        self.assertFalse(review["passed"])
        self.assertFalse(review["records"][0]["checks"]["native_compact_label"])

    def test_multiple_inclined_skylights_keep_disjoint_opaque_host_pieces(self):
        model = synthetic_model()
        face = roof_face(model.rooms[-1])
        face.add_aperture(Aperture("SecondRoofWindow", Face3D(tuple(Point3D(x, y, 6 + (x - 10) * 0.25) for x, y in ((13, 21), (14, 21), (14, 22), (13, 22))))))
        drawing, report = build_roof(model)
        validate(model, drawing.bytes(), report)
        self.assertEqual(len(report["native_skylight_properties"]["records"]), 2)
        self.assertTrue(report["native_skylight_properties"]["passed"])
        self.assert_roof_geometry(model, report)

    def test_fully_glazed_inclined_roof_uses_its_native_skylight_as_the_envelope(self):
        model = synthetic_model()
        face = roof_face(model.rooms[-1])
        face.remove_apertures()
        face.add_aperture(Aperture("GlazedRoof", face.geometry))
        drawing, report = build_roof(model)
        validate(model, drawing.bytes(), report)
        self.assertTrue(inspect_export(report)["passed"])
        self.assertEqual([row for row in report["surfaces"] if row["kind"] == "RoofCeiling" and row["identifier"] == face.identifier], [])
        self.assertEqual(len(report["skylights"]), 1)

    def test_compact_inclined_skylights_keep_the_opaque_host_cutout_gate(self):
        model = synthetic_model()
        original = planar_roof_parts
        with patch("honeybee_gbas.dwg.surfaces.planar_roof_parts", side_effect=lambda face: (face.geometry.boundary,) if face.apertures else original(face)):
            drawing, report = build_roof(model)
        validate(model, drawing.bytes(), report)
        self.assertFalse(inspect_export(report)["required_checks"]["native_skylight_properties"])
        review = next(row for row in report["native_skylight_properties"]["records"] if not row["passed"])
        self.assertTrue(review["checks"]["native_compact_label"])
        self.assertFalse(review["checks"]["opaque_host_cutout"])

    def test_attached_shades_preserve_enclosure_height_and_roof_anchor(self):
        room = Room.from_box("ShadedRoom", 5, 4, 3, origin=Point3D(11, 12, 5))
        roof = roof_face(room)
        roof.add_outdoor_shade(Shade.from_vertices("UpperShade", [(11, 12, 12), (16, 12, 12), (16, 16, 12), (11, 16, 12)]))
        next(face for face in room.faces if str(face.type) == "Wall").add_outdoor_shade(Shade.from_vertices("LowerShade", [(11, 12, 2), (16, 12, 2), (16, 16, 2), (11, 16, 2)]))
        roof.add_aperture(Aperture("RoofWindow", Face3D(tuple(Point3D(x, y, 8) for x, y in ((12, 13), (13, 13), (13, 14), (12, 14))))))
        model = Model("ShadedHeight", [room])
        original = model.to_dict()
        self.assertEqual((room.min.z, room.max.z), (2, 12))
        self.assertEqual(hb.room_elevation_bounds(room), (5, 8))
        drawing, report = build_roof(model)
        validate(model, drawing.bytes(), report)
        self.assertEqual(report["rooms"][0]["height_mm"], 3000)
        self.assertEqual(report["roof_layout"]["reference_elevation_mm"], 8000)
        self.assertEqual(report["counts"]["native_shades"], 2)
        self.assertTrue(inspect_export(report)["passed"])
        self.assertEqual(model.to_dict(), original)

    def test_elevated_inclined_wall_completes_native_roof_projection(self):
        def face(identifier, points, kind=None):
            return Face(identifier, Face3D(tuple(Point3D(*point) for point in points)), kind)
        from honeybee.facetype import Floor, RoofCeiling, Wall
        section = ((0, 0), (12, 0), (12, 3.6), (4, 5), (3, 7), (0, 7))
        room = Room("RaisedRoof", [
            face("Ground", ((0, 0, 0), (0, 12, 0), (6, 12, 0), (6, 0, 0)), Floor()),
            face("Front", ((0, 0, 0), (6, 0, 0), (6, 0, 7), (0, 0, 7)), Wall()),
            face("Back", ((6, 12, 0), (0, 12, 0), (0, 12, 3.6), (6, 12, 3.6)), Wall()),
            face("West", tuple((0, y, z) for y, z in reversed(section)), Wall()),
            face("East", tuple((6, y, z) for y, z in section), Wall()),
            face("RaisedRoof", ((0, 0, 7), (6, 0, 7), (6, 3, 7), (0, 3, 7)), RoofCeiling()),
            face("UpperWall", ((0, 3, 7), (6, 3, 7), (6, 4, 5), (0, 4, 5)), Wall()),
            face("LowerRoof", ((0, 4, 5), (6, 4, 5), (6, 12, 3.6), (0, 12, 3.6)), RoofCeiling()),
        ])
        raised = next(item for item in room.faces if item.identifier == "RaisedRoof")
        raised.add_aperture(Aperture("RoofWindow", Face3D(tuple(Point3D(x, y, 7) for x, y in ((1, 1), (2, 1), (2, 2), (1, 2))))))
        model = Model("ElevatedInclinedWall", [room])
        original = model.to_dict()
        drawing, report = build(model, inclined_carriers=True)
        validate(model, drawing.bytes(), report)
        self.assertTrue(inspect_export(report)["passed"])
        self.assertFalse(any(row["identifier"] == "UpperWall" for row in report["walls"]))
        carriers = [row for row in report["surfaces"] if row["identifier"] == "UpperWall"]
        self.assertTrue(carriers)
        self.assertTrue(all(row["source_kind"] == "Wall" and row["native_carrier"] == "inclined_roof" and row["room_bound"] for row in carriers))
        residual = next(row for row in report["native_wall_completion"]["source_residuals"] if row["identifier"] == "UpperWall")
        self.assertEqual(residual["reason"], "elevated_inclined_wall_requires_native_roof_projection")
        projections = [Polygon([(point.x, point.y) for point in source_points(row, report, 1000)]) for row in report["surfaces"] if row["kind"] in {"RoofCeiling", "Skylight"} or row in carriers]
        observed = unary_union(projections)
        self.assertAlmostEqual(observed.area, 72)
        self.assertAlmostEqual(sum(polygon.area for polygon in projections), observed.area)
        self.assertEqual(model.to_dict(), original)

    def test_inclined_opaque_pieces_require_independent_native_room_binding(self):
        model = synthetic_model()
        drawing, report = build_roof(model)
        validate(model, drawing.bytes(), report)
        decoded = inspect_native(drawing.bytes(), report, include_geometry=True)
        face = roof_face(model.rooms[-1])
        host = next(row for row in report["surfaces"] if row["identifier"] == face.identifier)
        decoded["geometry"]["entities"][host["handle"]]["source_room"] = "OtherRoom"
        review = inspect_skylight_properties(model, report, decoded)
        self.assertFalse(review["passed"])
        self.assertFalse(review["records"][0]["checks"]["opaque_host_room_binding"])

    def test_bound_glazed_eave_retains_its_full_projection_and_one_piece_of_glass(self):
        box = Room.from_box("EaveRoom", 4, 4, 3)
        def raised(p):
            x = p.x * 1.25 if p.z else p.x
            return Point3D(x, p.y, p.z + (0.25 * x if p.z else 0))
        room = Room("EaveRoom", [Face(f.identifier, Face3D(tuple(raised(p) for p in f.geometry.boundary)), f.type) for f in box.faces])
        face = roof_face(room)
        face.add_aperture(Aperture("EaveGlass", Face3D(tuple(Point3D(x, y, 3 + 0.25*x)
                               for x, y in ((4.1, 1), (4.9, 1), (4.9, 3), (4.1, 3))))))
        model = Model("BoundEave", [room])
        drawing, report = build(model, inclined_carriers=True)
        data = drawing.bytes()
        validate(model, data, report)
        native = inspect_native(data, report, include_geometry=True)
        panels = [r for r in report["surfaces"] if r.get("source_face_identifier") == face.identifier]
        self.assertEqual({r["native_class"] for r in panels}, {"POLYLINE_PFACE"})
        record = next(r for r in report["source_geometry_validation"]["faces"] if r["identifier"] == face.identifier)
        self.assertTrue(record["outward_mesh_faces"] and record["passed"])
        self.assertAlmostEqual(record["mesh_sum_area_m2"], 20 * (1 + 0.25**2)**0.5)
        self.assertLess(record["mesh_overlap_area_m2"], 1e-9)
        sky, = [r for r in panels if r["kind"] == "Skylight"]
        glass = native["geometry"]["polyfaces"][sky["handle"]]
        self.assertEqual([len(p) for p in glass["polygons_mm"]], [4])
        self.assertTrue(inspect_mesh_glazing(model, report, native)["passed"])
        glass["polygons_mm"].append(glass["polygons_mm"][0])
        self.assertFalse(inspect_mesh_glazing(model, report, native)["passed"])
        glass["polygons_mm"].pop()
        glass["property_records"]["BECS_ROOF_SPACE_CODE"] = [(280, 1), (301, "other")]
        self.assertFalse(inspect_mesh_glazing(model, report, native)["passed"])

    def test_mesh_glazing_requires_independent_native_glazing_properties(self):
        model = synthetic_model()
        drawing, report = build_roof(model)
        native = inspect_native(drawing.bytes(), report, include_geometry=True)
        record, = [r for r in report["surfaces"] if r["kind"] == "Skylight"]
        props = next(p["values"] for p in native["geometry"]["properties"].values()
                     if p["values"].get("ID") == record["construction_id"] and "WinSHGC" in p["values"])
        props["WinSHGC"] *= .5
        self.assertFalse(inspect_mesh_glazing(model, report, native)["passed"])

    def test_short_edge_flat_skylight_retains_one_complete_native_definition(self):
        room = Room.from_box("ShortEdgeRoom", 5, 4, 3)
        room.story = "Top"
        face = roof_face(room)
        geometry = Face3D(tuple(Point3D(x, y, 3) for x, y in ((1, 1), (2.396658, 1), (1.00647, 2.390187), (1, 2.390187))))
        face.add_aperture(Aperture("ShortEdgeSkylight", geometry))
        model = Model("ShortEdgeRoof", [room])
        original = model.to_dict()
        drawing, report = build_roof(model)
        expected = source_polygon(geometry, geometry.plane)
        record, = report["defined_skylights"]
        decoded = inspect_native(drawing.bytes(), report, include_geometry=True)
        poly = decoded["geometry"]["polylines"][record["handle"]]
        self.assertEqual(len(poly["points"]), 4)
        self.assertTrue(poly["closed"])
        self.assertEqual(model.to_dict(), original)
        self.assertEqual((record["identifier"], record["host_identifier"], record["story"]), ("ShortEdgeSkylight", face.identifier, "Top"))
        self.assertAlmostEqual(sum(row["area_m2"] for row in report["skylights"]), expected.area, places=8)
        self.assertEqual(decoded["native_skylights"], 1)

    def test_short_edge_partition_threshold_uses_model_units(self):
        geometry = Face3D(tuple(Point3D(x, y, 3) for x, y in ((0, 0), (1400, 0), (6.47, 1390), (0, 1390))))
        self.assertEqual(len(skylight_parts(geometry, 1)), 2)
        self.assertEqual(len(skylight_parts(geometry, 1000)), 1)


class OfflineSampleRoofRoutingTest(RoofGeometryAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Model.from_file(str(ROOT / "samples/HoneybeeModel_RevitHouse/HoneybeeModel_RevitHouse.hbjson"))
        cls.drawing, cls.report = build_roof(cls.model)

    def test_sample_preserves_sloped_and_special_elevation_roofs_with_two_skylights(self):
        report = self.report
        self.assertEqual(report["roof_layout"]["source_story"], "Level 2")
        self.assertEqual(report["counts"]["native_roof_faces"], 16)
        self.assertEqual(report["counts"]["native_skylights"], 2)
        self.assertEqual(report["counts"]["source_roof_faces"], 21)
        self.assertEqual(report["counts"]["reference_openings"], 0)
        roof_surfaces = [row for row in report["surfaces"] if row["kind"] in {"RoofCeiling", "Skylight"}]
        self.assertEqual({row["story"] for row in roof_surfaces}, {"Roof"})
        self.assertEqual({row["source_story"] for row in roof_surfaces}, {"Level 1", "Level 2"})
        self.assertTrue(all(row["room_bound"] for row in report["surfaces"]))
        self.assert_roof_geometry(self.model, report)

    def test_sample_frame_roles_and_absolute_anchor(self):
        report = self.report
        anchor = max(room.max.z for room in self.model.rooms) * hb.unit_scale(self.model.units)
        self.assertAlmostEqual(report["roof_layout"]["native_elevation_mm"], anchor)
        self.assertAlmostEqual(anchor, 8913.059657002157)
        self.assertEqual(report["roof_frame"]["floor_number"], 3)
        self.assertEqual(report["counts"]["floor_frames"], 5)
        self.assertEqual(report["ground_frame"]["height_mm"], -550)
        self.assertGreater(frame_bounds(report["roof_frame"])[0], frame_bounds(report["ground_frame"])[2])
        self.assertEqual([(row["floor_number"], row["series"]) for row in report["floors"]], [(1, ""), (1, "A"), (2, "")])

    def test_sample_skylight_world_polygons_and_host_identities(self):
        report = self.report
        scale = hb.unit_scale(self.model.units)
        apertures = {item.identifier: (room, face, item) for room in self.model.rooms if room.story == "Level 2" for face in room.faces if str(face.type) == "RoofCeiling" for item in face.apertures}
        skylights = [row for row in report["surfaces"] if row["kind"] == "Skylight"]
        self.assertEqual({row["identifier"] for row in skylights}, set(apertures))
        for row in skylights:
            room, face, item = apertures[row["identifier"]]
            actual = observed_polygons([row], report, scale, face.geometry.plane)[0]
            expected = source_polygon(item.geometry, face.geometry.plane)
            self.assertLess(actual.symmetric_difference(expected).area, 1e-8)
            self.assertEqual(row["host_identifier"], face.identifier)
            self.assertEqual(row["source_room"], room.identifier)
            self.assertEqual(row["board_type"], 3)
            self.assertTrue(all(face.geometry.plane.distance_to_point(point) < 1e-8 for point in source_points(row, report, scale)))

    def test_sample_native_readback_counts_roof_once(self):
        inspection = inspect_native(self.drawing.bytes(), self.report)
        self.assertEqual(inspection["class_counts"]["SWR_FLOOR_FRAME"], 5)
        self.assertEqual(inspection["class_counts"].get("SWR_SLAB", 0), self.report["counts"]["native_slabs"])
        self.assertEqual(inspection["native_roof_meshes"], self.report["counts"]["native_roof_meshes"])
        self.assertEqual(inspection["native_skylights"], 2)


if __name__ == "__main__":
    unittest.main()
