from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path

from honeybee.aperture import Aperture
from honeybee.door import Door
from honeybee.face import Face
from honeybee.model import Model
from honeybee.room import Room
from honeybee_energy.boundarycondition import Adiabatic
from ladybug_geometry.geometry3d import Face3D, Point3D, Polyface3D, Vector3D

from honeybee_gbas.dwg import build
from honeybee_gbas.dwg.validation import inspect_export
from honeybee_gbas.dwg.geometry import coverage, sloped_oblique_wall, source_oblique_wall, wall_points
from honeybee_gbas.dwg.surfaces import planar_polygon, wall_boundary
from honeybee_gbas.dwg.validation import compare_source, inspect_native, inspect_quality, require_export, validate
from honeybee_gbas.dwg.walls import obstacles, sloped_base_wall, source_native_wall, source_native_wall_parts, source_native_wall_reconstruction_rows, source_wall


def oblique_model(openings=False, slope=-0.25):
    room = Room.from_box("ObliqueRoom", 5, 4, 3)
    host = next(face for face in room.faces if face.normal.x > 0.5)
    if openings:
        host.add_aperture(Aperture("Window", Face3D(tuple(Point3D(5, y, z) for y, z in ((2.5, 1), (3.5, 1), (3.5, 2), (2.5, 2))))))
        host.add_door(Door("Door", Face3D(tuple(Point3D(5, y, z) for y, z in ((0.4, 0), (1.4, 0), (1.4, 2), (0.4, 2))))))
    def transform(point):
        return Point3D(point.x + slope * point.z if point.x > 4.99 else point.x, point.y, point.z)
    faces = []
    for original in room.faces:
        face = Face(original.identifier, Face3D(tuple(transform(point) for point in original.geometry.boundary)), original.type)
        for item in original.apertures:
            face.add_aperture(Aperture(item.identifier, Face3D(tuple(transform(point) for point in item.geometry.boundary))))
        for item in original.doors:
            face.add_door(Door(item.identifier, Face3D(tuple(transform(point) for point in item.geometry.boundary))))
        faces.append(face)
    room = Room(room.identifier, faces)
    room.story = "First"
    return Model("ObliqueFixture", [room])


def overhang_model(openings=True):
    room = Room.from_box("Overhang", 5, 4, 3)
    host = next(face for face in room.faces if face.normal.y < -0.99)
    if openings:
        host.add_aperture(Aperture("SplitWindow", Face3D(tuple(Point3D(x, 0, z) for x, z in ((4, 1), (4.8, 1), (4.8, 2), (4, 2))))))
    faces = []
    for source in room.faces:
        def transform(geometry):
            return Face3D(tuple(Point3D(point.x + point.z * 0.8 / 3, point.y, point.z) for point in geometry.boundary))
        face = Face(source.identifier, transform(source.geometry), source.type)
        for aperture in source.apertures:
            face.add_aperture(Aperture(aperture.identifier, transform(aperture.geometry)))
        faces.append(face)
    room = Room(room.identifier, faces)
    room.story = "First"
    return Model("OverhangFixture", [room])


def raised_shared_model(right_base=0):
    rooms = []
    for name, outer_x in (("Left", 0), ("Right", 10)):
        base = right_base if name == "Right" else 0
        joint = [Point3D(5 + (1 + y / 4) / 4, y, 1 + y / 4) for y in (0, 4)]
        bottom = [Point3D(p.x, p.y, base) for p in joint]
        top = [Point3D(6, y, 4) for y in (0, 4)]
        low, high = [[Point3D(outer_x, y, z) for y in (0, 4)] for z in (base, 4)]
        rings = [(low[0], *bottom, low[1]), (high[0], *top, high[1]), (*low, *reversed(high)),
                 (*bottom, *reversed(joint)), (*joint, *reversed(top))]
        rings.extend((low[i], bottom[i], joint[i], top[i], high[i]) for i in (0, 1))
        room = Room.from_polyface3d(name, Polyface3D.from_faces([Face3D(ring) for ring in rings], 1e-6))
        room.story = "Occupied"
        wall = next(face for face in room.faces if 0.1 < abs(face.normal.z) < 0.9)
        boundary = tuple(Point3D(5 + z / 4, y, z) for y, z in ((1, 2), (3, 2), (2, 3.5)))
        window = Face3D(boundary)
        wall.add_aperture(Aperture(name + "Window", window if wall.normal.dot(window.normal) > 0 else window.flip()))
        rooms.append(room)
    Room.solve_adjacency(rooms, 1e-6)
    return Model("RaisedShared", rooms)


class NativeObliqueTest(unittest.TestCase):
    def test_faceted_downward_exterior_wall_keeps_its_wall_role_and_source_normal(self):
        box = Room.from_box("Canopy", 5, 4, 3)
        faces = []
        for source in box.faces:
            ring = []
            for p in source.geometry.boundary:
                z = p.z + (p.y / 4 if p.z > 0 else 0)
                ring.append(Point3D(p.x + (1.6 * z if p.x == 5 else 0), p.y, z))
            pieces = (ring[:3], [ring[0], *ring[2:]]) if source.normal.x > 0.99 else (ring,)
            faces.extend(Face(f"{source.identifier}_{i}", Face3D(piece), source.type) for i,piece in enumerate(pieces))
        model = Model("SlopingEave", [Room("Canopy", faces)])
        drawing, report = build(model, inclined_carriers=True)
        validate(model, drawing.bytes(), report)
        targets = {f.identifier:f for f in faces if str(f.type) == "Wall" and f.normal.z < -0.8}
        panels = [s for s in report["surfaces"] if s["identifier"] in targets]
        self.assertEqual({s["identifier"] for s in panels}, set(targets))
        self.assertEqual(len(targets), 2)
        native = inspect_native(drawing.bytes(), report, include_geometry=True)["geometry"]["entities"]
        for s in panels:
            row = native[s["handle"]]
            self.assertEqual((row["type"], row["face_usage"]), ("POLYLINE_PFACE", 5))
            self.assertEqual(row["source_room"], report["rooms"][0]["native_code"])
            self.assertEqual(s["normal"], tuple(targets[s["identifier"]].normal))

    def test_profiled_floor_across_storeys_keeps_the_full_upward_shared_wall_projection(self):
        def ring(a, b, low, high, z, left_z):
            return tuple(Point3D(*p) for p in ((a, low, left_z), (b, low, z), (b, high, z), (a, high, left_z)))

        def solid(name, levels):
            faces = [Face3D(levels[0]), Face3D(levels[-1])]
            for low, high in zip(levels, levels[1:]):
                for i in range(4):
                    a, b, c, d = low[i], low[i-1], high[i-1], high[i]
                    face = Face3D((a, b, c, d))
                    faces.extend([face] if face.check_planar(1e-6, False) else [Face3D((a, b, c)), Face3D((a, c, d))])
            room = Room.from_polyface3d(name, Polyface3D.from_faces(faces, 1e-6))
            room.story = name
            return room

        upper = solid("Upper", [ring(0, 5, 0, 10, 3, 3.4), ring(0, 4.4, 4, 5, 6, 6)])
        lower = solid("Lower", [ring(5, 10, 0, 10, 0, 0), ring(5, 10, 0, 10, 3, 3), ring(4.4, 10, 4, 5, 6, 6)])
        Room.solve_adjacency([upper, lower], 1e-6)
        model = Model("CrossStoreyIncline", [upper, lower])
        shared, = [f for f in upper.faces if str(f.boundary_condition) == "Surface"]
        drawing, report = build(model, inclined_carriers=True)
        validate(model, drawing.bytes(), report)
        panels = [p for p in report["wall_panels"] if p["identifier"] == shared.identifier]
        self.assertEqual({(p["kind"], p["boundary_condition"]) for p in panels}, {("Wall", "Surface")})
        self.assertFalse(any(shared.identifier in w.get("source_identifiers", []) for w in report["walls"] if not w.get("virtual_plan")))
        self.assertAlmostEqual(sum(Face3D(tuple(Point3D(*v) for v in p["boundary_mm"])).area for p in panels) / 1e6, shared.area)
        self.assertEqual(report["source_geometry_validation"]["summary"]["review_items"], 0)

    def test_raised_shared_oblique_wall_retains_both_rooms_and_one_source_contour(self):
        model = raised_shared_model()
        model.rotate_xy(33, Point3D())
        model.convert_to_units("Feet")
        original = model.to_dict()
        drawing, report = build(model, inclined_carriers=True)
        validate(model, drawing.bytes(), report)
        require_export(report)
        walls = [row for row in report["walls"] if row.get("source_oblique")]
        self.assertEqual(len(walls), 1)
        wall = walls[0]
        self.assertTrue(wall["source_oblique_sloped_base"])
        self.assertEqual({wall["left_space"], wall["right_space"]}, {room["handle"] for room in report["rooms"]})
        self.assertEqual(len(wall["source_identifiers"]), 2)
        self.assertEqual(wall["catalog"], 1)
        self.assertEqual(len(report["openings"]), 1)
        self.assertEqual(report["openings"][0]["wall"], wall["handle"])
        self.assertEqual(set(report["openings"][0]["source_identifiers"]), {"LeftWindow", "RightWindow"})
        self.assertTrue(all(row["source_shape_passed"] for row in report["source_geometry_validation"]["openings"]))
        self.assertEqual(original, model.to_dict())

    def test_elevated_shared_wall_in_separate_frames_retains_panel_review(self):
        model = raised_shared_model(right_base=-1)
        drawing, report = build(model, inclined_carriers=True)
        validate(model, drawing.bytes(), report)
        self.assertEqual(len({row["story"] for row in report["rooms"]}), 2)
        self.assertTrue(any(row["boundary_condition"] == "Surface" for row in report["wall_panels"]))
        self.assertFalse(inspect_export(report)["required_checks"]["native_wall_window_panels_zero"])

    def test_wall_without_openings_preserves_source_plane_and_area(self):
        model = oblique_model()
        original = model.to_dict()
        drawing, report = build(model)
        data = drawing.bytes()
        validate(model, data, report)
        native = inspect_native(data, report, include_geometry=True)
        self.assertEqual(report["counts"]["native_oblique_walls"], 1)
        self.assertEqual(report["wall_panels"], [])
        row = next(row for row in report["walls"] if row.get("source_oblique"))
        decoded = native["geometry"]["entities"][row["handle"]]
        self.assertAlmostEqual(decoded["oblique_angle"], -math.atan(0.25))
        floor = next(floor for floor in report["floors"] if floor["story"] == row["story"])
        geometry = Face3D(tuple(Point3D(point[0] - floor["offset_mm"][0], point[1] - floor["offset_mm"][1], point[2] + floor["reference_elevation_mm"]) for point in wall_points(decoded)))
        source = next(face for face in model.rooms[0].faces if face.identifier == row["identifier"]).geometry.scale(1000)
        self.assertAlmostEqual(source.area, geometry.area, places=3)
        self.assertLess(max(source.plane.distance_to_point(point) for point in geometry.boundary), 1e-5)
        self.assertEqual(report["source_geometry_validation"]["summary"]["review_items"], 0)
        self.assertEqual(model.to_dict(), original)

    def test_rectangular_window_and_door_attach_to_native_oblique_wall(self):
        model = oblique_model(openings=True)
        drawing, report = build(model)
        validate(model, drawing.bytes(), report)
        walls = {row["handle"]: row for row in report["walls"]}
        self.assertEqual(len(report["openings"]), 2)
        for row in report["openings"]:
            host = walls[row["wall"]]
            self.assertTrue(host["source_oblique"])
            self.assertIn(row["handle"], host["openings"])
            height = 2000 if row["kind"] == "Door" else 1000
            self.assertAlmostEqual(row["height_mm"] * math.cos(host["oblique_angle"]), height, places=5)
        self.assertEqual(report["wall_panels"], [])
        self.assertEqual(report["reference_openings"], [])
        self.assertTrue(report["native_quality"]["opening_bindings_passed"])
        for row in report["source_geometry_validation"]["openings"]:
            self.assertTrue(row["raw_source_coverage_passed"])
            self.assertAlmostEqual(row["missing_area_m2"], 0, places=8)
            self.assertAlmostEqual(row["extra_area_m2"], 0, places=8)

    def test_rotated_elevated_feet_model_retains_source_coverage(self):
        model = oblique_model(openings=True)
        model.rotate_xy(37, Point3D(0, 0, 0))
        model.move(Vector3D(7, 8, 1.5))
        model.convert_to_units("Feet")
        drawing, report = build(model)
        validate(model, drawing.bytes(), report)
        self.assertEqual(report["counts"]["native_oblique_walls"], 1)
        self.assertEqual(report["wall_panels"], [])
        self.assertEqual(report["source_geometry_validation"]["summary"]["review_items"], 0)
        self.assertTrue(report["native_quality"]["opening_bindings_passed"])

    def test_decoded_angle_and_opening_height_drive_independent_coverage(self):
        model = oblique_model(openings=True)
        drawing, report = build(model)
        decoded = inspect_native(drawing.bytes(), report, include_geometry=True)
        host = next(row for row in report["walls"] if row.get("source_oblique"))
        changed = copy.deepcopy(decoded)
        changed["geometry"]["entities"][host["handle"]]["oblique_angle"] = 0.0
        result = compare_source(model, report, changed)
        self.assertGreater(result["summary"]["native_face_review"], 0)
        self.assertGreater(result["summary"]["review_openings"], 0)
        changed = copy.deepcopy(decoded)
        changed["geometry"]["entities"][report["openings"][0]["handle"]]["height_mm"] *= 4
        self.assertGreater(compare_source(model, report, changed)["summary"]["review_openings"], 0)
        self.assertFalse(inspect_quality(report, changed)["opening_bindings_passed"])

    def test_oblique_top_profile_keeps_its_native_source_completion_requirement(self):
        room = oblique_model().rooms[0]
        points = tuple(Point3D(5 - 0.25 * z, y, z) for y, z in ((0, 0), (4, 0), (4, 2), (2, 3), (0, 2)))
        face = Face("Profile", Face3D(points))
        floor = {"scale": 1000, "offset_mm": (0, 0), "reference_elevation_mm": 0, "story": "First", "room_handle": 1}
        row = source_oblique_wall(face, room, floor)
        self.assertIsNone(row)
        rectangular = source_oblique_wall(next(item for item in room.faces if str(item.type) == "Wall" and abs(item.normal.z) > 1e-4), room, floor)
        expected = wall_points(rectangular)
        rectangular["profile_mm"] = [(0, 2000), (2000, 3000), (4000, 2000)]
        self.assertEqual(wall_points(rectangular), expected)

    def test_oblique_3D_baseline_preserves_sloped_source_lower_edge(self):
        room = oblique_model().rooms[0]
        floor = {"scale": 1000, "offset_mm": (0, 0), "reference_elevation_mm": 0, "story": "First", "room_handle": 1}
        face = Face("SlopedBase", Face3D(tuple(Point3D(5 - 0.25 * z, y, z) for y, z in ((0, 0), (4, 1), (4, 3), (0, 3)))))
        row = sloped_oblique_wall(face, room, floor)
        self.assertIsNotNone(row)
        self.assertTrue(row["source_oblique_sloped_base"])
        self.assertAlmostEqual(row["end_mm"][2] - row["start_mm"][2], 1062.5)
        result = coverage(face.geometry, [(wall_points(row), [])], 1000, tolerance=1e-8)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["missing_area_m2"], 0)
        self.assertAlmostEqual(result["extra_area_m2"], 0)

    def test_sloped_base_and_inner_ring_remain_source_completion_candidates(self):
        room = oblique_model().rooms[0]
        floor = {"scale": 1000, "offset_mm": (0, 0), "reference_elevation_mm": 0, "story": "First", "room_handle": 1}
        boundary = tuple(Point3D(5 - 0.25 * z, y, z) for y, z in ((0, 0), (4, 1), (4, 3), (0, 3)))
        face = Face("SlopedBase", Face3D(boundary))
        self.assertIsNone(source_oblique_wall(face, room, floor))
        host = next(face for face in room.faces if abs(face.normal.z) > 1e-4 and str(face.type) == "Wall")
        hole = tuple(Point3D(5 - 0.25 * z, y, z) for y, z in ((1, 1), (2, 1), (2, 2), (1, 2)))
        face = Face("InnerRing", Face3D(host.geometry.boundary, holes=(hole,)))
        self.assertIsNone(source_oblique_wall(face, room, floor))

    def test_inner_ring_partitions_preserve_source_plane_and_coverage(self):
        model = oblique_model()
        room = model.rooms[0]
        host = next(face for face in room.faces if abs(face.normal.z) > 1e-4 and str(face.type) == "Wall")
        hole = tuple(Point3D(5 - 0.25 * z, y, z) for y, z in ((1, 1), (2, 1), (2, 2), (1, 2)))
        replacement = Face(host.identifier, Face3D(host.geometry.boundary, holes=(hole,)), host.type)
        room = Room(room.identifier, [replacement if face is host else face.duplicate() for face in room.faces])
        room.story = "First"
        model = Model("InnerRingFallback", [room])
        drawing, report = build(model)
        validate(model, drawing.bytes(), report)
        self.assertEqual(report["counts"]["native_oblique_walls"], 4)
        self.assertEqual(report["wall_panels"], [])
        self.assertEqual(report["source_geometry_validation"]["summary"]["review_items"], 0)

    def test_clearance_projects_intersection_of_oblique_wall_volumes(self):
        host = {"handle": 1, "identifier": "Host", "start_mm": (0, 0, 0), "end_mm": (10000, 0, 0), "height_mm": 3000, "left_width_mm": 100, "right_width_mm": 100, "oblique_angle": -math.atan(0.5)}
        other = {"handle": 2, "identifier": "Other", "start_mm": (0, 1000, 0), "end_mm": (10000, 1000, 0), "height_mm": 3000, "left_width_mm": 100, "right_width_mm": 100, "oblique_angle": math.atan(0.5)}
        masks = obstacles(host, {1: host, 2: other})
        self.assertEqual(len(masks), 1)
        for observed, expected in zip(masks[0]["shape"].bounds, (0, 800, 10000, 1200)):
            self.assertAlmostEqual(observed, expected)
        self.assertAlmostEqual(masks[0]["shape"].area, 4000000)

    def test_distinct_oblique_planes_keep_separate_wall_uniqueness(self):
        model = oblique_model()
        drawing, report = build(model)
        decoded = inspect_native(drawing.bytes(), report, include_geometry=True)
        wall = next(row for row in report["walls"] if row.get("source_oblique"))
        clone = {**wall, "handle": 999999}
        report["walls"].append(clone)
        decoded["geometry"]["entities"][clone["handle"]] = {**decoded["geometry"]["entities"][wall["handle"]], "oblique_angle": -wall["oblique_angle"]}
        self.assertTrue(inspect_quality(report, decoded)["same_storey_wall_uniqueness_passed"])
        decoded["geometry"]["entities"][clone["handle"]]["oblique_angle"] = wall["oblique_angle"]
        self.assertFalse(inspect_quality(report, decoded)["same_storey_wall_uniqueness_passed"])

    def test_vertical_side_overhang_uses_bounded_native_profile_parts(self):
        room = Room.from_box("TopOverhang", 5, 4, 3)
        face = Face("Overhang", Face3D(tuple(Point3D(x, 0, z) for x, z in ((0, 0), (5, 0), (5.8, 3), (0.8, 3)))))
        floor = {"scale": 1000, "offset_mm": (0, 0), "reference_elevation_mm": 0, "story": "First", "room_handle": 1}
        row = source_wall(face, room, floor, allow_base=True)
        self.assertIsNone(row)
        rows = source_native_wall_parts(face, room, floor, planar_polygon(face.geometry, face.geometry.plane, 1000))
        self.assertEqual(len(rows), 2)
        self.assertTrue(coverage(face.geometry, [(wall_points(row), []) for row in rows], 1000, tolerance=1e-8)["passed"])
        for row in rows:
            length = math.dist(row["start_mm"], row["end_mm"])
            self.assertTrue(all(-0.001 <= x <= length + 0.001 for x, z in row.get("profile_mm", [])))

    def test_vertical_top_steps_keep_repeated_along_coordinates(self):
        room = Room.from_box("SteppedTop", 4, 4, 3)
        face = Face("Stepped", Face3D(tuple(Point3D(x, 0, z) for x, z in ((0, 0), (4, 0), (4, 3), (3, 3), (3, 1), (1, 1), (1, 3), (0, 3)))))
        floor = {"scale": 1000, "offset_mm": (0, 0), "reference_elevation_mm": 0, "story": "First", "room_handle": 1}
        row = source_wall(face, room, floor, allow_base=True)
        self.assertIsNotNone(row)
        self.assertTrue(any(first[0] == last[0] and first[1] != last[1] for first, last in zip(row["profile_mm"], row["profile_mm"][1:])))
        result = coverage(face.geometry, [(wall_points(row), [])], 1000, tolerance=1e-8)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["missing_area_m2"], 0)
        self.assertAlmostEqual(result["extra_area_m2"], 0)

    def test_sloped_lower_boundary_combines_native_cap_and_sloped_base(self):
        room = Room.from_box("SlopedBottom", 4, 4, 3)
        face = Face("Sloped", Face3D(tuple(Point3D(x, 0, z) for x, z in ((0, 0), (4, 1), (4, 3), (0, 3)))))
        floor = {"scale": 1000, "offset_mm": (0, 0), "reference_elevation_mm": 0, "story": "First", "room_handle": 1}
        shape = planar_polygon(face.geometry, face.geometry.plane, 1000)
        rows = source_native_wall_parts(face, room, floor, shape)
        self.assertEqual(len(rows), 2)
        result = coverage(face.geometry, [(wall_points(row), []) for row in rows], 1000, tolerance=1e-8)
        self.assertAlmostEqual(result["source_area_m2"], 10)
        self.assertAlmostEqual(result["missing_area_m2"], 0)
        self.assertAlmostEqual(result["extra_area_m2"], 0)
        self.assertAlmostEqual(rows[0]["start_mm"][2], 1000)
        self.assertTrue(any(row.get("source_sloped_base") for row in rows))
        self.assertTrue(all(row["height_mm"] > 0 for row in rows))

    def test_sdk_partition_uses_opening_base_to_keep_native_window_coverage(self):
        room = Room.from_box("OpeningBase", 4, 4, 3)
        face = Face("Sloped", Face3D(tuple(Point3D(x, 0, z) for x, z in ((0, 0), (4, 1), (4, 3), (0, 3)))))
        aperture = Aperture("LowWindow", Face3D(tuple(Point3D(x, 0, z) for x, z in ((1, 0.5), (2, 0.5), (2, 1.5), (1, 1.5)))))
        face.add_aperture(aperture)
        floor = {"scale": 1000, "offset_mm": (0, 0), "reference_elevation_mm": 0, "story": "First", "room_handle": 1}
        rows = source_native_wall_parts(face, room, floor, planar_polygon(face.geometry, face.geometry.plane, 1000))
        self.assertEqual(len(rows), 2)
        self.assertTrue(any(row.get("source_sloped_base") for row in rows))
        self.assertTrue(coverage(aperture.geometry, [(wall_points(row), []) for row in rows], 1000, tolerance=1e-8, check_extra=False)["passed"])
        result = coverage(face.geometry, [(wall_points(row), []) for row in rows], 1000, tolerance=1e-8)
        self.assertAlmostEqual(result["missing_area_m2"], 0)
        self.assertAlmostEqual(result["extra_area_m2"], 0)

    def test_clearance_covers_side_overhang_at_absolute_elevation(self):
        for slope in (0, 0.1):
            host = {"handle": 1, "identifier": "Host", "start_mm": (0, 0, 1500), "end_mm": (10000, 0, 1500), "height_mm": 3000, "left_width_mm": 100, "right_width_mm": 100}
            other = {"handle": 2, "identifier": "Overhang", "start_mm": (8000, -1000, 2000), "end_mm": (8000, 1000, 2000), "height_mm": 3000, "left_width_mm": 100, "right_width_mm": 100, "profile_mm": [(3000, 3000), (5000, 3000)], "oblique_angle": math.atan(slope)}
            masks = obstacles(host, {1: host, 2: other})
            self.assertEqual(len(masks), 1)
            expected_bounds = (7900, 2000, 8100, 3100) if slope == 0 else (7900, 2000, 8350, 4500)
            for observed, expected in zip(masks[0]["shape"].bounds, expected_bounds):
                self.assertAlmostEqual(observed, expected, places=4)
            self.assertAlmostEqual(masks[0]["shape"].area, 220000 if slope == 0 else 500000, places=3)

    def test_sdk_vertical_reconstruction_retains_complete_v_notch(self):
        room = Room.from_box("Notch", 10, 4, 3.8)
        coordinates = ((0, 3.8), (0, 0), (3, 0), (3, 3.3227762695), (5, 2.5469027007), (7, 3.3227762791), (7, 0), (10, 0), (10, 3.8))
        for slope in (0, 0.2):
            face = Face("NotchedExterior", Face3D(tuple(Point3D(x, slope * z, z) for x, z in coordinates)))
            floor = {"scale": 1000, "offset_mm": (0, 0), "reference_elevation_mm": 0, "story": "First", "room_handle": 1}
            rows = source_native_wall_reconstruction_rows(face, room, floor)
            self.assertEqual(rows, [])
            rows = source_native_wall_parts(face, room, floor, planar_polygon(face.geometry, face.geometry.plane, 1000))
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(-0.001 <= x <= math.dist(row["start_mm"], row["end_mm"]) + 0.001 for row in rows for x, z in row.get("profile_mm", [])))
            result = coverage(face.geometry, [(wall_points(row), []) for row in rows], 1000, tolerance=1e-8)
            self.assertTrue(result["passed"])
            self.assertLess(result["missing_area_m2"], 1e-9)
            self.assertLess(result["extra_area_m2"], 1e-9)

    def test_notched_exterior_reconstruction_keeps_coplanar_boundary_and_openings(self):
        room = Room.from_box("SemanticNotch", 10, 4, 3.8)
        original = next(face for face in room.faces if face.normal.y < -0.5)
        outside = Face("OutdoorNotch", Face3D(tuple(Point3D(x, 0, z) for x, z in ((0, 3.8), (0, 0), (3, 0), (3, 3.3227762695), (5, 2.5469027007), (7, 3.3227762791), (7, 0), (10, 0), (10, 3.8)))))
        inside = Face("InsulatedNotch", Face3D(tuple(Point3D(x, 0, z) for x, z in ((3, 0), (7, 0), (7, 3.3227762791), (5, 2.5469027007), (3, 3.3227762695)))))
        inside.boundary_condition = Adiabatic()
        for identifier, coordinates in (("LeftWindow", ((0.5, 1), (2, 1), (2, 2), (0.5, 2))), ("RightWindow", ((8, 1), (9, 1), (9, 2), (8, 2))), ("NotchWindow", ((4.995, 2.55), (5.005, 2.55), (5.005, 3.5), (4.995, 3.5)))):
            outside.add_aperture(Aperture(identifier, Face3D(tuple(Point3D(x, 0, z) for x, z in coordinates))))
        room = Room(room.identifier, [face for face in room.faces if face is not original] + [outside, inside])
        room.story = "First"
        model = Model("SemanticNotch", [room])
        model.convert_to_units("Feet")
        original_model = model.to_dict()
        drawing, report = build(model)
        validate(model, drawing.bytes(), report)
        require_export(report)
        self.assertTrue(report["native_opening_baseline_compatibility"]["records"])
        self.assertEqual(report["wall_panels"], [])
        self.assertEqual(report["source_geometry_validation"]["summary"]["passed_openings"], 3)
        self.assertTrue(report["native_quality"]["opening_bindings_passed"])
        self.assertTrue(report["native_quality"]["same_storey_wall_uniqueness_passed"])
        self.assertTrue(report["source_geometry_validation"]["boundary_conditions"]["passed"])
        rebuilt = [row for row in report["walls"] if row.get("source_completion") and outside.identifier in row.get("source_identifiers", [])]
        self.assertTrue(rebuilt)
        self.assertTrue(all(row["source_identifiers"] == [outside.identifier] and row["boundary_condition"] == "Outdoors" for row in rebuilt))
        insulated = [row for row in report["walls"] if inside.identifier in row.get("source_identifiers", [])]
        self.assertEqual(len(insulated), 1)
        self.assertEqual(insulated[0]["boundary_condition"], "Adiabatic")
        self.assertNotEqual(insulated[0]["construction_id"], rebuilt[0]["construction_id"])
        self.assertEqual(model.to_dict(), original_model)

    def test_bounded_overhang_profiles_keep_source_openings_and_room_links(self):
        for angle, units in ((0, "Meters"), (90, "Meters"), (33, "Feet")):
            with self.subTest(angle=angle, units=units):
                model = overhang_model()
                model.rotate_xy(angle, Point3D())
                model.convert_to_units(units)
                original = model.to_dict()
                drawing, report = build(model, inclined_carriers=True)
                validate(model, drawing.bytes(), report)
                self.assertEqual(len(report["walls"]), 6)
                self.assertEqual(report["wall_panels"], [])
                self.assertTrue(report["native_profile_compatibility"]["records"])
                self.assertTrue(report["native_quality"]["native_profile_corner_independence_passed"])
                self.assertTrue(report["native_quality"]["native_profile_baseline_bounds_passed"])
                self.assertEqual(report["source_geometry_validation"]["summary"]["review_items"], 0)
                self.assertEqual(report["source_geometry_validation"]["summary"]["passed_openings"], 1)
                self.assertTrue(all(row["left_space"] or row["right_space"] for row in report["walls"]))
                self.assertTrue(inspect_export(report)["required_checks"]["native_opening_base_projection"])
                self.assertTrue(report["native_opening_baseline_compatibility"]["records"])
                self.assertEqual(model.to_dict(), original)

    def test_bounded_overhang_without_openings_keeps_the_export_gate(self):
        model = overhang_model(openings=False)
        drawing, report = build(model, inclined_carriers=True)
        validate(model, drawing.bytes(), report)
        self.assertEqual(len(report["walls"]), 6)
        self.assertTrue(inspect_export(report)["passed"])

    def test_vertical_profile_walls_keep_their_native_corner_connections(self):
        box = Room.from_box("Pitched", 5, 4, 3)
        room = Room(box.identifier, [Face(face.identifier, Face3D(tuple(Point3D(point.x, point.y, point.z + point.x * 0.2 if point.z > 0 else 0) for point in face.geometry.boundary)), face.type) for face in box.faces])
        drawing, report = build(Model("Pitched", [room]))
        profiles = [row for row in report["walls"] if row.get("profile_mm")]
        self.assertTrue(profiles)
        self.assertEqual(report["native_profile_compatibility"]["records"], [])
        self.assertTrue(all(all(row["neighbors"]) for row in profiles))

    def test_sloped_baseline_origin_keeps_source_profile_and_independent_projection_gate(self):
        model = overhang_model()
        drawing, report = build(model, inclined_carriers=True)
        validate(model, drawing.bytes(), report)
        self.assertTrue(report["native_quality"]["opening_base_projection_passed"])
        decoded = inspect_native(drawing.bytes(), report, include_geometry=True)
        repair = report["native_opening_baseline_compatibility"]["records"][0]
        native = decoded["geometry"]["entities"][repair["handle"]]
        native["position_mm"] = (*native["position_mm"][:2], native["position_mm"][2] + 100)
        self.assertFalse(inspect_quality(report, decoded)["opening_base_projection_passed"])

    def test_independent_profile_gate_detects_restored_corner_dependencies(self):
        model = overhang_model()
        drawing, report = build(model, inclined_carriers=True)
        validate(model, drawing.bytes(), report)
        changed = inspect_native(drawing.bytes(), report, include_geometry=True)
        profile = next(row for row in report["walls"] if row.get("profile_mm"))
        oblique = next(row for row in report["walls"] if row.get("oblique_angle"))
        changed["geometry"]["entities"][profile["handle"]]["neighbors"][0] = oblique["handle"]
        changed["geometry"]["entities"][oblique["handle"]]["neighbors"][0] = profile["handle"]
        quality = inspect_quality(report, changed)
        self.assertFalse(quality["native_profile_corner_independence_passed"])
        report["native_quality"] = quality
        self.assertFalse(inspect_export(report)["required_checks"]["native_profile_corner_independence"])

    def test_independent_profile_gate_detects_baseline_overhang(self):
        model = overhang_model()
        drawing, report = build(model, inclined_carriers=True)
        validate(model, drawing.bytes(), report)
        changed = inspect_native(drawing.bytes(), report, include_geometry=True)
        profile = next(row for row in report["walls"] if row.get("profile_mm"))
        row = changed["geometry"]["entities"][profile["handle"]]
        x, z = row["profile_mm"][0]
        row["profile_mm"][0] = (-100.0, z)
        self.assertFalse(inspect_quality(report, changed)["native_profile_baseline_bounds_passed"])

    def test_floor_profile_carriers_include_raised_horizontal_source_faces(self):
        boundary = tuple(Point3D(0, y, z) for y, z in ((0, 0), (2, 0), (2, 1), (4, 1), (4, 3), (0, 3)))
        shift = Vector3D(5, 0, 0)
        far = tuple(point.move(shift) for point in boundary)
        faces = [Face3D(boundary), Face3D(tuple(reversed(far)))]
        faces.extend(Face3D((first, last, last.move(shift), first.move(shift))) for first, last in zip(boundary, (*boundary[1:], boundary[0])))
        room = Room.from_polyface3d("SteppedFloor", Polyface3D.from_faces(faces, 0.001))
        room.story = "First"
        model = Model("SteppedFloor", [room])
        original = model.to_dict()
        drawing, report = build(model, inclined_carriers=True, inclined_floors=True)
        validate(model, drawing.bytes(), report)
        carriers = [row for row in report["enclosure_carriers"]["parts"] if row.get("source_kind") == "Floor"]
        self.assertAlmostEqual(room.volume, 50)
        self.assertEqual(len(carriers), 1)
        self.assertEqual(carriers[0]["source_floor_profile"], "elevated_horizontal")
        self.assertEqual(carriers[0]["face_usage"], 8)
        self.assertLess(carriers[0]["normal"][2], -0.9999)
        self.assertEqual(carriers[0]["source_room_floor_base_elevation_mm"], 0)
        self.assertEqual(report["source_geometry_validation"]["summary"]["review_items"], 0)
        self.assertEqual(model.to_dict(), original)


class NativeWallToleranceTest(unittest.TestCase):
    def setUp(self):
        self.room = Room.from_box("NearVertical", 7, 4, 3)
        self.floor = {"scale": 1000, "offset_mm": (140000, 0), "reference_elevation_mm": 13000,
                      "story": "First", "room_handle": 1}

    def face(self, slope):
        return Face("SlopedBase", Face3D(tuple(Point3D(x, slope * z, 13 + z) for x, z in ((0, 0), (6.5, 0.24), (6.3, 0.32), (0, 1.07)))))

    def test_near_vertical_sloped_base_uses_exact_native_profile_tolerance(self):
        face = self.face(6.5e-7)
        row = source_native_wall(face, self.room, self.floor)
        self.assertIsNotNone(row)
        self.assertTrue(row["source_sloped_base"])
        self.assertGreater(row["height_mm"], 0)
        self.assertEqual(row["left_space"], 1)
        self.assertEqual(row["right_space"], 0)
        native = wall_boundary(row, self.floor)
        source = face.geometry.scale(1000)
        deviation = max(source.plane.distance_to_point(point) for point in native)
        self.assertGreater(deviation, 0.0001)
        self.assertLess(deviation, 0.001)
        result = coverage(face.geometry, [([tuple(point) for point in native], [])], 1000, tolerance=1e-8)
        self.assertTrue(result["passed"])
        self.assertLess(result["missing_area_m2"], 1e-9)
        self.assertLess(result["extra_area_m2"], 1e-9)
        self.assertEqual(len(wall_points(row)), len(face.geometry.boundary))

    def test_near_vertical_plane_deviation_preserves_the_exact_gate(self):
        self.assertIsNone(source_native_wall(self.face(2e-6), self.room, self.floor))

    def test_source_nonplanarity_preserves_the_exact_gate(self):
        boundary = list(self.face(6.5e-7).geometry.boundary)
        point = boundary[2]
        boundary[2] = Point3D(point.x, point.y + 0.000003, point.z)
        face = Face("Nonplanar", Face3D(tuple(boundary)))
        self.assertGreater(max(face.geometry.plane.distance_to_point(point) * 1000 for point in boundary), 0.001)
        self.assertIsNone(source_native_wall(face, self.room, self.floor))

    def test_sub_square_millimeter_residual_retains_the_area_gate(self):
        face = Face("TinyResidual", self.face(6.5e-7).geometry.scale(1e-5))
        self.assertLess(face.area * self.floor["scale"] ** 2, 1.0)
        self.assertIsNone(sloped_base_wall(face, self.room, self.floor, face.geometry))


if __name__ == "__main__":
    unittest.main()
