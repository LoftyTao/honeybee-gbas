from __future__ import annotations

import copy
import struct
import unittest
from math import cos, sin

from honeybee.aperture import Aperture
from honeybee.boundarycondition import boundary_conditions
from honeybee.door import Door
from honeybee.face import Face
from honeybee.model import Model
from honeybee.room import Room
from ladybug_geometry.geometry3d import Face3D, Point3D
from shapely.geometry import Polygon

from honeybee_gbas.dwg import build
from honeybee_gbas.dwg.binary import extended_data
from honeybee_gbas.dwg.geometry import opening_points
from honeybee_gbas.dwg.records import new_drawing, opening_block
from honeybee_gbas.dwg.surfaces import add_wall_profiles
from honeybee_gbas.dwg.validation import compare_source, inspect_export, inspect_native, inspect_quality, ring_area, validate
from honeybee_gbas.dwg.walls import obstacles


def rectangle(identifier, low, high, bottom, top, door=False):
    geometry = Face3D(tuple(Point3D(x, 0, z) for x, z in ((low, bottom), (high, bottom), (high, top), (low, top))))
    return (Door if door else Aperture)(identifier, geometry)


def junction_model():
    room = Room.from_box("Main", 10, 3, 3)
    neighbor = Room.from_box("Neighbor", 2, 4, 3, origin=Point3D(5, -4, 0))
    for item in (room, neighbor):
        item.story = "Occupied"
    face = next(face for face in room.faces if face.normal.y < -0.5)
    face.add_aperture(Aperture("Window", Face3D(tuple(Point3D(x, 0, z) for x, z in ((4, 0.5), (8, 0.5), (8, 2.5), (4, 2.5))))))
    return Model("Clearance", [room, neighbor])


def asymmetric_windows_model():
    box = Room.from_box("Room", 5, 4, 3)
    room = Room("Room", [Face(face.identifier, Face3D(tuple(Point3D(point.x, point.y, point.z + point.x * 0.4 if point.z > 0 else 0)
                                                               for point in face.geometry.boundary)), face.type) for face in box.faces])
    room.story = "Occupied"
    windows = (((0, -1), "South", ((0.2, 0, 0.3), (4.8, 0, 0.3), (4.8, 0, 4.6), (0.2, 0, 2.8))),
               ((0, 1), "North", ((4.8, 4, 0.3), (0.2, 4, 0.3), (0.2, 4, 2.8), (4.8, 4, 4.6))),
               ((1, 0), "East", ((5, 0.2, 0.3), (5, 3.8, 0.3), (5, 3.8, 4.7), (5, 0.2, 4.3))),
               ((-1, 0), "West", ((0, 3.8, 0.3), (0, 0.2, 0.3), (0, 0.2, 2.6), (0, 3.8, 2.8))))
    for normal, identifier, points in windows:
        face = next(face for face in room.faces if face.normal.x * normal[0] + face.normal.y * normal[1] > 0.99)
        face.add_aperture(Aperture(identifier, Face3D(tuple(Point3D(*point) for point in points))))
    return Model("AsymmetricWindows", [room])


class NativeOpeningsTest(unittest.TestCase):
    def test_t_junction_partitions_the_window_and_trims_wall_volume_conflicts(self):
        first = Room.from_box("Main", 10, 3, 3)
        second = Room.from_box("Adjoining", 2, 4, 3, origin=Point3D(5, -4, 0))
        for room in (first, second):
            room.story = "Occupied"
        face = next(face for face in first.faces if face.normal.y < -0.5)
        window = rectangle("CrossSegment", 4, 8, 0.5, 2.5)
        face.add_aperture(window)
        model = Model("PartitionedOpening", [first, second])
        drawing, manifest = build(model)
        validate(model, drawing.bytes(), manifest)
        native = [row for row in manifest["openings"] if window.identifier in row["source_identifiers"]]
        self.assertGreater(len(native), 1)
        self.assertLess(sum(row["width_mm"] for row in native), 4000)
        self.assertEqual(manifest["wall_panels"], [])
        comparison = manifest["source_geometry_validation"]["openings"][0]
        self.assertGreater(comparison["clearance_trimmed_area_m2"], 0)
        self.assertAlmostEqual(comparison["missing_area_m2"], comparison["clearance_trimmed_area_m2"])
        self.assertLessEqual(comparison["effective_missing_area_m2"], 1e-6)
        self.assertLessEqual(comparison["effective_extra_area_m2"], 1e-6)
        self.assertTrue(comparison["passed"])
        self.assertFalse(comparison["source_shape_passed"])
        self.assertEqual(comparison["source_shape_status"], "clearance_adapted")
        summary = manifest["source_geometry_validation"]["summary"]
        self.assertEqual(summary["adapted_source_opening_shapes"], 1)
        self.assertEqual(summary["passed_source_opening_shapes"], 0)
        warning = next(row for row in inspect_export(manifest)["warnings"] if row["code"] == "source_opening_clearance_geometry_change")
        self.assertAlmostEqual(warning["trimmed_area_m2"], comparison["clearance_trimmed_area_m2"])

    def test_dense_windows_keep_fifty_mm_gaps_and_account_for_corner_trimming(self):
        first = Room.from_box("ComputerLab", 9.06925, 3, 3)
        second = Room.from_box("Junction", 1, 4, 3, origin=Point3D(0, -4, 0))
        for room in (first, second):
            room.story = "Occupied"
        face = next(face for face in first.faces if face.normal.y < -0.5)
        for index in range(10):
            low = 0.11925 + index * 0.85
            face.add_aperture(rectangle(f"Dense{index}", low, low + 0.8, 1, 2))
        model = Model("DenseSourceWindows", [first, second])
        drawing, manifest = build(model)
        validate(model, drawing.bytes(), manifest)
        native = [row for row in manifest["openings"] if row["identifier"].startswith("Dense")]
        self.assertEqual({identifier for row in native for identifier in row["source_identifiers"]}, {f"Dense{index}" for index in range(10)})
        self.assertLess(sum(row["width_mm"] for row in native), 8000)
        self.assertEqual(manifest["wall_panels"], [])
        comparisons = manifest["source_geometry_validation"]["openings"]
        self.assertEqual(len(comparisons), 10)
        self.assertTrue(all(row["effective_missing_area_m2"] <= 1e-6 and row["effective_extra_area_m2"] <= 1e-6 for row in comparisons))
        self.assertTrue(all(abs(row["missing_area_m2"] - row["clearance_trimmed_area_m2"]) <= 1e-6 for row in comparisons))
        self.assertAlmostEqual(face.apertures[1].min.x - face.apertures[0].max.x, 0.05)

    def test_elevated_source_door_keeps_its_raised_native_host(self):
        first = Room.from_box("Tall", 5, 4, 8.4)
        below = Room.from_box("Below", 5, 4, 4.2, origin=Point3D(0, -4, 0))
        original = next(face for face in first.faces if face.normal.y < -0.5)
        lower = Face("SharedBelow", rectangle("BelowGeometry", 0, 5, 0, 4.2).geometry)
        elevated = Face("ElevatedHost", rectangle("AboveGeometry", 0, 5, 4.2, 8.4).geometry)
        door = rectangle("RaisedDoor", 1, 2.5, 4.25, 6.65, door=True)
        elevated.add_door(door)
        first = Room("Tall", [face for face in first.faces if face is not original] + [lower, elevated])
        for room in (first, below):
            room.story = "Occupied"
        model = Model("ElevatedDoor", [below, first])
        drawing, manifest = build(model)
        validate(model, drawing.bytes(), manifest)
        native = next(row for row in manifest["openings"] if door.identifier in row["source_identifiers"])
        host = next(row for row in manifest["walls"] if row["handle"] == native["wall"])
        self.assertTrue(host["source_elevated"])
        floor = next(row for row in manifest["floors"] if row["story"] == host["story"])
        self.assertAlmostEqual(host["start_mm"][2] + floor["reference_elevation_mm"], 4200)
        self.assertAlmostEqual(native["position_mm"][2] + floor["reference_elevation_mm"], 4250)
        self.assertAlmostEqual(native["height_mm"], 2400)
        self.assertEqual(manifest["wall_panels"], [])
        self.assertEqual(manifest["source_geometry_validation"]["summary"]["passed_openings"], 1)

    def test_profile_span_accepts_submicron_precision_rounding_at_wall_endpoint(self):
        width = 3.14159265359
        face = Face("Gable", Face3D(tuple(Point3D(x, 0, z) for x, z in ((0, 0), (width, 0), (width, 2), (width / 2, 3), (0, 2)))))
        room = Room("Room", [face])
        row = {"handle": 1, "identifier": face.identifier, "source_room": room.identifier, "story": "Occupied", "start_mm": (0, 0, 0), "end_mm": (width * 1000, 0, 0), "height_mm": 3000}
        report = {"walls": [row], "floors": [{"story": "Occupied", "offset_mm": (0, 0), "reference_elevation_mm": 0}], "wall_profile_geometry": []}
        add_wall_profiles(Model("EndpointRounding", [room]), report)
        self.assertEqual(report["wall_profile_geometry"][0]["status"], "native_profile")
        self.assertAlmostEqual(row["profile_mm"][0][1], 2000)
        self.assertAlmostEqual(row["profile_mm"][-1][1], 2000)


class OpeningClearanceTest(unittest.TestCase):
    def test_stacked_corner_blockers_trim_the_native_symbol_to_the_usable_span(self):
        for gap, expected_start in ((1.5e-12, 62.65), (0.0005, 50.0)):
            with self.subTest(gap_m=gap):
                box = Room.from_box("Lounge", 7, 6, 7.6)
                west = next(face for face in box.faces if face.normal.x < -0.9)
                south = next(face for face in box.faces if face.normal.y < -0.9)
                south.add_aperture(rectangle("AcrossSeam", 0.05, 0.65, 3.5, 4.25))
                def jamb(identifier, bottom, top):
                    return Face(identifier, Face3D(tuple(Point3D(0, y, z) for y, z in
                                ((6, bottom), (0, bottom), (0, top), (6, top)))))
                lower, upper = jamb("Lower", 0, 3.8), jamb("Upper", 3.8 + gap, 7.6)
                upper.boundary_condition = boundary_conditions.adiabatic
                room = Room("Lounge", [face for face in box.faces if face is not west] + [lower, upper])
                room.story = "Entry"
                drawing, report = build(Model("StoreySeam", [room]))
                decoded = inspect_native(drawing.bytes(), report, include_geometry=True)["geometry"]
                row, = report["openings"]
                native = decoded["entities"][row["handle"]]
                host = decoded["entities"][native["wall"]]
                self.assertEqual(host["openings"], [row["handle"]])
                self.assertAlmostEqual(native["width_mm"], 650.0 - expected_start)
                self.assertAlmostEqual(native["position_mm"][0], (650.0 + expected_start) / 2)
                block = decoded["blocks"][native["block_3d"]]
                boundary = decoded["polylines"][dict(block["xdata"])[1005]]
                points = opening_points(native, host, [(-u, v) for u, v in boundary["points"]])
                self.assertAlmostEqual(min(x for x, y, z in points), expected_start)
                self.assertAlmostEqual(max(x for x, y, z in points), 650.0)

    def test_blocker_projection_uses_the_actual_host_vertical_extent(self):
        host = {"handle": 1, "identifier": "Host", "start_mm": (0, 0, 1000), "end_mm": (10000, 0, 1000), "height_mm": 2000, "left_width_mm": 100, "right_width_mm": 100}
        other = {"handle": 2, "identifier": "Blocker", "start_mm": (5000, -1000, 0), "end_mm": (5000, 1000, 0), "height_mm": 6000, "left_width_mm": 200, "right_width_mm": 200}
        masks = obstacles(host, {1: host, 2: other})
        self.assertEqual(len(masks), 1)
        self.assertEqual(masks[0]["shape"].bounds, (4800, 1000, 5200, 3000))
        self.assertAlmostEqual(masks[0]["shape"].area, 800000)

    def test_raised_and_sloped_blockers_keep_their_vertical_limits(self):
        host = {"handle": 1, "identifier": "Host", "start_mm": (0, 0, 0), "end_mm": (10000, 0, 0), "height_mm": 3000, "left_width_mm": 100, "right_width_mm": 100}
        other = {"handle": 2, "identifier": "Blocker", "start_mm": (5000, -1000, 1900), "end_mm": (5000, 1000, 1900), "height_mm": 200, "left_width_mm": 200, "right_width_mm": 200}
        mask = obstacles(host, {1: host, 2: other})[0]["shape"]
        self.assertEqual(mask.bounds, (4800, 1900, 5200, 2100))
        other.update(start_mm=(5000, -1000, 0), end_mm=(5000, 1000, 0), height_mm=4000, profile_mm=[(0, 0), (2000, 4000)])
        mask = obstacles(host, {1: host, 2: other})[0]["shape"]
        self.assertEqual(mask.bounds, (4800, 0, 5200, 2200))

    def test_independent_clearance_uses_decoded_walls_and_source_geometry(self):
        model = junction_model()
        drawing, manifest = build(model)
        readback = inspect_native(drawing.bytes(), manifest, include_geometry=True)
        manifest["opening_clearance"]["records"] = []
        self.assertTrue(compare_source(model, manifest, readback)["openings"][0]["passed"])
        changed = copy.deepcopy(readback)
        blockers = {entry["wall"] for row in manifest["openings"] for entry in row["clipping_blockers"]}
        self.assertTrue(blockers)
        for handle in blockers:
            changed["geometry"]["entities"][handle]["left_width_mm"] += 500
            changed["geometry"]["entities"][handle]["right_width_mm"] += 500
        comparison = compare_source(model, manifest, changed)["openings"][0]
        self.assertFalse(comparison["passed"])
        self.assertGreater(comparison["effective_extra_area_m2"], 0)

    def test_opening_crossing_shared_wall_top_uses_both_final_native_hosts(self):
        room = Room.from_box("Tall", 5, 4, 6)
        neighbor = Room.from_box("Lower", 5, 4, 3, origin=Point3D(5, 0, 0))
        for item in (room, neighbor):
            item.story = "Occupied"
        face = next(face for face in room.faces if face.normal.x > 0.5)
        face.add_aperture(Aperture("HighWindow", Face3D(tuple(Point3D(5, y, z) for y, z in ((1, 2), (3, 2), (3, 5), (1, 5))))))
        model = Model("TwoHostWindow", [room, neighbor])
        drawing, manifest = build(model)
        validate(model, drawing.bytes(), manifest)
        openings = [row for row in manifest["openings"] if row["identifier"] == "HighWindow"]
        self.assertEqual(len(openings), 2)
        self.assertEqual(len({row["wall"] for row in openings}), 2)
        self.assertTrue(manifest["native_quality"]["opening_bindings_passed"])
        self.assertTrue(manifest["source_geometry_validation"]["openings"][0]["passed"])
        self.assertEqual(manifest["wall_panels"], [])


class OpeningNativeOutlineTest(unittest.TestCase):
    def fixture(self):
        room = Room.from_box("room", 5, 4, 3)
        room.story = "Level 1"
        face = next(face for face in room.faces if str(face.type) == "Wall" and face.normal.y < -0.9)
        face.add_aperture(Aperture("triangle", Face3D([Point3D(1, 0, 1), Point3D(3, 0, 1), Point3D(2, 0, 2.5)])))
        return build(Model("shaped_window", [room]))

    def test_triangle_has_native_boundary_protocol(self):
        drawing, report = self.fixture()
        opening = report["openings"][0]
        block = opening["block_3d"]
        tags = drawing.block_xdata[block][0][1]
        boundary = next(entity for entity in drawing.block_entities[block] if entity.handle == tags[2][1])
        self.assertEqual(tags[:2], ((1040, 1200.0), (1040, 1200.0)))
        self.assertEqual(tags[2][0], 1005)
        self.assertEqual(boundary.layer, "_TCH_BOUNDARY")
        self.assertEqual(boundary.type_code, 0x4D)
        self.assertAlmostEqual(ring_area(opening["profile_mm"]) / 1200.0 ** 2, 0.5)
        self.assertEqual(inspect_native(drawing.bytes(), report)["status"], "offline_readback_passed")

    def test_asymmetric_native_contours_preserve_source_shapes_at_rotated_wall_angles(self):
        for rotation in (0, 37, 123):
            model = asymmetric_windows_model()
            model.rotate_xy(rotation, Point3D())
            original = model.to_dict()
            drawing, report = build(model)
            decoded = inspect_native(drawing.bytes(), report, include_geometry=True)["geometry"]
            floors = {row["story"]: row for row in report["floors"]}
            sources = {item.identifier: item for room in model.rooms for face in room.faces for item in face.apertures}
            for row in report["openings"]:
                native = decoded["entities"][row["handle"]]
                block = decoded["blocks"][native["block_3d"]]
                boundary = decoded["polylines"][dict(block["xdata"])[1005]]
                angle = native["angle_rad"]
                normal = boundary["normal"]
                direction = 1 if -normal[0] * sin(angle) + normal[1] * cos(angle) < -1e-6 else -1
                profile = [(direction * x, z) for x, z in boundary["points"]]
                points = opening_points(native, decoded["entities"][native["wall"]], profile)
                floor = floors[row["story"]]
                source = sources[row["identifier"]].geometry.scale(1000)
                plane = source.plane
                expected = Polygon([tuple(plane.xyz_to_xy(point)) for point in source.boundary])
                actual = Polygon([tuple(plane.xyz_to_xy(Point3D(x - floor["offset_mm"][0], y - floor["offset_mm"][1], z + floor["reference_elevation_mm"]))) for x, y, z in points])
                with self.subTest(rotation=rotation, opening=row["identifier"]):
                    self.assertLessEqual(expected.symmetric_difference(actual).area, 1.0)
            self.assertEqual(model.to_dict(), original)

    def test_opposite_opening_directions_use_distinct_orientation_bound_blocks(self):
        drawing, _ = new_drawing()
        profile = [(-600, 0), (600, 0), (600, 1200), (-600, 600)]
        forward = opening_block(drawing, "WIN", profile, 200, 0)
        backward = opening_block(drawing, "WIN", profile, 200, 3.141592653589793)
        self.assertNotEqual(forward, backward)
        self.assertEqual(forward, opening_block(drawing, "WIN", profile, 200, 0.4))
        self.assertEqual(backward, opening_block(drawing, "WIN", profile, 200, -3.141592653589793))

    def test_independent_readback_detects_mirrored_contours_and_host_overhang(self):
        model = asymmetric_windows_model()
        drawing, manifest = build(model)
        readback = inspect_native(drawing.bytes(), manifest, include_geometry=True)
        self.assertTrue(inspect_quality(manifest, readback)["opening_bindings_passed"])
        row = next(row for row in manifest["openings"] if row["identifier"] == "South")
        block = readback["geometry"]["blocks"][row["block_3d"]]
        boundary = readback["geometry"]["polylines"][dict(block["xdata"])[1005]]
        boundary["normal"] = (0, -1, 0)
        comparison = next(item for item in compare_source(model, manifest, readback)["openings"] if item["identifier"] == "South")
        self.assertFalse(comparison["passed"])
        self.assertAlmostEqual(comparison["extra_area_m2"], 2.07)
        quality = inspect_quality(manifest, readback)
        self.assertFalse(quality["opening_bindings_passed"])
        self.assertTrue(any(item["code"] == "native_opening_attachment_geometry_required" and item["outside_host_area_m2"] > 1 for item in quality["issues"]))

    def test_missing_outline_handle_requires_review(self):
        drawing, report = self.fixture()
        block = report["openings"][0]["block_3d"]
        appid, tags = drawing.block_xdata[block][0]
        drawing.block_xdata[block] = ((appid, (*tags[:2], (1005, 0))),)
        with self.assertRaisesRegex(ValueError, "boundary handle"):
            inspect_native(drawing.bytes(), report)

    def test_extended_handle_uses_fixed_big_endian_bytes(self):
        self.assertEqual(extended_data([(1005, 0x1234)]), b"\x05" + struct.pack(">Q", 0x1234))


if __name__ == "__main__":
    unittest.main()
