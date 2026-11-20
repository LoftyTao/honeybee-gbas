from __future__ import annotations

import unittest
from pathlib import Path

from honeybee.face import Face
from honeybee.facetype import face_types
from honeybee.model import Model
from honeybee.room import Room
from ladybug_geometry.geometry3d import Face3D, Point3D, Polyface3D, Vector3D

from honeybee_gbas import layout
from honeybee_gbas import model as hb
from honeybee_gbas.layout import WallSegment


ROOT = Path(__file__).resolve().parents[1]


def wall_model(identifier, boundary, holes=None):
    face = Face(identifier, Face3D(tuple(Point3D(*point) for point in boundary), holes=[
        tuple(Point3D(*point) for point in hole) for hole in holes
    ] if holes else None), face_types.wall)
    return Model("model", [Room("room", [face])]), face


def coordinates(segments):
    return sorted(tuple(sorted(((round(segment.start.x, 6), round(segment.start.y, 6)),
                               (round(segment.end.x, 6), round(segment.end.y, 6)))))
                  for segment in segments)


class FloorSectionTestCase(unittest.TestCase):
    def test_box_surface_and_centerline(self):
        model = Model("model", [Room.from_box("room", 5, 4, 3)])
        surface = layout.floor_section(model, axis="surface", cut_height_mm=1200)
        self.assertEqual(coordinates(surface), [
            ((0.0, 0.0), (0.0, 4.0)),
            ((0.0, 0.0), (5.0, 0.0)),
            ((0.0, 4.0), (5.0, 4.0)),
            ((5.0, 0.0), (5.0, 4.0)),
        ])
        centerline = layout.floor_section(model, axis="centerline", cut_height_mm=1200)
        self.assertEqual(len(centerline), 4)
        for index, segment in enumerate(centerline):
            for point in (segment.start, segment.end):
                self.assertTrue(any(
                    other_index != index and layout.point_on_segment(point, other.start, other.end, 1e-7)
                    for other_index, other in enumerate(centerline)
                ))
        self.assertEqual(layout.floor_section(model, axis="surface", cut_height_mm=3001), [])
        self.assertEqual(len(layout.floor_section(model, axis="surface", cut_height_mm=3000)), 4)

    def test_closed_wedge_uses_true_triangle_sections(self):
        a, b, c, d, e, f = (
            Point3D(0, 0, 0), Point3D(4, 0, 0), Point3D(4, 4, 0),
            Point3D(0, 4, 0), Point3D(4, 4, 3), Point3D(4, 0, 3),
        )
        polyface = Polyface3D.from_faces([
            Face3D((a, d, c, b)), Face3D((a, b, f)), Face3D((d, e, c)),
            Face3D((b, c, e, f)), Face3D((a, f, e, d)),
        ], 0.001)
        room = Room.from_polyface3d("wedge", polyface)
        self.assertTrue(room.geometry.is_solid)
        sections = layout.floor_section(Model("model", [room]), axis="surface", cut_height_mm=1200)
        self.assertEqual(coordinates(sections), [
            ((1.6, 0.0), (4.0, 0.0)),
            ((1.6, 4.0), (4.0, 4.0)),
            ((4.0, 0.0), (4.0, 4.0)),
        ])

    def test_triangle_with_single_low_vertex(self):
        model, face = wall_model("triangle", [(0, 0, 0), (4, 0, 2), (0, 0, 2)])
        self.assertIsNone(hb.wall_bottom_edge(face))
        self.assertEqual(coordinates(layout.floor_section(model, axis="surface", cut_height_mm=1000)), [
            ((0.0, 0.0), (2.0, 0.0)),
        ])

    def test_hole_produces_disjoint_segments(self):
        model, _ = wall_model("wall", [(0, 0, 0), (10, 0, 0), (10, 0, 3), (0, 0, 3)], holes=[
            [(3, 0, 1), (7, 0, 1), (7, 0, 2), (3, 0, 2)],
        ])
        self.assertEqual(coordinates(layout.floor_section(model, axis="surface", cut_height_mm=1500)), [
            ((0.0, 0.0), (3.0, 0.0)), ((7.0, 0.0), (10.0, 0.0)),
        ])
        self.assertEqual(coordinates(layout.floor_section(model, axis="surface", cut_height_mm=500)), [
            ((0.0, 0.0), (10.0, 0.0)),
        ])
        self.assertEqual(coordinates(layout.floor_section(model, axis="surface", cut_height_mm=1000)), [
            ((0.0, 0.0), (10.0, 0.0)),
        ])

    def test_concave_wall_produces_disjoint_segments(self):
        model, _ = wall_model("wall", [
            (0, 0, 0), (10, 0, 0), (10, 0, 3), (7, 0, 3),
            (7, 0, 1), (3, 0, 1), (3, 0, 3), (0, 0, 3),
        ])
        self.assertEqual(coordinates(layout.floor_section(model, axis="surface", cut_height_mm=2000)), [
            ((0.0, 0.0), (3.0, 0.0)), ((7.0, 0.0), (10.0, 0.0)),
        ])

    def test_diagonal_wall_hole(self):
        model, _ = wall_model("wall", [(0, 0, 0), (10, 10, 0), (10, 10, 3), (0, 0, 3)], holes=[
            [(3, 3, 1), (7, 7, 1), (7, 7, 2), (3, 3, 2)],
        ])
        self.assertEqual(coordinates(layout.floor_section(model, axis="surface", cut_height_mm=1500)), [
            ((0.0, 0.0), (3.0, 3.0)), ((7.0, 7.0), (10.0, 10.0)),
        ])

    def test_inclined_wall_section_and_centerline(self):
        model, face = wall_model("wall", [(0, 0, 0), (4, 0, 0), (4, 2, 4), (0, 2, 4)])
        surface = layout.floor_section(model, axis="surface", cut_height_mm=1200)
        self.assertEqual(coordinates(surface), [((0.0, 0.6), (4.0, 0.6))])
        centerline = layout.floor_section(model, axis="centerline", cut_height_mm=1200)
        normal = face.normal
        offset = hb.face_thickness_m(face) / 2 * normal.y / (normal.x ** 2 + normal.y ** 2)
        self.assertAlmostEqual(centerline[0].start.y, 0.6 + offset)
        self.assertEqual(centerline[0].start.z, 0.0)

    def test_partial_overlapping_sections_are_deduplicated(self):
        first, _ = wall_model("first", [(0, 0, 0), (10, 0, 0), (10, 0, 3), (0, 0, 3)])
        second, _ = wall_model("second", [(15, 0, 0), (5, 0, 0), (5, 0, 3), (15, 0, 3)])
        first.rooms[0].identifier = "first_room"
        second.rooms[0].identifier = "second_room"
        model = Model("combined", [first.rooms[0], second.rooms[0]])
        sections = layout.floor_section(model, axis="surface", cut_height_mm=1200)
        self.assertAlmostEqual(sum(segment.length for segment in sections), 15.0)
        self.assertEqual(coordinates(sections), [
            ((0.0, 0.0), (10.0, 0.0)), ((10.0, 0.0), (15.0, 0.0)),
        ])

    def test_deduplication_preserves_gap_above_tolerance(self):
        first, _ = wall_model("first", [(0, 0, 0), (1, 0, 0), (1, 0, 3), (0, 0, 3)])
        second, _ = wall_model("second", [(1.00015, 0, 0), (2, 0, 0), (2, 0, 3), (1.00015, 0, 3)])
        first.rooms[0].identifier = "first_room"
        second.rooms[0].identifier = "second_room"
        sections = layout.floor_section(Model("combined", [first.rooms[0], second.rooms[0]]), axis="surface")
        self.assertEqual(coordinates(sections), [
            ((0.0, 0.0), (1.0, 0.0)), ((1.00015, 0.0), (2.0, 0.0)),
        ])

    def test_distinct_sections_with_same_bottom_edge(self):
        first, _ = wall_model("first", [(0, 0, 0), (4, 0, 0), (4, 2, 4), (0, 2, 4)])
        second, _ = wall_model("second", [(4, 0, 0), (0, 0, 0), (0, -2, 4), (4, -2, 4)])
        first.rooms[0].identifier = "first_room"
        second.rooms[0].identifier = "second_room"
        model = Model("combined", [first.rooms[0], second.rooms[0]])
        self.assertEqual(coordinates(layout.floor_section(model, axis="surface", cut_height_mm=1200)), [
            ((0.0, -0.6), (4.0, -0.6)), ((0.0, 0.6), (4.0, 0.6)),
        ])

    def test_basic_sample_requested_height(self):
        model = Model.from_file(str(ROOT / "samples/basic_sample_model/basic_sample_model.hbjson"))
        identifier = "Room_e6ac360b-aaed-4c3b-a130-36b4c2ac9d13-000d14bf..Face9_Above"
        low = layout.floor_section(model, "Level 1", "surface", 1200)
        high = layout.floor_section(model, "Level 1", "surface", 2800)
        self.assertFalse(any(segment.face_identifier == identifier for segment in low))
        self.assertTrue(any(segment.face_identifier == identifier for segment in high))

    def test_shared_wall_is_emitted_once(self):
        left = Room.from_box("left", 5, 4, 3)
        right = Room.from_box("right", 5, 4, 3, origin=Point3D(5, 0, 0))
        Room.solve_adjacency([left, right], 0.01)
        sections = layout.floor_section(Model("model", [left, right]), axis="surface")
        self.assertEqual(len(sections), 7)
        self.assertEqual(len([segment for segment in sections if not segment.exterior]), 1)

    def test_geometric_shared_wall_keeps_its_axis_with_outdoor_conditions(self):
        left = Room.from_box("left", 5, 4, 3)
        right = Room.from_box("right", 5, 4, 3, origin=Point3D(5, 0, 0))
        model = Model("two_rooms", [left, right])
        sections = layout.floor_section(model, axis="centerline")
        shared = [segment for segment in sections if not segment.exterior]
        self.assertEqual(len(shared), 1)
        self.assertAlmostEqual(shared[0].start.x, 5.0)
        self.assertAlmostEqual(shared[0].end.x, 5.0)
        self.assertTrue(all(str(face.boundary_condition) == "Outdoors" for room in model.rooms for face in room.faces if str(face.type) == "Wall"))

    def test_shared_thickness_applies_only_to_opposed_rooms_in_both_axis_modes(self):
        left = Room.from_box("left", 5, 4, 3)
        right = Room.from_box("right", 5, 4, 3, origin=Point3D(5, 0, 0))
        model = Model("two_rooms", [left, right])
        for axis in ("surface", "centerline"):
            with self.subTest(axis=axis):
                default = layout.floor_section(model, axis=axis)
                self.assertEqual(default, layout.floor_section(model, axis=axis, shared_wall_thickness_mm=None))
                actual = layout.floor_section(model, axis=axis, shared_wall_thickness_mm=125.4)
                walls = [segment for segment in actual if not isinstance(segment, layout.AxisConnector)]
                shared = [segment for segment in walls if abs(segment.thickness_m - 0.1254) < 1e-9]
                self.assertEqual(len(shared), 1)
                self.assertAlmostEqual(shared[0].thickness_m, 0.1254)
                self.assertAlmostEqual(shared[0].start.x, 5.0)
                self.assertAlmostEqual(shared[0].end.x, 5.0)
                self.assertEqual(sorted(round(point.y, 6) for point in (shared[0].start, shared[0].end)), [0.0, 4.0])
                self.assertTrue(all(abs(segment.thickness_m - 0.3627) < 1e-9 for segment in walls if segment is not shared[0]))
                self.assertAlmostEqual(sum(segment.length for segment in default), sum(segment.length for segment in actual))
                for face in {segment.face_identifier for segment in default}:
                    original = [segment for segment in default if segment.face_identifier == face]
                    pieces = [segment for segment in actual if segment.face_identifier == face]
                    self.assertAlmostEqual(sum(segment.length for segment in original), sum(segment.length for segment in pieces))
                    for dimension in ("x", "y"):
                        source_points = [getattr(point, dimension) for segment in original for point in (segment.start, segment.end)]
                        actual_points = [getattr(point, dimension) for segment in pieces for point in (segment.start, segment.end)]
                        self.assertAlmostEqual(min(source_points), min(actual_points))
                        self.assertAlmostEqual(max(source_points), max(actual_points))
        self.assertTrue(all(segment.exterior for segment in layout.floor_section(model, axis="surface")))

    def test_partial_shared_interval_retains_single_sided_thickness(self):
        normal = Vector3D(0, 1, 0)
        first = WallSegment(Point3D(0, 0, 0), Point3D(10, 0, 0), 0.3627, 3, False, "first", "first_face", normal)
        second = WallSegment(Point3D(8, 0, 0), Point3D(2, 0, 0), 0.3627, 3, False, "second", "second_face", -normal)
        sections = layout._deduplicate_sections([first, second], 0.0001, infer_shared=True, shared_thickness_m=0.1254)
        for actual in (sections, layout.wall_axes(sections, 1000)):
            self.assertEqual(coordinates(actual), [((0.0, 0.0), (2.0, 0.0)), ((2.0, 0.0), (8.0, 0.0)), ((8.0, 0.0), (10.0, 0.0))])
            self.assertEqual([round(segment.thickness_m, 4) for segment in actual], [0.3627, 0.1254, 0.3627])
            self.assertTrue(all(segment.face_identifier == "first_face" and not segment.exterior for segment in actual))

    def test_surface_option_splits_partial_geometric_adjacency(self):
        first, _ = wall_model("first", [(0, 0, 0), (10, 0, 0), (10, 0, 3), (0, 0, 3)])
        second, _ = wall_model("second", [(15, 0, 0), (5, 0, 0), (5, 0, 3), (15, 0, 3)])
        first.rooms[0].identifier = "first_room"
        second.rooms[0].identifier = "second_room"
        model = Model("combined", [first.rooms[0], second.rooms[0]])
        sections = layout.floor_section(model, axis="surface", shared_wall_thickness_mm=125.4)
        self.assertEqual(coordinates(sections), [((0.0, 0.0), (5.0, 0.0)), ((5.0, 0.0), (10.0, 0.0)), ((10.0, 0.0), (15.0, 0.0))])
        self.assertEqual([round(segment.thickness_m, 4) for segment in sections], [0.3627, 0.1254, 0.3627])
        self.assertEqual([segment.exterior for segment in sections], [True, False, True])

    def test_centerline_miters_keep_single_sided_source_thickness(self):
        for height, shared_face in ((3, "left_Right"), (4, "right_Left")):
            with self.subTest(height=height):
                left = Room.from_box("left", 5, 10, 3)
                right = Room.from_box("right", 5, 5, height, origin=Point3D(5, 2.5, 0))
                model = Model("partial", [left, right])
                sections = layout.floor_section(model, axis="centerline", shared_wall_thickness_mm=125.4)
                walls = [segment for segment in sections if segment.face_identifier in ("left_Right", "right_Left") and not isinstance(segment, layout.AxisConnector)]
                thin = [segment for segment in walls if abs(segment.thickness_m - 0.1254) < 1e-9]
                source_thickness = [segment for segment in walls if abs(segment.thickness_m - 0.3627) < 1e-9]
                self.assertEqual(len(thin), 1)
                self.assertEqual(thin[0].face_identifier, shared_face)
                self.assertEqual(len(source_thickness), 4)
                self.assertTrue(all(segment.face_identifier == "left_Right" for segment in source_thickness))
                self.assertEqual(coordinates(thin), [((5.0, 2.5), (5.0, 7.5))])
                self.assertTrue(any(abs(segment.start.x - 5.0) < 1e-9 and min(segment.start.y, segment.end.y) < 2.5 for segment in source_thickness))
                self.assertTrue(any(abs(segment.start.x - 5.0) < 1e-9 and max(segment.start.y, segment.end.y) > 7.5 for segment in source_thickness))
                connectors = [segment for segment in sections if isinstance(segment, layout.AxisConnector)]
                self.assertEqual(connectors, [])
                walls = [segment for segment in sections if not isinstance(segment, layout.AxisConnector)]
                for index, wall in enumerate(walls):
                    for point in (wall.start, wall.end):
                        self.assertTrue(any(
                            other_index != index and layout.point_on_segment(point, other.start, other.end, 1e-7)
                            for other_index, other in enumerate(walls)
                        ))

    def test_same_room_or_same_direction_does_not_define_a_shared_interval(self):
        normal = Vector3D(0, 1, 0)
        first = WallSegment(Point3D(0, 0, 0), Point3D(10, 0, 0), 0.3627, 3, False, "first", "first_face", normal)
        for room, other_normal in (("first", -normal), ("second", normal)):
            with self.subTest(room=room, normal=other_normal):
                other = WallSegment(Point3D(8, 0, 0), Point3D(2, 0, 0), 0.3627, 3, False, room, "other_face", other_normal)
                sections = layout._deduplicate_sections([first, other], 0.0001, infer_shared=True, shared_thickness_m=0.1254)
                self.assertEqual(len(sections), 1)
                self.assertAlmostEqual(sections[0].thickness_m, 0.3627)

    def test_shared_pair_requires_mutual_coplanarity_within_tolerance(self):
        normal = Vector3D(0, 1, 0)
        segments = [
            WallSegment(Point3D(0, y, 0), Point3D(10, y, 0), 0.3627, 3, False, room, face, direction)
            for y, room, face, direction in (
                (0.0, "first", "first_face", normal),
                (0.00009, "second", "second_face", normal),
                (-0.00009, "first", "third_face", -normal),
            )
        ]
        sections = layout._deduplicate_sections(segments, 0.0001, infer_shared=True, shared_thickness_m=0.1254)
        self.assertEqual(len(sections), 1)
        self.assertAlmostEqual(sections[0].thickness_m, 0.3627)

    def test_shared_thickness_uses_actual_section_elevation(self):
        lower = Room.from_box("lower", 5, 4, 3)
        upper = Room.from_box("upper", 5, 4, 3, origin=Point3D(5, 0, 4))
        lower.story = upper.story = "combined"
        model = Model("different_elevations", [lower, upper])
        for axis in ("surface", "centerline"):
            with self.subTest(axis=axis):
                sections = layout.floor_section(model, "combined", axis, shared_wall_thickness_mm=125.4)
                walls = [segment for segment in sections if not isinstance(segment, layout.AxisConnector)]
                self.assertEqual(len(walls), 8)
                self.assertTrue(all(segment.exterior and abs(segment.thickness_m - 0.3627) < 1e-9 for segment in walls))

    def test_shared_thickness_is_unit_invariant(self):
        left = Room.from_box("left", 5, 4, 3)
        right = Room.from_box("right", 5, 4, 3, origin=Point3D(5, 0, 0))
        model = Model("two_rooms", [left, right])
        metric = layout.floor_section(model, axis="centerline", shared_wall_thickness_mm=125.4)
        model.convert_to_units("Feet")
        imperial = layout.floor_section(model, axis="centerline", shared_wall_thickness_mm=125.4)
        self.assertEqual(len(metric), len(imperial))
        for first, second in zip(metric, imperial):
            self.assertAlmostEqual(first.thickness_m, second.thickness_m)
            for metric_point, imperial_point in ((first.start, second.start), (first.end, second.end)):
                self.assertAlmostEqual(metric_point.x * 1000, imperial_point.x * 304.8)
                self.assertAlmostEqual(metric_point.y * 1000, imperial_point.y * 304.8)

    def test_shared_thickness_requires_positive_finite_numeric_value(self):
        model = Model("model", [Room.from_box("room")])
        for value in (True, False, "125.4", object()):
            with self.subTest(value=value), self.assertRaises(TypeError):
                layout.floor_section(model, shared_wall_thickness_mm=value)
        for value in (0, -1, float("nan"), float("inf"), -float("inf"), 5e-324, 10 ** 1000):
            with self.subTest(value=value), self.assertRaises(ValueError):
                layout.floor_section(model, shared_wall_thickness_mm=value)

    def test_section_is_invariant_under_unit_conversion(self):
        room = Room.from_box("room", 5, 4, 3, origin=Point3D(2, 3, 6))
        room.story = "upper"
        model = Model("model", [room])
        metric = layout.floor_section(model, "upper", "centerline", 1200)
        model.convert_to_units("Feet")
        imperial = layout.floor_section(model, "upper", "centerline", 1200)
        self.assertEqual(len(metric), len(imperial))
        for first, second in zip(metric, imperial):
            for metric_point, imperial_point in ((first.start, second.start), (first.end, second.end)):
                self.assertAlmostEqual(metric_point.x * 1000, imperial_point.x * 304.8)
                self.assertAlmostEqual(metric_point.y * 1000, imperial_point.y * 304.8)

    def test_section_height_adapts_only_short_rooms(self):
        short = Room.from_box("short", 4, 3, 1, origin=Point3D(0, 0, 2))
        tall = Room.from_box("tall", 4, 3, 3, origin=Point3D(10, 0, 2))
        other = Room.from_box("other", 4, 3, 0.8, origin=Point3D(20, 0, 2))
        short.story = tall.story = "mixed"
        other.story = "other"
        model = Model("mixed", [short, tall, other])
        before = model.to_dict()
        default = layout.floor_section(model, "mixed", "surface", 1200)
        adapted = layout.floor_section(model, "mixed", "surface", 1200, adapt_cut_height=True)
        self.assertEqual(len(default), 4)
        self.assertEqual({wall.room_identifier for wall in default}, {"tall"})
        self.assertEqual(len(adapted), 8)
        self.assertEqual({wall.room_identifier for wall in adapted}, {"short", "tall"})
        self.assertCountEqual([wall for wall in adapted if wall.room_identifier == "tall"], default)
        self.assertEqual(layout.room_section_height_mm(short, 1000, 1200, False), 1200)
        self.assertEqual(layout.room_section_height_mm(short, 1000, 1200, True), 500)
        self.assertEqual(layout.room_section_height_mm(tall, 1000, 1200, True), 1200)
        self.assertEqual(model.to_dict(), before)

    def test_adaptive_height_combines_with_shared_thickness(self):
        short = Room.from_box("short", 4, 3, 1)
        tall = Room.from_box("tall", 4, 3, 3, origin=Point3D(4, 0, 0))
        model = Model("mixed", [short, tall])
        sections = layout.floor_section(model, axis="surface", shared_wall_thickness_mm=125.4, adapt_cut_height=True)
        self.assertEqual(len(sections), 7)
        shared = [segment for segment in sections if abs(segment.thickness_m - 0.1254) < 1e-9]
        self.assertEqual(len(shared), 1)
        self.assertAlmostEqual(shared[0].start.x, 4)
        self.assertAlmostEqual(shared[0].end.x, 4)

    def test_adaptive_section_is_invariant_under_unit_conversion(self):
        short = Room.from_box("short", 4, 3, 1, origin=Point3D(2, 3, 6))
        tall = Room.from_box("tall", 4, 3, 3, origin=Point3D(10, 3, 6))
        model = Model("mixed", [short, tall])
        metric = layout.floor_section(model, axis="centerline", adapt_cut_height=True)
        model.convert_to_units("Feet")
        imperial = layout.floor_section(model, axis="centerline", adapt_cut_height=True)
        self.assertEqual(len(metric), len(imperial))
        for first, second in zip(metric, imperial):
            for first_point, second_point in ((first.start, second.start), (first.end, second.end)):
                self.assertAlmostEqual(first_point.x * 1000, second_point.x * 304.8)
                self.assertAlmostEqual(first_point.y * 1000, second_point.y * 304.8)
        self.assertAlmostEqual(layout.room_section_height_mm(short, 304.8, 1200, True), 500)

    def test_adaptation_requires_boolean(self):
        model = Model("model", [Room.from_box("room")])
        for value in (None, 1, "true"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                layout.floor_section(model, adapt_cut_height=value)

    def test_invalid_heights(self):
        model = Model("model", [Room.from_box("room")])
        for height in (0, -1, float("nan"), float("inf")):
            with self.subTest(height=height), self.assertRaises(ValueError):
                layout.floor_section(model, cut_height_mm=height)


class AxisConnectorTestCase(unittest.TestCase):
    def test_single_sided_wall_matching_override_keeps_its_source_thickness(self):
        first = WallSegment(Point3D(0, 0, 0), Point3D(5, 0, 0), 0.1254, 3, False, "a", "first", Vector3D(0, -1, 0))
        second = WallSegment(Point3D(5, 0, 0), Point3D(10, 0, 0), 0.3627, 3, True, "b", "second", Vector3D(0, -1, 0))
        perpendicular = WallSegment(Point3D(5, 0, 0), Point3D(5, 5, 0), 0.3627, 3, True, "c", "corner", Vector3D(1, 0, 0))
        walls = layout.wall_axes([first, second, perpendicular], 1000, shared_thickness_m=0.1254, source_thicknesses={"first": 0.1254, "second": 0.3627, "corner": 0.3627})
        first_walls = [wall for wall in walls if wall.face_identifier == "first" and not isinstance(wall, layout.AxisConnector)]
        self.assertEqual(len(first_walls), 1)
        self.assertAlmostEqual(first_walls[0].thickness_m, 0.1254)
        self.assertGreater(first_walls[0].end.x, 5.0)

    def test_post_union_preserves_projected_source_elevations(self):
        first = WallSegment(Point3D(0, 0, 0), Point3D(5, 0, 0), 0.2, 3, True, "lower", "first", Vector3D(0, 1, 0))
        second = WallSegment(Point3D(0, 0.1, 0), Point3D(5, 0.1, 0), 0.1254, 3, False, "upper", "second", Vector3D(0, -1, 0))
        sections = layout.wall_axes([first, second], 1000, source_elevations={"lower": 0, "upper": 3})
        self.assertEqual(len(sections), 2)
        self.assertEqual({segment.face_identifier for segment in sections}, {"first", "second"})
        self.assertTrue(all(abs(segment.start.y - 0.1) < 1e-8 for segment in sections))

    def test_collinear_exterior_interior_connector(self):
        normal = Vector3D(0, -1, 0)
        first = WallSegment(Point3D(0, 0, 0), Point3D(5, 0, 0), 0.4, 3, True, "room", "outside", normal)
        second = WallSegment(Point3D(5, 0, 0), Point3D(10, 0, 0), 0.1254, 3, False, "room", "inside", normal)
        sections = layout.wall_axes([first, second], 1000, "centerline")
        self.assertEqual(len(sections), 3)
        self.assertIsInstance(sections[2], layout.AxisConnector)
        self.assertEqual(sections[2].adjoining_face_identifier, "inside")
        self.assertAlmostEqual(sections[2].length, 0.2)
        self.assertTrue(layout.point_on_segment(sections[0].end, sections[2].start, sections[2].end, 1e-7))
        self.assertTrue(layout.point_on_segment(sections[1].start, sections[2].start, sections[2].end, 1e-7))
        self.assertEqual(layout.wall_axes([first, second], 1000, "surface"), [first, second])

    def test_collinear_different_exterior_thicknesses(self):
        normal = Vector3D(0, -1, 0)
        first = WallSegment(Point3D(0, 0, 0), Point3D(5, 0, 0), 0.2, 3, True, "room", "first", normal)
        second = WallSegment(Point3D(5, 0, 0), Point3D(10, 0, 0), 0.4, 3, True, "room", "second", normal)
        sections = layout.wall_axes([first, second], 1000)
        self.assertEqual(len(sections), 3)
        self.assertAlmostEqual(sections[2].length, 0.1)
        self.assertAlmostEqual(sections[2].start.x, 5.0)
        self.assertAlmostEqual(sections[2].end.x, 5.0)

    def test_separate_elevations_get_separate_connectors(self):
        normal = Vector3D(0, -1, 0)
        segments = [
            WallSegment(Point3D(0, 0, z), Point3D(5, 0, z), 0.4, 3, True, f"room{z}", f"outside{z}", normal)
            for z in (0, 3)
        ] + [
            WallSegment(Point3D(5, 0, z), Point3D(10, 0, z), 0.1254, 3, False, f"room{z}", f"inside{z}", normal)
            for z in (0, 3)
        ]
        sections = layout.wall_axes(segments, 1000)
        connectors = [segment for segment in sections if isinstance(segment, layout.AxisConnector)]
        walls = [segment for segment in sections if not isinstance(segment, layout.AxisConnector)]
        self.assertEqual(len(walls), 4)
        self.assertEqual({segment.start.z for segment in walls}, {0, 3})
        self.assertEqual(len(connectors), 2)
        self.assertEqual({segment.start.z for segment in connectors}, {0, 3})

    def test_real_exterior_interior_transition(self):
        model = Model.from_file(str(ROOT / "samples/2023_rac_advanced_sample_project/2023_rac_advanced_sample_project.hbjson"))
        sections = layout.floor_section(model, "02 - Floor", "centerline", 1200)
        first = next(segment for segment in sections if segment.face_identifier.endswith("0003483f..Face1_0") and segment.length > 1)
        second = next(segment for segment in sections if segment.face_identifier.endswith("0003483f..Face1_2") and segment.length > 1)
        endpoints = [(first.start, second.start), (first.start, second.end), (first.end, second.start), (first.end, second.end)]
        point_a, point_b = min(endpoints, key=lambda pair: pair[0].distance_to_point(pair[1]))
        self.assertAlmostEqual(point_a.distance_to_point(point_b) * 304.8, 181.35, places=2)
        self.assertTrue(any(
            isinstance(segment, layout.AxisConnector)
            and layout.point_on_segment(point_a, segment.start, segment.end, 1e-6)
            and layout.point_on_segment(point_b, segment.start, segment.end, 1e-6)
            for segment in sections
        ))


if __name__ == "__main__":
    unittest.main()
