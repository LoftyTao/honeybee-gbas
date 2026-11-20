from __future__ import annotations

import copy
import unittest

from honeybee.aperture import Aperture
from honeybee.boundarycondition import boundary_conditions
from honeybee.model import Model
from honeybee.room import Room
from ladybug_geometry.geometry3d import Face3D, Point3D

from honeybee_gbas.dwg import build
from honeybee_gbas.dwg.metadata import geometry_model, planar_shared_walls
from honeybee_gbas.dwg.validation import inspect_export, inspect_native, inspect_quality, validate
from honeybee_gbas.dwg.walls import plan_wall_conflicts, separate_plan_wall_axes


def near_planar_pair(displacement=0.0002, window=False):
    first = Room.from_box("First", 5, 4, 3)
    second = Room.from_box("Second", 5, 4, 3, origin=Point3D(5, 0, 0))
    first.story = second.story = "Occupied"
    if window:
        face = next(face for face in first.faces if face.normal.x > 0.99)
        geometry = Face3D(tuple(Point3D(5, y, z) for y, z in ((1, 1), (3, 1), (3, 2), (1, 2))))
        face.add_aperture(Aperture("SharedWindow", geometry))
        neighbor = next(face for face in second.faces if face.normal.x < -0.99)
        neighbor.add_aperture(Aperture("SharedWindowOther", geometry.flip()))
    Room.solve_adjacency([first, second], 1e-6)
    model = Model("NearPlanar", [first, second])
    data = model.to_dict()
    face = next(face for face in data["rooms"][0]["faces"] if face["boundary_condition"]["type"] == "Surface")
    boundary = [list(point) for point in face["geometry"]["boundary"]]
    boundary[1][0] += displacement
    face["geometry"]["boundary"] = boundary
    face["geometry"].pop("plane", None)
    data["tolerance"] = 0
    result = Model.from_dict(data)
    result.tolerance = model.tolerance
    return result


class GeometryPreparationTest(unittest.TestCase):
    def test_reciprocal_shared_plane_repair_preserves_source_and_properties(self):
        model = near_planar_pair(window=True)
        original = model.to_dict()
        repairs = []
        working, semantics = geometry_model(model, repairs)
        self.assertEqual(model.to_dict(), original)
        self.assertEqual(len(repairs), 1)
        faces = {face.identifier: face for room in working.rooms for face in room.faces}
        first, second = (faces[identifier] for identifier in repairs[0]["source_faces"])
        self.assertLess(max(first.geometry.plane.distance_to_point(point) for point in first.geometry.boundary), 1e-12)
        self.assertLess(first.normal.dot(second.normal), -0.999999999)
        self.assertTrue(first.geometry.is_geometrically_equivalent(second.geometry, 1e-9))
        self.assertEqual(working.tolerance, model.tolerance)
        self.assertEqual(semantics[first.identifier], "Physical")
        source = next(face for room in model.rooms for face in room.faces if face.identifier == first.identifier)
        self.assertEqual(first.properties.energy.to_dict(), source.properties.energy.to_dict())
        self.assertEqual({item.identifier for room in working.rooms for face in room.faces for item in face.apertures},
                         {item.identifier for room in model.rooms for face in room.faces for item in face.apertures})

    def test_exact_pairs_and_excess_displacement_keep_the_working_geometry(self):
        for displacement in (0, 0.003):
            model = near_planar_pair(displacement)
            working, repairs = planar_shared_walls(model)
            self.assertIs(working, model)
            self.assertEqual(repairs, [])

    def test_repair_requires_reciprocal_source_adjacency(self):
        model = near_planar_pair()
        face = next(face for face in model.rooms[1].faces if str(face.boundary_condition) == "Surface")
        face.boundary_condition = boundary_conditions.outdoors
        working, repairs = planar_shared_walls(model)
        self.assertIs(working, model)
        self.assertEqual(repairs, [])

    def test_product_build_records_the_bounded_shared_plane_repair(self):
        model = near_planar_pair()
        drawing, report = build(model)
        validate(model, drawing.bytes(), report)
        self.assertEqual(len(report["source_geometry_repairs"]), 1)
        self.assertEqual(report["wall_panels"], [])
        self.assertTrue(report["native_quality"]["same_storey_wall_uniqueness_passed"])
        self.assertEqual(report["source_geometry_validation"]["summary"]["review_items"], 0)


class NativePlanCompatibilityTest(unittest.TestCase):
    def fixture(self):
        tall = Room.from_box("Tall", 5, 4, 6)
        low = Room.from_box("Lower", 5, 4, 3, origin=Point3D(5, 0, 0))
        tall.story = low.story = "Occupied"
        return Model("StackedWalls", [tall, low])

    def test_stacked_wall_axes_keep_bounded_source_geometry_and_distinct_heights(self):
        model = self.fixture()
        original = model.to_dict()
        drawing, report = build(model, inclined_carriers=True)
        validate(model, drawing.bytes(), report)
        self.assertEqual(model.to_dict(), original)
        self.assertEqual(plan_wall_conflicts(report["walls"]), [])
        repairs = report["native_plan_compatibility"]["records"]
        self.assertEqual(len(repairs), 1)
        self.assertAlmostEqual(repairs[0]["maximum_displacement_mm"], 0.05)
        self.assertEqual(report["source_geometry_validation"]["summary"]["review_items"], 0)
        self.assertTrue(inspect_export(report)["passed"])

    def test_rotated_elevated_openings_follow_the_final_host_axis(self):
        for angle in (0, 90, 180, 270, 33):
            with self.subTest(angle=angle):
                model = self.fixture()
                face = next(face for face in model.rooms[0].faces if face.normal.x > 0.99)
                face.add_aperture(Aperture("UpperWindow", Face3D(tuple(Point3D(5, y, z) for y, z in ((1, 4), (3, 4), (2.5, 5), (1, 5))))))
                model.rotate_xy(angle, Point3D())
                original = model.to_dict()
                drawing, report = build(model, inclined_carriers=True)
                validate(model, drawing.bytes(), report)
                self.assertEqual(model.to_dict(), original)
                self.assertEqual(len(report["openings"]), 1)
                moved = {row["handle"] for row in report["native_plan_compatibility"]["records"]}
                self.assertIn(report["openings"][0]["wall"], moved)
                self.assertTrue(report["native_quality"]["opening_bindings_passed"])
                self.assertLess(report["native_quality"]["opening_bindings"][0]["axis_distance_mm"], 1e-8)
                self.assertTrue(inspect_export(report)["passed"])

    def test_projected_conflicts_use_the_native_frame_and_absolute_axis_direction(self):
        first = {"handle": 1, "identifier": "First", "story": "Occupied", "start_mm": (0, 0, 0), "end_mm": (4000, 0, 0), "height_mm": 500}
        second = {**first, "handle": 2, "identifier": "Second", "start_mm": (4000, 0, 1000), "end_mm": (0, 0, 1000)}
        self.assertEqual(len(plan_wall_conflicts([first, second])), 1)
        separate_plan_wall_axes({"walls": [first, second]})
        self.assertEqual(plan_wall_conflicts([first, second]), [])
        self.assertEqual(second["start_mm"], (4000.0, 0.05, 1000.0))
        second.update(story="Other", start_mm=(4000, 0, 1000), end_mm=(0, 0, 1000))
        self.assertEqual(plan_wall_conflicts([first, second]), [])

    def test_displacement_limit_is_checked_before_geometry_changes(self):
        rows = [{"handle": index, "identifier": f"Wall{index}", "story": "Occupied", "start_mm": (0, 0, index * 1000),
                 "end_mm": (4000, 0, index * 1000), "height_mm": 500} for index in range(7)]
        original = copy.deepcopy(rows)
        with self.assertRaisesRegex(ValueError, "within 0.25 millimeters"):
            separate_plan_wall_axes({"walls": rows})
        self.assertEqual(rows, original)

    def test_physical_surface_overlap_retains_its_geometry_review(self):
        first = {"handle": 1, "identifier": "First", "story": "Occupied", "start_mm": (0, 0, 0), "end_mm": (4000, 0, 0), "height_mm": 3000}
        second = {**first, "handle": 2, "identifier": "Second"}
        with self.assertRaisesRegex(ValueError, "disjoint physical wall"):
            separate_plan_wall_axes({"walls": [first, second]})

    def test_independent_quality_detects_plan_overlap_with_disjoint_vertical_spans(self):
        model = self.fixture()
        drawing, report = build(model, inclined_carriers=True)
        validate(model, drawing.bytes(), report)
        readback = inspect_native(drawing.bytes(), report, include_geometry=True)
        changed = copy.deepcopy(readback)
        repair = report["native_plan_compatibility"]["records"][0]
        row = changed["geometry"]["entities"][repair["handle"]]
        row["start_mm"], row["end_mm"] = repair["source_start_mm"], repair["source_end_mm"]
        quality = inspect_quality(report, changed)
        self.assertTrue(quality["same_storey_wall_uniqueness_passed"])
        self.assertFalse(quality["native_plan_wall_uniqueness_passed"])
        report["native_quality"] = quality
        self.assertFalse(inspect_export(report)["required_checks"]["native_projected_wall_axis_uniqueness"])


if __name__ == "__main__":
    unittest.main()
