from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from pathlib import Path

from honeybee.aperture import Aperture
from honeybee.face import Face
from honeybee.model import Model
from honeybee.room import Room
from honeybee.units import conversion_factor_to_meters
from ladybug_geometry.geometry3d import Face3D, Plane, Point3D, Polyface3D, Vector3D
from shapely.geometry import Polygon


ROOT = Path(__file__).resolve().parents[1]

from honeybee_gbas.dwg.geometry import OVERLAY_PRECISION_MM, _Surface, _union_surfaces, compare_model_geometry, coverage


def observations(model):
    scale = conversion_factor_to_meters(model.units) * 1000

    def record(item, kind, host_identifier=""):
        return {
            "identifier": item.identifier,
            "kind": kind,
            "host_identifier": host_identifier,
            "boundary_mm": [[coordinate * scale for coordinate in point] for point in item.geometry.boundary],
            "holes_mm": [[[coordinate * scale for coordinate in point] for point in hole] for hole in item.geometry.holes or ()],
        }

    return [{
        "identifier": room.identifier,
        "surfaces": [record(face, str(face.type)) for face in room.faces],
        "openings": [record(child, kind, face.identifier) for face in room.faces for kind, children in (("Window", face.apertures), ("Door", face.doors)) for child in children],
    } for room in model.rooms]


def box_model(identifier="Box", width=10000, depth=8000, height=3000):
    return Model("Geometry", [Room.from_box(identifier, width, depth, height)], units="Millimeters")


def sloped_model():
    def top(point):
        return Point3D(point.x, point.y, point.z + point.x * 0.25 if point.z > 0 else point.z)

    original = Room.from_box("Sloped", 10000, 8000, 3000)
    faces = [Face(face.identifier, Face3D(tuple(top(point) for point in face.geometry.boundary)), face.type) for face in original.faces]
    room = Room("Sloped", faces)
    roof = next(face for face in room.faces if str(face.type) == "RoofCeiling")
    aperture_points = [Point3D(x, y, 3000 + x * 0.25) for x, y in ((2000, 2000), (3000, 2000), (3000, 3000), (2000, 3000))]
    roof.add_aperture(Aperture("Skylight", Face3D(aperture_points)))
    return Model("SlopedGeometry", [room], units="Millimeters")


class OfflineDwgGeometryTest(unittest.TestCase):
    def test_coplanar_native_tiles_share_roundoff_vertices_and_keep_a_resolved_slit(self):
        plane = Plane(Vector3D(0, 0, 1), Point3D(0, 0, 0))
        seam = 1600 + OVERLAY_PRECISION_MM / 2
        def tile(name, low, high):
            vertices = tuple(Point3D(x, y, 0) for x,y in ((low,0),(high,0),(high,100),(low,100)))
            return _Surface(name, "Floor", Face3D(vertices, plane))
        parts = [tile("left", 0, math.nextafter(seam, -math.inf)), tile("right", math.nextafter(seam, math.inf), 3000)]
        joined, = _union_surfaces(parts, 1.0)
        self.assertAlmostEqual(joined.geometry.area, 300000)
        separated = _union_surfaces([tile("left", 0, seam-1e-5), tile("right", seam+1e-5, 3000)], 1.0)
        self.assertEqual(len(separated), 2)

    def assertPassed(self, report):
        self.assertTrue(report["passed"], json.dumps(report["issues"], ensure_ascii=False))
        self.assertEqual(report["status"], "passed")
        json.dumps(report, allow_nan=False)

    def assertReview(self, report):
        self.assertFalse(report["passed"])
        self.assertEqual(report["status"], "geometry_review_required")
        json.dumps(report, allow_nan=False)

    def test_complete_box_has_3d_coverage_closed_edges_and_volume(self):
        model = box_model()
        observed = observations(model)
        before = copy.deepcopy(observed)
        report = compare_model_geometry(model, observed)
        self.assertPassed(report)
        room = report["rooms"][0]
        for kind in ("Floor", "RoofCeiling", "Wall"):
            self.assertAlmostEqual(room["surface_kinds"][kind]["coverage"], 1)
        for key in ("source_shell", "observed_shell"):
            self.assertTrue(room[key]["closed"])
            self.assertEqual(room[key]["edge_pairing"]["naked_edge_count"], 0)
            self.assertEqual(room[key]["boundary_loops"], [])
            self.assertEqual(room[key]["edge_pairing"]["paired_edge_count"], 12)
        self.assertAlmostEqual(room["volume"]["source_m3"], 240)
        self.assertAlmostEqual(room["volume"]["observed_m3"], 240)
        self.assertEqual(observed, before)

    def test_empty_observation_has_explicit_review(self):
        report = compare_model_geometry(box_model(), [])
        self.assertReview(report)
        self.assertIn("observed_rooms_empty", [issue["code"] for issue in report["issues"]])
        self.assertFalse(report["summary"]["volume_comparison_available"])
        self.assertIsNone(report["rooms"][0]["volume"]["observed_m3"])

    def test_floor_and_roof_missing_force_open_shell_review(self):
        model = box_model()
        for kind in ("Floor", "RoofCeiling"):
            with self.subTest(kind=kind):
                observed = observations(model)
                observed[0]["surfaces"] = [surface for surface in observed[0]["surfaces"] if surface["kind"] != kind]
                report = compare_model_geometry(model, observed)
                self.assertReview(report)
                room = report["rooms"][0]
                self.assertEqual(room["surface_kinds"][kind]["missing_ratio"], 1)
                self.assertFalse(room["observed_shell"]["closed"])
                self.assertEqual(room["observed_shell"]["boundary_loop_count"], 1)
                self.assertIn({"code": "critical_surface_kind_missing", "kind": kind}, room["observed_issues"])

    def test_flattened_sloped_roof_with_same_xy_projection_requires_review(self):
        model = sloped_model()
        observed = observations(model)
        roof = next(surface for surface in observed[0]["surfaces"] if surface["kind"] == "RoofCeiling")
        for point in roof["boundary_mm"]:
            point[2] = 3000
        report = compare_model_geometry(model, observed)
        self.assertReview(report)
        room = report["rooms"][0]
        self.assertEqual(room["surface_kinds"]["RoofCeiling"]["coverage"], 0)
        self.assertFalse(room["skylights"][0]["host_plane_match"])

    def test_equal_areas_after_translation_still_measure_missing_and_extra(self):
        model = box_model()
        observed = observations(model)
        for surface in observed[0]["surfaces"]:
            for point in surface["boundary_mm"]:
                point[0] += 2000
        report = compare_model_geometry(model, observed)
        self.assertReview(report)
        floor = report["rooms"][0]["surface_kinds"]["Floor"]
        self.assertAlmostEqual(floor["source_area_m2"], floor["observed_area_m2"])
        self.assertAlmostEqual(floor["missing_ratio"], 0.2)
        self.assertAlmostEqual(floor["extra_ratio"], 0.2)
        self.assertAlmostEqual(floor["coverage"], 0.8)

    def test_closed_shell_volume_difference_is_reported_separately(self):
        model = box_model()
        observed = observations(model)
        for surface in observed[0]["surfaces"]:
            for point in surface["boundary_mm"]:
                point[2] *= 1.06
        report = compare_model_geometry(model, observed)
        self.assertReview(report)
        room = report["rooms"][0]
        self.assertTrue(room["observed_shell"]["closed"])
        self.assertAlmostEqual(room["volume"]["relative_difference"], 0.06)
        self.assertFalse(room["volume"]["passed"])

    def test_missing_small_skylight_receives_individual_failure(self):
        model = sloped_model()
        wall = next(face for face in model.rooms[0].faces if str(face.type) == "Wall")
        wall.apertures_by_ratio(0.5)
        observed = observations(model)
        observed[0]["openings"] = [opening for opening in observed[0]["openings"] if opening["identifier"] != "Skylight"]
        report = compare_model_geometry(model, observed)
        self.assertReview(report)
        skylight = report["rooms"][0]["skylights"][0]
        self.assertFalse(skylight["passed"])
        self.assertEqual(skylight["shape_comparison"]["missing_ratio"], 1)
        self.assertAlmostEqual(skylight["source_area_3d_m2"], (1 + 0.25 ** 2) ** 0.5)

    def test_exact_five_percent_opening_boundary_passes_and_excess_reviews(self):
        model = sloped_model()
        for factor, passed in ((0.95, True), (0.9499, False)):
            with self.subTest(factor=factor):
                observed = observations(model)
                for point in observed[0]["openings"][0]["boundary_mm"]:
                    point[0] = 2000 + (point[0] - 2000) * factor
                    point[2] = 3000 + point[0] * 0.25
                report = compare_model_geometry(model, observed)
                self.assertEqual(report["passed"], passed)
                skylight = report["rooms"][0]["skylights"][0]
                self.assertAlmostEqual(skylight["shape_comparison"]["missing_ratio"], 1 - factor)

    def test_coplanar_splits_match_without_source_identifiers(self):
        model = box_model()
        observed = observations(model)
        original = next(surface for surface in observed[0]["surfaces"] if surface["kind"] == "RoofCeiling")
        observed[0]["surfaces"].remove(original)
        for start, end in ((0, 4000), (4000, 7000), (7000, 10000)):
            observed[0]["surfaces"].append({
                "identifier": "piece_{}".format(start),
                "kind": "RoofCeiling",
                "boundary_mm": [[start, 0, 3000], [end, 0, 3000], [end, 8000, 3000], [start, 8000, 3000]],
            })
        report = compare_model_geometry(model, observed)
        self.assertPassed(report)
        self.assertEqual(len(report["rooms"][0]["surface_kinds"]["RoofCeiling"]["observed_surfaces"]), 3)

    def test_source_t_junction_normalizes_collinear_edges(self):
        model = box_model()
        source = model.rooms[0]
        floor = next(face for face in source.faces if str(face.type) == "Floor")
        faces = [face.duplicate() for face in source.faces if face is not floor]
        for index, (start, end) in enumerate(((0, 4000), (4000, 10000))):
            points = [Point3D(start, 0, 0), Point3D(end, 0, 0), Point3D(end, 8000, 0), Point3D(start, 8000, 0)]
            faces.append(Face("Floor_{}".format(index), Face3D(points), floor.type))
        model = Model("Tjunction", [Room("Box", faces)], units="Millimeters")
        report = compare_model_geometry(model, observations(model))
        self.assertPassed(report)
        shell = report["rooms"][0]["source_shell"]
        self.assertGreater(shell["raw_edge_pairing"]["naked_edge_count"], 0)
        self.assertTrue(shell["edge_normalization_applied"])
        self.assertEqual(shell["edge_pairing"]["naked_edge_count"], 0)

    def test_retraced_exterior_slit_requires_every_segment_within_one_mm(self):
        model = box_model()
        for width, closed in ((0.05, True), (4.0, False)):
            with self.subTest(width=width):
                observed = observations(model)
                roof = next(s for s in observed[0]["surfaces"] if s["kind"] == "RoofCeiling")
                roof["boundary_mm"] = [[x, y, 3000] for x, y in (
                    (0, 0), (4000, 0), (4000, 3000), (4000 + width, 3000),
                    (4000 + width, 0), (10000, 0), (10000, 8000), (0, 8000))]
                shell = compare_model_geometry(model, observed)["rooms"][0]["observed_shell"]
                self.assertEqual(shell["closed"], closed)
                if closed:
                    self.assertGreater(shell["cancelled_same_ring_retraces"], 0)
                    self.assertAlmostEqual(shell["maximum_retrace_distance_mm"], width)
                else:
                    self.assertGreater(shell["edge_pairing"]["naked_edge_count"], 0)
                    self.assertIsNone(shell["volume_m3"])

    def test_sydney_backstage_floor_chord_preserves_the_open_boundary(self):
        chain = [(-34.735234, 64303.793853), (1849.372595, 64299.062476), (3632.685246, 64297.641000),
                 (5304.877449, 64299.537658), (6856.267313, 64304.741467)]
        outline = [*chain, (6856.267313, 66000), (-34.735234, 66000)]
        base = Face3D(tuple(Point3D(x, y, 13157.2) for x, y in outline))
        prism = Polyface3D.from_offset_face(base, 3000)
        model = Model("CapturedSeam", [Room.from_polyface3d("Backstage", prism)], units="Millimeters")
        observed = observations(model)
        self.assertPassed(compare_model_geometry(model, observed))
        floor = next(s for s in observed[0]["surfaces"] if s["kind"] == "Floor")
        floor["boundary_mm"] = [[x, y, 13157.2] for x, y in (outline[0], *outline[4:])]
        room = compare_model_geometry(model, observed)["rooms"][0]
        self.assertTrue(room["surface_kinds"]["Floor"]["passed"])
        self.assertFalse(room["observed_shell"]["closed"])
        self.assertGreater(room["observed_shell"]["edge_pairing"]["naked_edge_count"], 2)
        self.assertIsNone(room["volume"]["observed_m3"])

    def test_sydney_plinth_clipped_corner_is_open_within_floor_area_tolerance(self):
        a = (28033.890, 93328.116)
        b = (28541.910, 95936.337)
        c = (28761.230, 95883.745)
        clipped = (28539.996, 95926.785)
        outline = [a, b, c, (29200, 93200)]
        floor = Face3D(tuple(Point3D(x, y, 5486.4) for x, y in reversed(outline)))
        model = Model("PlinthCorner", [Room.from_polyface3d("Corner", Polyface3D.from_offset_face(floor, 3000))], units="Millimeters")
        observed = observations(model)
        base = next(s for s in observed[0]["surfaces"] if s["kind"] == "Floor")
        base["boundary_mm"] = [[x, y, 5486.4] for x, y in (a, clipped, c, outline[-1])]
        room = compare_model_geometry(model, observed)["rooms"][0]
        self.assertTrue(room["surface_kinds"]["Floor"]["passed"])
        self.assertEqual(room["observed_shell"]["edge_pairing"]["naked_edge_count"], 3)
        self.assertFalse(room["observed_shell"]["closed"])

    def test_nearby_split_planes_cannot_accumulate_two_welding_tolerances(self):
        model = box_model()
        for high, closed in ((1.0, True), (1.6, False)):
            with self.subTest(high=high):
                observed = observations(model)
                surfaces = observed[0]["surfaces"]
                surfaces.remove(next(s for s in surfaces if s["kind"] == "RoofCeiling"))
                for start, end, elevation in ((0, 4000, 3000.8), (4000, 10000, 3000 + high)):
                    surfaces.append({"kind": "RoofCeiling", "boundary_mm": [
                        [x, y, elevation] for x, y in ((start, 0), (end, 0), (end, 8000), (start, 8000))]})
                shell = compare_model_geometry(model, observed)["rooms"][0]["observed_shell"]
                self.assertEqual(shell["closed"], closed)
                self.assertLessEqual(shell["maximum_paired_endpoint_distance_mm"], 1.0)

    def test_duplicate_floor_observations_cannot_supply_a_missing_roof(self):
        model = box_model()
        observed = observations(model)
        surfaces = observed[0]["surfaces"]
        surfaces.remove(next(s for s in surfaces if s["kind"] == "RoofCeiling"))
        duplicate = copy.deepcopy(next(s for s in surfaces if s["kind"] == "Floor"))
        duplicate["identifier"] = "DuplicateFloor"
        surfaces.append(duplicate)
        room = compare_model_geometry(model, observed)["rooms"][0]
        self.assertFalse(room["observed_shell"]["closed"])
        self.assertIsNone(room["volume"]["observed_m3"])
        self.assertIn("overlapping_surfaces", [e["code"] for e in room["observed_issues"]])

    def test_small_unsupported_interior_hole_remains_unpaired(self):
        model = box_model()
        for length in (0.25, 3000):
            with self.subTest(length=length):
                observed = observations(model)
                roof = next(s for s in observed[0]["surfaces"] if s["kind"] == "RoofCeiling")
                roof["holes_mm"] = [[[2000 + x, 2000 + y, 3000] for x, y in
                                     ((0, 0), (length, 0), (length, 0.25), (0, 0.25))]]
                room = compare_model_geometry(model, observed)["rooms"][0]
                self.assertFalse(room["observed_shell"]["closed"])
                self.assertIsNone(room["volume"]["observed_m3"])

    def test_partitioned_roof_distinguishes_overlay_seam_from_aperture(self):
        model = box_model()
        for width, closed in ((3e-7, True), (0.25, False)):
            with self.subTest(width_mm=width):
                observed = observations(model)
                surfaces = observed[0]["surfaces"]
                surfaces.remove(next(s for s in surfaces if s["kind"] == "RoofCeiling"))
                for x0, y0, x1, y1 in ((0, 0, 10000, 2500), (0, 5500, 10000, 8000),
                                        (0, 2500, 4000, 5500), (4000 + width, 2500, 10000, 5500)):
                    surfaces.append({"kind": "RoofCeiling", "boundary_mm": [
                        [x, y, 3000] for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]})
                shell = compare_model_geometry(model, observed)["rooms"][0]["observed_shell"]
                self.assertEqual(shell["closed"], closed)
                if closed:
                    self.assertAlmostEqual(shell["volume_m3"], 240)

    def test_courtyard_inner_rings_preserve_closed_shell_and_volume(self):
        boundary = tuple(Point3D(x, y, 0) for x, y in ((0, 0), (10000, 0), (10000, 10000), (0, 10000)))
        hole = tuple(Point3D(x, y, 0) for x, y in ((3000, 3000), (7000, 3000), (7000, 7000), (3000, 7000)))
        polyface = Polyface3D.from_offset_face(Face3D(boundary, holes=(hole,)), 3000)
        model = Model("Courtyard", [Room.from_polyface3d("Court", polyface)], units="Millimeters")
        observed = observations(model)
        report = compare_model_geometry(model, observed)
        self.assertPassed(report)
        self.assertAlmostEqual(report["rooms"][0]["volume"]["source_m3"], 252)
        roof = next(surface for surface in observed[0]["surfaces"] if surface["kind"] == "RoofCeiling")
        roof["holes_mm"] = []
        changed = compare_model_geometry(model, observed)
        self.assertReview(changed)
        self.assertGreater(changed["rooms"][0]["surface_kinds"]["RoofCeiling"]["extra_ratio"], 0.05)

    def test_concave_polygons_and_triangular_observations_match_in_3d(self):
        boundary = tuple(Point3D(x, y, 0) for x, y in ((0, 0), (10000, 0), (10000, 4000), (4000, 4000), (4000, 8000), (0, 8000)))
        polyface = Polyface3D.from_offset_face(Face3D(boundary), 3000)
        model = Model("Concave", [Room.from_polyface3d("LShape", polyface)], units="Millimeters")
        observed = observations(model)
        roof = next(face for face in model.rooms[0].faces if str(face.type) == "RoofCeiling")
        observed[0]["surfaces"] = [surface for surface in observed[0]["surfaces"] if surface["kind"] != "RoofCeiling"]
        for index, triangle in enumerate(roof.geometry.triangulated_mesh3d.face_vertices):
            observed[0]["surfaces"].append({
                "identifier": "Triangle_{}".format(index),
                "kind": "RoofCeiling",
                "boundary_mm": [list(point) for point in triangle],
            })
        report = compare_model_geometry(model, observed)
        self.assertPassed(report)
        self.assertAlmostEqual(report["rooms"][0]["volume"]["observed_m3"], 168)

    def test_actual_opening_panels_close_observed_cutouts(self):
        model = sloped_model()
        observed = observations(model)
        opening = observed[0]["openings"][0]
        roof = next(surface for surface in observed[0]["surfaces"] if surface["kind"] == "RoofCeiling")
        roof["holes_mm"] = [copy.deepcopy(opening["boundary_mm"])]
        report = compare_model_geometry(model, observed)
        self.assertPassed(report)
        self.assertEqual(len(report["rooms"][0]["observed_opening_fills"]), 1)
        observed[0]["openings"] = []
        changed = compare_model_geometry(model, observed)
        self.assertReview(changed)
        self.assertFalse(changed["rooms"][0]["observed_shell"]["closed"])

    def test_flattened_skylight_on_correct_roof_requires_shape_plane_review(self):
        model = sloped_model()
        observed = observations(model)
        for point in observed[0]["openings"][0]["boundary_mm"]:
            point[2] = 3500
        report = compare_model_geometry(model, observed)
        self.assertReview(report)
        skylight = report["rooms"][0]["skylights"][0]
        self.assertEqual(skylight["shape_comparison"]["missing_ratio"], 1)
        self.assertFalse(skylight["plane_matches"][0]["coplanar_with_source_host"])

    def test_identical_open_source_shell_is_explicit_source_anomaly(self):
        source = box_model().rooms[0]
        model = Model("OpenSource", [Room("Box", [face.duplicate() for face in source.faces if str(face.type) != "Floor"])], units="Millimeters")
        report = compare_model_geometry(model, observations(model))
        self.assertReview(report)
        room = report["rooms"][0]
        self.assertFalse(room["source_shell"]["closed"])
        self.assertIn("shell_open", [issue["code"] for issue in room["source_issues"]])
        self.assertEqual(report["summary"]["source_anomaly_room_count"], 1)

    def test_nonplanar_source_and_overlapping_source_openings_are_anomalies(self):
        source = box_model().rooms[0]
        roof = next(face for face in source.faces if str(face.type) == "RoofCeiling")
        points = list(roof.geometry.boundary)
        points[0] = Point3D(points[0].x, points[0].y, points[0].z + 30)
        faces = [face.duplicate() for face in source.faces if face is not roof]
        faces.append(Face(roof.identifier, Face3D(points), roof.type))
        model = Model("NonplanarSource", [Room("Box", faces)], units="Millimeters")
        report = compare_model_geometry(model, observations(model))
        self.assertReview(report)
        self.assertIn("nonplanar_surface", [issue["code"] for issue in report["rooms"][0]["source_issues"]])
        model = sloped_model()
        roof = next(face for face in model.rooms[0].faces if str(face.type) == "RoofCeiling")
        roof.add_aperture(Aperture("DuplicateSkylight", roof.apertures[0].geometry))
        report = compare_model_geometry(model, observations(model))
        self.assertReview(report)
        self.assertIn("overlapping_openings", [issue["code"] for issue in report["rooms"][0]["source_issues"]])

    def test_lost_small_source_roof_face_receives_individual_review(self):
        source = box_model().rooms[0]
        roof = next(face for face in source.faces if str(face.type) == "RoofCeiling")
        faces = [face.duplicate() for face in source.faces if face is not roof]
        for index, (start, end) in enumerate(((0, 100), (100, 10000))):
            points = [Point3D(start, 0, 3000), Point3D(end, 0, 3000), Point3D(end, 8000, 3000), Point3D(start, 8000, 3000)]
            faces.append(Face("Roof_{}".format(index), Face3D(points), roof.type))
        model = Model("SmallRoof", [Room("Box", faces)], units="Millimeters")
        observed = observations(model)
        observed[0]["surfaces"] = [surface for surface in observed[0]["surfaces"] if surface["identifier"] != "Roof_0"]
        report = compare_model_geometry(model, observed)
        self.assertReview(report)
        roof_report = report["rooms"][0]["surface_kinds"]["RoofCeiling"]
        self.assertAlmostEqual(roof_report["missing_ratio"], 0.01)
        self.assertEqual(roof_report["source_surfaces"][0]["missing_ratio"], 1)

    def test_per_room_failure_survives_large_matching_room(self):
        model = box_model()
        model.add_room(Room.from_box("Small", 100, 100, 100, origin=Point3D(20000, 0, 0)))
        observed = observations(model)
        observed[1]["surfaces"] = [surface for surface in observed[1]["surfaces"] if surface["kind"] != "RoofCeiling"]
        report = compare_model_geometry(model, observed)
        self.assertReview(report)
        self.assertTrue(report["rooms"][0]["passed"])
        self.assertFalse(report["rooms"][1]["passed"])
        self.assertLess(report["rooms"][1]["surface_area"]["missing_area_m2"], 0.05 * report["summary"]["source_area_m2"])

    def test_extra_small_displaced_component_receives_individual_review(self):
        model = box_model()
        observed = observations(model)
        observed[0]["surfaces"].append({
            "kind": "RoofCeiling",
            "identifier": "StrayRoof",
            "boundary_mm": [[20000, 0, 3000], [20100, 0, 3000], [20100, 100, 3000], [20000, 100, 3000]],
        })
        report = compare_model_geometry(model, observed)
        self.assertReview(report)
        roof = report["rooms"][0]["surface_kinds"]["RoofCeiling"]
        self.assertLess(roof["extra_ratio"], 0.05)
        self.assertEqual(roof["observed_surfaces"][-1]["extra_ratio"], 1)

    def test_absolute_plane_tolerance_is_independent_of_relative_tolerance(self):
        model = box_model(width=100000, depth=80000)
        observed = observations(model)
        for point in next(surface for surface in observed[0]["surfaces"] if surface["kind"] == "RoofCeiling")["boundary_mm"]:
            point[2] += 10
        report = compare_model_geometry(model, observed)
        self.assertReview(report)
        self.assertEqual(report["rooms"][0]["surface_kinds"]["RoofCeiling"]["coverage"], 0)

    def test_invalid_and_nonplanar_observations_are_reported(self):
        model = box_model()
        for coordinate in (float("nan"), 30.0):
            with self.subTest(coordinate=coordinate):
                observed = observations(model)
                roof = next(surface for surface in observed[0]["surfaces"] if surface["kind"] == "RoofCeiling")
                roof["boundary_mm"][0][2] += coordinate
                report = compare_model_geometry(model, observed)
                self.assertReview(report)
                self.assertTrue(report["rooms"][0]["observed_issues"])

    def test_duplicate_and_unexpected_rooms_require_review(self):
        model = box_model()
        observed = observations(model)
        duplicate = compare_model_geometry(model, observed + copy.deepcopy(observed))
        self.assertReview(duplicate)
        self.assertIn("duplicate_room_identifier", [issue["code"] for issue in duplicate["rooms"][0]["observed_issues"]])
        other = copy.deepcopy(observed[0])
        other["identifier"] = "Unexpected"
        report = compare_model_geometry(model, observed + [other])
        self.assertReview(report)
        self.assertEqual(report["unexpected_rooms"][0]["identifier"], "Unexpected")

    def test_model_units_scale_into_observation_millimetres(self):
        model = Model("Metres", [Room.from_box("Room", 10, 8, 3)], units="Meters")
        report = compare_model_geometry(model, observations(model))
        self.assertPassed(report)
        self.assertAlmostEqual(report["rooms"][0]["volume"]["source_m3"], 240)

    def test_fully_occluded_opening_has_complete_zero_area_metrics(self):
        geometry = Face3D(tuple(Point3D(x, y, 0) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))))
        report = coverage(geometry, [], 1000, expected_shape=Polygon())
        self.assertTrue(report["passed"])
        for key in ("source_area_m2", "native_area_m2", "missing_area_m2", "extra_area_m2"):
            self.assertEqual(report[key], 0)

    def test_parameter_validation(self):
        model = box_model()
        for tolerance in (-0.01, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                compare_model_geometry(model, [], tolerance=tolerance)
        for absolute in (0, -1, float("inf")):
            with self.assertRaises(ValueError):
                compare_model_geometry(model, [], absolute_tolerance_mm=absolute)


class RevitHouseSourceGeometryTest(unittest.TestCase):
    def test_rebuilt_source_shells_and_skylight_planes(self):
        model = Model.from_file(str(ROOT / "samples/HoneybeeModel_RevitHouse/HoneybeeModel_RevitHouse.hbjson"))
        report = compare_model_geometry(model, observations(model))
        self.assertTrue(report["passed"], json.dumps({room["identifier"]: room["issues"] for room in report["rooms"] if room["issues"]}))
        self.assertEqual(report["summary"]["source_room_count"], 15)
        self.assertEqual(report["summary"]["source_closed_shell_count"], 15)
        self.assertEqual(report["summary"]["source_anomaly_room_count"], 0)
        self.assertEqual(report["summary"]["source_skylight_count"], 2)
        self.assertEqual(report["summary"]["passed_skylight_count"], 2)
        self.assertTrue(all(not room["source_issues"] for room in report["rooms"]))
        for room in report["rooms"]:
            for skylight in room["skylights"]:
                self.assertTrue(skylight["host_plane_match"])
                self.assertTrue(skylight["passed"])
                self.assertLess(abs(skylight["source_host_plane"]["normal"][2]), 1)
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
