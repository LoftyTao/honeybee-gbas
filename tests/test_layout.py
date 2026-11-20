from __future__ import annotations

import unittest
from pathlib import Path

from honeybee.model import Model
from honeybee.room import Room
from honeybee.shade import Shade
from ladybug_geometry.geometry3d import Point3D

from honeybee_gbas.layout import plan_levels, plan_native_levels

ROOT = Path(__file__).resolve().parents[1]


def building() -> Model:
    lower = Room.from_box("lower", 5, 4, 3, origin=Point3D(10, 20, -3))
    lower.story = "Basement"
    upper = Room.from_box("upper", 4, 3, 3, origin=Point3D(11, 21, 0))
    upper.story = "Upper"
    model = Model("building", [upper, lower])
    model.add_shade(Shade.from_vertices("shade", [(10, 20, 7), (11, 20, 7), (11, 21, 7), (10, 21, 7)]))
    return model


class LevelLayoutTest(unittest.TestCase):
    def test_layout_keeps_shared_world_anchor_and_uses_story_spacing(self) -> None:
        floors = plan_levels(building())
        self.assertEqual([floor.story for floor in floors], ["Basement", "Upper"])
        self.assertEqual([floor.floor_number for floor in floors], [1, 2])
        self.assertEqual([floor.source_elevation_mm for floor in floors], [-3000, 0])
        self.assertEqual([floor.height_mm for floor in floors], [3000, 3000])
        self.assertGreater(floors[1].frame_bounds_mm[0] - floors[0].frame_bounds_mm[2], 0)
        self.assertAlmostEqual(floors[1].alignment_mm[0] - floors[0].alignment_mm[0], floors[1].offset_mm[0])

    def test_far_shade_expands_frame_and_retains_alignment(self) -> None:
        model = building()
        model.add_shade(Shade.from_vertices("far", [(50, 60, 2), (52, 60, 2), (52, 61, 2)]))
        floors = plan_levels(model)
        self.assertGreater(floors[0].frame_bounds_mm[2], 52000)
        self.assertGreater(floors[0].frame_bounds_mm[3], 61000)
        self.assertGreater(floors[1].frame_bounds_mm[0], floors[0].frame_bounds_mm[2])
        self.assertEqual(floors[1].alignment_mm[0] - floors[0].alignment_mm[0], floors[1].offset_mm[0])

    def test_staggered_rooms_share_floor_number_and_gain_series(self) -> None:
        upper = Room.from_box("upper", 5, 4, 3, origin=Point3D(10, 20, 0))
        lower = Room.from_box("lower", 5, 4, 3, origin=Point3D(10, 20, -0.55))
        upper.story = "Level 1"
        lower.story = "Level 1"
        model = Model("staggered", [upper, lower])
        levels = plan_levels(model)
        self.assertEqual(len(levels), 2)
        self.assertEqual({level.floor_number for level in levels}, {1})
        self.assertEqual({level.story for level in levels}, {"Level 1"})
        self.assertEqual({level.series for level in levels}, {"", "A"})
        self.assertEqual([round(level.source_elevation_mm, 1) for level in levels], [-550.0, 0.0])
        self.assertEqual([round(level.height_mm, 1) for level in levels], [3000.0, 3550.0])
        self.assertEqual({level.series: round(level.reference_elevation_mm, 1) for level in levels}, {"": -550.0, "A": -550.0})

    def test_native_primary_levels_preserve_source_story_and_absolute_elevation(self) -> None:
        model = Model("StaggeredNative", [
            Room.from_box("Sunken", 5, 4, 3, origin=Point3D(0, 0, -0.55)),
            Room.from_box("Regular", 5, 4, 3, origin=Point3D(10, 0, 0)),
            Room.from_box("Upper", 5, 4, 3, origin=Point3D(10, 0, 3)),
        ])
        for room in model.rooms:
            room.story = "Second" if room.identifier == "Upper" else "First"
        levels = plan_native_levels(model)
        self.assertEqual([row.floor_number for row in levels], [1, 2, 3])
        self.assertEqual([row.reference_elevation_mm for row in levels], [-550, 0, 3000])
        self.assertTrue(all(row.series == "" for row in levels))
        self.assertEqual([row.story for row in levels], ["First", "First", "Second"])

    def test_tall_lower_room_preserves_next_storey_base(self) -> None:
        tall = Room.from_box("Atrium", 5, 4, 8.4)
        lower = Room.from_box("Lower", 5, 4, 4.2, origin=Point3D(10, 0, 0))
        upper = Room.from_box("Upper", 5, 4, 4.2, origin=Point3D(10, 0, 4.2))
        tall.story = lower.story = "First"
        upper.story = "Second"
        levels = plan_levels(Model("OverlappingLevels", [tall, lower, upper]))
        self.assertEqual([row.height_mm for row in levels], [4200, 4200])
        self.assertEqual([row.reference_elevation_mm for row in levels], [0, 4200])

    def test_invalid_layout_parameters_require_review(self) -> None:
        model = Model("model", [Room.from_box("room")])
        for gap in (0, -1, float("nan"), float("inf")):
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                plan_levels(model, gap_mm=gap)
        for tolerance in (0, -1, float("nan"), float("inf")):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                plan_levels(model, tolerance_mm=tolerance)


if __name__ == "__main__":
    unittest.main()
