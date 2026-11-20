from __future__ import annotations

import contextlib
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import get_type_hints
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from click.testing import CliRunner
from honeybee.cli import main as honeybee_main
from honeybee.face import Face
from honeybee.model import Model
from honeybee.room import Room
from honeybee.writer import model as model_writer
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.load.setpoint import Setpoint
from honeybee_energy.material.opaque import EnergyMaterial
from honeybee_energy.schedule.ruleset import ScheduleRuleset
from ladybug_geometry.geometry3d import Face3D, Point3D

import honeybee_gbas
import honeybee_gbas.dwg as dwg_module
from honeybee_gbas import model as gbas_model
from honeybee_gbas import writer as writer_module
from honeybee_gbas.cli import gbas
from honeybee_gbas.cli.translate import model_to_dwg as cli_model_to_dwg
from honeybee_gbas.cli.translate import model_to_dwg_cli
from honeybee_gbas.dwg.validation import ExportValidationError
from honeybee_gbas.properties.model import ModelGbasProperties
from honeybee_gbas.writer import model_to_dwg

def temporary_directory():
    return tempfile.TemporaryDirectory()


def window_room(name="WindowRoom"):
    room = Room.from_box(name, 5, 4, 3)
    room.story = "Level 1"
    wall = next(face for face in room.faces if str(face.type) == "Wall")
    wall.aperture_by_width_height(1, 1, 1)
    return room


class GbasExportTest(unittest.TestCase):
    def assert_ac1015(self, target):
        self.assertTrue(Path(target).read_bytes().startswith(b"AC1015"))

    def test_box_window_export_is_native_dwg_with_gate_and_preserved_source(self):
        model = Model("BoxWindow", [window_room()])
        original = copy.deepcopy(model.to_dict())
        with temporary_directory() as folder:
            target = Path(folder) / "model.dwg"
            written, report = model_to_dwg(model, target)
            self.assertIsInstance(written, Path)
            self.assertEqual(written.resolve(), target.resolve())
            self.assertEqual(sorted(path.name for path in Path(folder).iterdir()), ["model.dwg"])
            self.assert_ac1015(target)
            self.assertIsInstance(report, dict)
            self.assertEqual(report["counts"]["rooms"], 1)
            self.assertEqual(report["counts"]["native_openings"], 1)
            self.assertTrue(report["self_check"]["passed"])
            self.assertEqual(report["source_geometry_validation"]["summary"]["passed_room_wall_unions"], 1)
            self.assertTrue(report["native_quality"]["passed"])
            self.assertTrue(report["native_quality"]["opening_bindings_passed"])
        self.assertEqual(model.to_dict(), original)

    def test_default_export_omits_auxiliary_layers_and_keeps_native_source_ids(self):
        model = Model("AuxiliaryOmission", [window_room("SourceRoom")])
        with temporary_directory() as folder:
            target = Path(folder) / "model.dwg"
            _, report = model_to_dwg(model, target)
            data = target.read_bytes()
            for name in ("HB-Room-Source", "HB-Opening-Reference", "HB-Source-Labels"):
                self.assertNotIn(name.encode(), data)
            self.assertIn(b"HB-Source-3D", data)
            readback = dwg_module.validation.inspect_native(data, report, include_geometry=True)
            entities = readback["geometry"]["entities"]
            room = report["rooms"][0]
            native = entities[room["handle"]]
            self.assertIn((1000, room["identifier"]), native["xdata"])
            self.assertEqual(native["identifier"], room["native_code"])
            self.assertEqual(report["display_policy"]["hidden_reference_layers"], ["HB-Source-3D"])

    def test_report_is_returned_and_written_only_when_requested(self):
        model = Model("ReportOnDemand", [Room.from_box("Room")])
        with temporary_directory() as folder:
            folder = Path(folder)
            target = folder / "model.dwg"
            real_dumps = json.dumps
            with mock.patch.object(writer_module, "json") as json_ref:
                json_ref.dumps.side_effect = AssertionError("default export serialized the report")
                _, report = model_to_dwg(model, target)
                json_ref.dumps.assert_not_called()
            self.assertEqual(sorted(path.name for path in folder.iterdir()), ["model.dwg"])
            self.assertEqual(report["counts"]["rooms"], 1)
            self.assertEqual(set(report["runtime"]), {"writer", "format", "intermediate_format"})
            explicit = folder / "report.json"
            target_with_report = folder / "with-report.dwg"
            with mock.patch.object(writer_module, "json") as json_ref:
                json_ref.dumps.side_effect = real_dumps
                _, written_report = model_to_dwg(model, target_with_report, report_path=explicit)
                json_ref.dumps.assert_called_once()
            self.assertTrue(explicit.is_file())
            payload = json.loads(explicit.read_text(encoding="utf-8"))
            self.assertEqual(payload["counts"], written_report["counts"])
            self.assertEqual(payload["self_check"], written_report["self_check"])
            self.assertEqual(payload["rooms"][0]["identifier"], written_report["rooms"][0]["identifier"])

    def test_repeated_exports_do_not_register_process_audit_hooks(self):
        model = Model("NoAuditHooks", [Room.from_box("Room")])
        with temporary_directory() as folder, mock.patch("sys.addaudithook") as add_hook:
            folder = Path(folder)
            model_to_dwg(model, folder / "first.dwg")
            _, report = model_to_dwg(model, folder / "second.dwg")
            add_hook.assert_not_called()
        self.assertEqual(report["runtime"], {"writer": "Python", "format": "AC1015", "intermediate_format": "direct_binary"})

    def test_model_type_annotations_resolve_against_sdk_types(self):
        values = (
            gbas_model.WallSegment,
            gbas_model.OpeningRect,
            gbas_model.wall_bottom_edge,
            gbas_model.face_thickness_m,
            gbas_model.room_plan_boundaries,
            gbas_model.wall_segments,
        )
        for value in values:
            with self.subTest(value=getattr(value, "__name__", value)):
                self.assertTrue(get_type_hints(value))

    def test_existing_outputs_require_explicit_overwrite(self):
        model = Model("Overwrite", [Room.from_box("Room")])
        with temporary_directory() as folder:
            folder = Path(folder)
            target = folder / "model.dwg"
            model_to_dwg(model, target)
            before = target.read_bytes()
            with self.assertRaises(FileExistsError):
                model_to_dwg(model, target)
            self.assertEqual(target.read_bytes(), before)
            _, overwritten = model_to_dwg(model, target, overwrite=True)
            self.assertEqual(overwritten["counts"]["rooms"], 1)
            self.assert_ac1015(target)
            existing_report = folder / "existing.json"
            existing_report.write_text("{}\n", encoding="utf-8")
            second = folder / "second.dwg"
            with self.assertRaises(FileExistsError):
                model_to_dwg(model, second, report_path=existing_report)
            self.assertFalse(second.exists())
            _, report = model_to_dwg(model, second, report_path=existing_report, overwrite=True)
            payload = json.loads(existing_report.read_text(encoding="utf-8"))
            self.assertEqual(payload["counts"], report["counts"])

    def test_sdk_extension_registers_model_writer_and_properties(self):
        model = Model("Extension", [Room.from_box("Room")])
        self.assertFalse(hasattr(honeybee_gbas, "export_dwg"))
        self.assertIs(model_writer.dwg, model_to_dwg)
        self.assertIs(model.to.dwg, model_to_dwg)
        self.assertIsInstance(model.properties.gbas, ModelGbasProperties)
        self.assertIs(model.properties.gbas.host, model)
        self.assertEqual(repr(model.properties.gbas), "Model GBAS Properties: [host: Extension]")
        self.assertEqual(model.properties.gbas.ToString(), "Model GBAS Properties: [host: Extension]")
        with temporary_directory() as folder:
            target = Path(folder) / "extension.dwg"
            written, report = model.to.dwg(model, target)
            self.assertEqual(Path(written).resolve(), target.resolve())
            self.assertTrue(target.is_file())
            self.assertEqual(report["counts"]["rooms"], 1)

    def test_model_properties_serialization_and_extension_validation(self):
        model = Model("PropertyCheck", [window_room()])
        model_dict = model.to_dict()
        self.assertEqual(model_dict["properties"]["gbas"], {"type": "ModelGbasProperties"})
        restored = Model.from_dict(model_dict)
        self.assertEqual(restored.to_dict(), model_dict)
        without_gbas = copy.deepcopy(model_dict)
        del without_gbas["properties"]["gbas"]
        self.assertEqual(Model.from_dict(without_gbas).to_dict(), model_dict)
        with self.assertRaises(AssertionError):
            model.properties.gbas.apply_properties_from_dict({"properties": {}})
        self.assertEqual(model.check_for_extension("gbas", raise_exception=False), "")
        self.assertEqual(model.check_for_extension("gbas", raise_exception=False, detailed=True), [])

        empty_model = Model("EmptyModel", [])
        with self.assertRaises(ValueError):
            empty_model.check_for_extension("gbas")
        message = empty_model.check_for_extension("gbas", raise_exception=False)
        self.assertIn("requires at least one room", message)
        details = empty_model.check_for_extension("gbas", raise_exception=False, detailed=True)
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["code"], "040101")
        self.assertEqual(details[0]["extension_type"], "GBAS")
        self.assertEqual(
            empty_model.properties._check_func_from_code("040101").__name__,
            "check_has_rooms",
        )

        wall = Face(
            "WallOnly",
            Face3D((Point3D(0, 0, 0), Point3D(4, 0, 0), Point3D(4, 0, 3), Point3D(0, 0, 3))),
        )
        no_floor_room = Room("NoFloorRoom", [wall])
        floorless = Model("FloorlessModel", [no_floor_room])
        with self.assertRaises(ValueError):
            floorless.properties.gbas.check_room_floor_polygons()
        floor_details = floorless.properties.gbas.check_room_floor_polygons(
            raise_exception=False, detailed=True
        )
        self.assertEqual(len(floor_details), 1)
        self.assertEqual(floor_details[0]["code"], "040102")
        self.assertEqual(floor_details[0]["extension_type"], "GBAS")
        self.assertEqual(
            floorless.properties._check_func_from_code("040102").__name__,
            "check_room_floor_polygons",
        )

    def test_only_dwg_output_and_distinct_report_paths_are_accepted(self):
        model = Model("Formats", [Room.from_box("Room")])
        with temporary_directory() as folder:
            folder = Path(folder)
            attempts = (
                (folder / "model.dxf", None),
                (folder / "model.json", None),
                (folder / "model.dwg", folder / "model.dwg"),
            )
            for target, report_path in attempts:
                with self.subTest(target=target.name, report=report_path is not None):
                    with self.assertRaises(ValueError):
                        model_to_dwg(model, target, report_path=report_path)
            self.assertEqual(list(folder.iterdir()), [])

    def test_units_keep_absolute_geometry_and_elevations(self):
        cases = (
            ("Meters", 5, 4, 3, 2.5, 20.0, 3000.0, 2500.0),
            ("Feet", 10, 10, 10, 10, 9.290304, 3048.0, 3048.0),
        )
        for units, width, depth, height, origin_z, area, height_mm, elevation in cases:
            with self.subTest(units=units), temporary_directory() as folder:
                room = Room.from_box(f"{units}Room", width, depth, height, origin=Point3D(0, 0, origin_z))
                room.story = "Level 1"
                model = Model(f"{units}Model", [room], units=units)
                original = copy.deepcopy(model.to_dict())
                _, report = model_to_dwg(model, Path(folder) / "model.dwg")
                self.assertAlmostEqual(report["rooms"][0]["area_m2"], area, places=6)
                self.assertAlmostEqual(report["rooms"][0]["height_mm"], height_mm, places=6)
                self.assertTrue(report["walls"])
                self.assertTrue(all(abs(row["height_mm"] - height_mm) < 1e-6 for row in report["walls"]))
                floor = report["floors"][0]
                self.assertAlmostEqual(floor["source_elevation_mm"], elevation, places=6)
                self.assertAlmostEqual(floor["reference_elevation_mm"], elevation, places=6)
                self.assertEqual(report["source_geometry_validation"]["summary"]["passed_room_wall_unions"], 1)
                self.assertTrue(report["self_check"]["passed"])
                self.assertEqual(model.to_dict(), original)

    def test_small_metric_and_feet_rooms_use_sdk_unit_conversion(self):
        cases = (
            ("Meters", 2.5, 2.0, 2.5, 0.0, 5.0, 2500.0, 0.0),
            ("Feet", 3.0, 2.0, 2.5, 1.5, 0.55741824, 762.0, 457.2),
        )
        for units, width, depth, height, origin_z, area, height_mm, elevation in cases:
            with self.subTest(units=units), temporary_directory() as folder:
                room = Room.from_box(f"{units}Small", width, depth, height, origin=Point3D(0, 0, origin_z))
                room.story = "Level 1"
                model = Model(f"{units}SmallModel", [room], units=units)
                _, report = model_to_dwg(model, Path(folder) / "model.dwg")
                check = next(row for row in report["unit_cross_check"]["rooms"] if row["identifier"] == room.identifier)
                self.assertTrue(report["unit_cross_check"]["passed"])
                self.assertTrue(report["self_check"]["passed"])
                self.assertEqual(set(check["checks"]), {"native_area", "source_area", "native_height", "native_elevation"})
                self.assertAlmostEqual(check["native_area_m2"], area, places=6)
                self.assertAlmostEqual(check["native_source_area_m2"], area, places=6)
                self.assertAlmostEqual(check["native_height_mm"], height_mm, places=6)
                self.assertAlmostEqual(check["native_elevation_mm"], elevation, places=6)

    def test_shared_wall_height_split_keeps_absolute_unit_check_valid(self):
        tall = Room.from_box("Tall", 5, 4, 6, origin=Point3D(5, 0, 0))
        lower = Room.from_box("Lower", 5, 4, 3)
        for room in (tall, lower):
            room.story = "Level 1"
        model = Model("SharedHeight", [tall, lower])
        with temporary_directory() as folder:
            _, report = model_to_dwg(model, Path(folder) / "model.dwg")
        split_rows = [row for row in report["walls"] if row["identifier"] == "Tall_Left"]
        self.assertEqual(len(split_rows), 2)
        self.assertAlmostEqual(sum(row["height_mm"] for row in split_rows), 6000.0)
        self.assertTrue(report["unit_cross_check"]["passed"])
        self.assertTrue(report["self_check"]["passed"])
        self.assertTrue(any(row["identifier"] == "Tall_Left" and row["reason"] == "non_horizontal_section_rows" for row in report["unit_cross_check"]["skipped"]))

    def test_unit_scale_fault_fails_before_dwg_publish(self):
        model = Model("ScaleFault", [Room.from_box("Room", 5, 4, 3)])
        with temporary_directory() as folder:
            folder = Path(folder)
            with mock.patch("honeybee_gbas.model.unit_scale", return_value=1.0):
                with self.assertRaises(ExportValidationError) as raised:
                    model_to_dwg(model, folder / "model.dwg")
            self.assertFalse(raised.exception.report["unit_cross_check"]["passed"])
            self.assertFalse(raised.exception.report["self_check"]["checks"]["native_absolute_units"])
            self.assertEqual(list(folder.iterdir()), [])

    def test_skylight_roof_model_exports_a_native_definition(self):
        room = Room.from_box("RoofRoom", 5, 4, 3)
        room.story = "Top"
        roof = next(face for face in room.faces if str(face.type) == "RoofCeiling")
        roof.aperture_by_width_height(2, 1, 1)
        model = Model("Skylight", [room])
        original = copy.deepcopy(model.to_dict())
        with temporary_directory() as folder:
            target = Path(folder) / "skylight.dwg"
            _, report = model_to_dwg(model, target)
            self.assert_ac1015(target)
            self.assertEqual(report["counts"]["source_skylights"], 1)
            self.assertEqual(report["counts"]["native_skylights"], 1)
            self.assertEqual(report["counts"]["native_roof_faces"], 0)
            self.assertEqual(report["validation"]["native_skylight_definitions"], 1)
            self.assertEqual(report["source_geometry_validation"]["summary"]["review_items"], 0)
            self.assertTrue(report["native_quality"]["passed"])
            self.assertTrue(report["self_check"]["passed"])
        self.assertEqual(model.to_dict(), original)

    def test_revit_house_regression_preserves_native_and_thermal_sources(self):
        model = Model.from_file(str(ROOT / "samples/HoneybeeModel_RevitHouse/HoneybeeModel_RevitHouse.hbjson"))
        original = copy.deepcopy(model.to_dict())
        self.assertEqual(len(model.rooms), 15)
        with temporary_directory() as folder:
            target = Path(folder) / "revit-house.dwg"
            _, report = model_to_dwg(model, target)
            self.assert_ac1015(target)
            counts = report["counts"]
            self.assertEqual(counts["rooms"], 15)
            self.assertEqual(counts["native_skylights"], 2)
            self.assertEqual(counts["native_roof_faces"], 21)
            self.assertEqual(counts["floor_frames"], 7)
            self.assertEqual(report["source_geometry_validation"]["summary"]["passed_room_wall_unions"], 15)
            self.assertTrue(report["native_quality"]["opening_bindings_passed"])
            self.assertTrue(report["self_check"]["passed"])
            energy = report["source_energy"]
            self.assertTrue(energy["constructions"])
            assignments = energy["room_assignments"]
            self.assertEqual(set(assignments), {room.identifier for room in model.rooms})
            self.assertTrue(set(assignments.values()).issubset({row["name"] for row in energy["room_types"]}))
        self.assertEqual(model.to_dict(), original)
        self.assertTrue(all(room.properties.energy.program_type is not None for room in model.rooms))
        self.assertTrue(all(room.properties.energy.setpoint is not None for room in model.rooms))

    def test_real_models_with_elevated_and_oblique_walls_export(self):
        for name in ("basic_sample_model", "sample_model_small"):
            with self.subTest(model=name), temporary_directory() as folder:
                model = Model.from_file(str(ROOT / "samples" / name / f"{name}.hbjson"))
                target = Path(folder) / f"{name}.dwg"
                _, report = model_to_dwg(model, target)
                self.assert_ac1015(target)
                self.assertEqual(report["counts"]["rooms"], len(model.rooms))
                self.assertTrue(report["unit_cross_check"]["passed"])
                self.assertTrue(report["self_check"]["passed"])

    def test_custom_construction_and_program_survive_in_source_energy(self):
        room = Room.from_box("AttributeRoom", 5, 4, 3)
        room.story = "Level 1"
        wall = next(face for face in room.faces if str(face.type) == "Wall")
        material = EnergyMaterial("SourceMaterial", 0.2, 0.5, 1200, 900, roughness="MediumRough")
        wall.properties.energy.construction = OpaqueConstruction("SourceWall", [material])
        room.properties.energy.setpoint = Setpoint(
            "RoomSetpoint",
            ScheduleRuleset.from_constant_value("Heating", 18),
            ScheduleRuleset.from_constant_value("Cooling", 27),
            ScheduleRuleset.from_constant_value("Humidity", 55),
            ScheduleRuleset.from_constant_value("Dehumidity", 55),
        )
        model = Model("Attributes", [room])
        original = copy.deepcopy(model.to_dict())
        with temporary_directory() as folder:
            _, report = model_to_dwg(model, Path(folder) / "attributes.dwg")
            energy = report["source_energy"]
            record = next(row for row in energy["constructions"] if row["identifier"] == "SourceWall")
            layer = record["layers"][0]
            self.assertAlmostEqual(record["thickness_mm"], 200, places=6)
            self.assertAlmostEqual(layer["thickness_mm"], 200, places=6)
            source_material = next(row for row in energy["materials"] if row["id"] == layer["material_id"])
            self.assertAlmostEqual(source_material["conductivity"], 0.5)
            self.assertAlmostEqual(source_material["density"], 1200)
            self.assertAlmostEqual(source_material["specific_heat"], 900)
            self.assertEqual(energy["assignments"][wall.identifier], record["id"])
            room_type_name = energy["room_assignments"][room.identifier]
            room_type = next(row for row in energy["room_types"] if row["name"] == room_type_name)
            self.assertIn("setpoint", room_type["room_load_overrides"])
            self.assertEqual(room_type["sections"]["能耗分析"]["冬季温度"], "18")
            self.assertEqual(room_type["sections"]["能耗分析"]["夏季温度"], "27")
        self.assertEqual(model.to_dict(), original)

    def test_cli_help_and_translation(self):
        runner = CliRunner()
        help_result = runner.invoke(gbas, ["--help"])
        self.assertEqual(help_result.exit_code, 0)
        self.assertIn("translate", help_result.output)

        version_result = runner.invoke(gbas, ["--version"])
        self.assertEqual(version_result.exit_code, 0)
        self.assertIn(honeybee_gbas.__version__, version_result.output)

        translate_help = runner.invoke(gbas, ["translate", "--help"])
        self.assertEqual(translate_help.exit_code, 0)
        self.assertIn("model-to-dwg", translate_help.output)

        self.assertIn("gbas", honeybee_main.commands)
        hb_help = runner.invoke(honeybee_main, ["gbas", "translate", "--help"])
        self.assertEqual(hb_help.exit_code, 0)
        self.assertIn("model-to-dwg", hb_help.output)

        model = Model("CliModel", [Room.from_box("Room")])
        with temporary_directory() as folder:
            folder = Path(folder)
            source = Path(model.to_hbjson("source", folder=str(folder)))
            target = folder / "cli.dwg"
            report_path = folder / "cli.json"
            result = runner.invoke(
                gbas,
                ["translate", "model-to-dwg", str(source), "-o", str(target), "--report", str(report_path)],
            )
            self.assertEqual(result.exit_code, 0)
            summary = json.loads(result.output)
            self.assertEqual(summary["status"], "translated")
            self.assert_ac1015(target)
            self.assertTrue(report_path.is_file())
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["counts"]["rooms"], 1)

            default_output = source.with_suffix(".dwg")
            default_run = runner.invoke(model_to_dwg_cli, [str(source)])
            self.assertEqual(default_run.exit_code, 0)
            self.assert_ac1015(default_output)

            blocked_run = runner.invoke(model_to_dwg_cli, [str(source)])
            self.assertEqual(blocked_run.exit_code, 1)

            overwrite_run = runner.invoke(model_to_dwg_cli, [str(source), "--overwrite"])
            self.assertEqual(overwrite_run.exit_code, 0)

            direct_target = folder / "direct.dwg"
            written, report = cli_model_to_dwg(str(source), str(direct_target), overwrite=True)
            self.assertEqual(Path(written).resolve(), direct_target.resolve())
            self.assertEqual(report["counts"]["rooms"], 1)


if __name__ == "__main__":
    unittest.main()
