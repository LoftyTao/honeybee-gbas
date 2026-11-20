from __future__ import annotations

import copy
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from honeybee.model import Model
from honeybee.room import Room

from honeybee_gbas import model as hb
from honeybee_gbas.dwg import build
from honeybee_gbas.dwg.binary import BitReader, BitWriter
from honeybee_gbas.dwg.records import Filer, space
from honeybee_gbas.dwg.validation import decode_native, inspect_native


ROOT = Path(__file__).resolve().parents[1]


def named_model(identifier, names):
    rooms = []
    for identity, name in names:
        room = Room.from_box(identity)
        room.display_name = name
        rooms.append(room)
    return Model(identifier, rooms)


class RoomNameTest(unittest.TestCase):
    def test_display_names_preserve_unicode_and_business_numbers(self):
        model = named_model("LBT-Die-Model", [
            ("LBT-Die-F1_East_Office", "1层 · 东翼开放办公区"),
            ("LBT-Die-Room_101", "101-Kitchen & Dining"),
            ("LBT-Die-Room_102", "LBT-Die-Conference-102"),
        ])
        self.assertEqual(hb.room_names(model), {
            room.identifier: room.display_name for room in model.rooms
        })

    def test_sdk_identifier_fallback_uses_shared_model_namespace(self):
        model = named_model("LBT-Die-Model", [
            ("LBT-Die-101-Office", None),
            ("LBT-Die-F2-Meeting-A-02", None),
        ])
        self.assertEqual(hb.room_names(model), {
            "LBT-Die-101-Office": "101-Office",
            "LBT-Die-F2-Meeting-A-02": "F2-Meeting-A-02",
        })

    def test_sdk_roundtrip_keeps_equivalent_identifier_fallback(self):
        model = named_model("LBT-Die-Model", [("LBT-Die-101-Office", None)])
        restored = Model.from_dict(model.to_dict())
        self.assertEqual(hb.room_names(restored), hb.room_names(model))

    def test_complete_model_identity_is_a_namespace(self):
        for separator in ("-", "_", "."):
            with self.subTest(separator=separator):
                identity = "Project2026" + separator + "Office-A-101"
                model = named_model("Project2026", [(identity, None)])
                self.assertEqual(hb.room_names(model), {identity: "Office-A-101"})

    def test_room_only_prefix_preserves_business_identity(self):
        cases = (
            ("Building", [("Office-101", None), ("Office-102", None)]),
            ("101", [("101-Office", None), ("101-Storage", None)]),
            ("Level-1-Model", [("Level-1-Office", None), ("Level-1-Storage", None)]),
            ("LBT-Dieting-Model", [("LBT-Die-Office", None), ("LBT-Die-Storage", None)]),
            ("LBT-Die-Model", [("LBT-Die-Office", None), ("External-Storage", None)]),
        )
        for identifier, names in cases:
            with self.subTest(identifier=identifier, names=names):
                model = named_model(identifier, names)
                self.assertEqual(hb.room_names(model), {room.identifier: room.identifier for room in model.rooms})

    def test_shared_namespace_preserves_following_business_number(self):
        model = named_model("LBT-Die-101-Model", [
            ("LBT-Die-101-Office-A", None),
            ("LBT-Die-101-Storage-B", None),
        ])
        self.assertEqual(hb.room_names(model), {
            "LBT-Die-101-Office-A": "101-Office-A",
            "LBT-Die-101-Storage-B": "101-Storage-B",
        })

    def test_blank_display_names_use_source_identity(self):
        model = named_model("Project", [("Project_101-Office", ""), ("Project_102-Office", "  ")])
        self.assertEqual(hb.room_names(model), {
            "Project_101-Office": "101-Office", "Project_102-Office": "102-Office"
        })

    def test_duplicate_names_use_source_identity_and_stable_order(self):
        names = [("LBT-Die-East", "Office"), ("LBT-Die-West", "Office")]
        first = hb.room_names(named_model("LBT-Die-Model", names))
        second = hb.room_names(named_model("LBT-Die-Model", list(reversed(names))))
        self.assertEqual(first, second)
        self.assertEqual(first, {"LBT-Die-East": "Office [1]", "LBT-Die-West": "Office [2]"})

    def test_generated_duplicate_names_preserve_existing_visible_names(self):
        model = named_model("Building", [
            ("East", "Office"), ("West", "Office"), ("Reserved", "Office [1]")
        ])
        names = hb.room_names(model)
        self.assertEqual(names["Reserved"], "Office [1]")
        self.assertEqual(names["East"], "Office [2]")
        self.assertEqual(len(set(names.values())), 3)

    def test_source_model_and_calculation_codes_keep_their_identity(self):
        model = named_model("LBT-Die-Model", [("LBT-Die-101-Office", None), ("LBT-Die-102-Office", None)])
        original = copy.deepcopy(model.to_dict())
        codes = hb.room_codes(room.identifier for room in model.rooms)
        names = hb.room_names(model)
        self.assertEqual(model.to_dict(), original)
        self.assertEqual(hb.room_codes(names), codes)
        self.assertEqual(set(names), set(codes))

    def test_duplicate_source_identities_require_review(self):
        with self.assertRaises(ValueError):
            hb.room_names(named_model("Building", [("Room", "Office"), ("Room", "Lobby")]))

    def test_empty_model_has_empty_names(self):
        self.assertEqual(hb.room_names(Model("Empty")), {})

    def test_native_fields_separate_source_identity_code_and_visible_name(self):
        model = named_model("LBT-Die-Model", [("LBT-Die-101-Office", "101-办公区")])
        room = model.rooms[0]
        row = {
            "identifier": room.identifier,
            "native_code": hb.room_code(room.identifier),
            "name": hb.room_names(model)[room.identifier],
            "position_mm": (2500, 2000, 0),
            "area_m2": 20.0,
            "perimeter_mm": 18000.0,
            "height_mm": 3000.0,
            "outline_mm": [(0, 0), (0, 4000), (5000, 4000), (5000, 0)],
            "category": 0,
            "walls": [],
        }
        writer = BitWriter()
        references = []
        space(Filer(writer, references), row, SimpleNamespace(styles={"_SPACE": 77}))
        data = writer.finish()
        reader = BitReader(data, limit=writer.position)
        self.assertEqual(reader.rc(), 251)
        self.assertEqual((reader.bd(), reader.bd(), reader.rc(), reader.text()), (100.0, 0.0, 0, ""))
        self.assertEqual(reader.text(), "<SRootEntity><HoneybeeSource>LBT-Die-101-Office</HoneybeeSource></SRootEntity>")
        decoded = decode_native("SWR_SPACE", BitReader(data, limit=writer.position), references)
        self.assertEqual(decoded["identifier"], hb.room_code(room.identifier))
        self.assertEqual(decoded["name"], "101-办公区")

    def test_native_room_usage_and_name_share_the_source_display_name(self):
        model = named_model("Readable", [("Room_101", "首层办公区")])
        original = model.to_dict()
        drawing, manifest = build(model)
        geometry = inspect_native(drawing.bytes(), manifest, include_geometry=True)["geometry"]
        row = manifest["rooms"][0]
        native = geometry["entities"][row["handle"]]
        dictionary = geometry["dictionaries"][native["dictionary"]]
        properties = geometry["properties"][dictionary["SRootEntity"]]["values"]
        self.assertEqual(native["name"], "首层办公区")
        self.assertEqual(properties["SpcUsage"], native["name"])
        self.assertEqual(row["room_type"], native["name"])
        self.assertEqual(native["identifier"], hb.room_code("Room_101"))
        self.assertEqual(model.to_dict(), original)


class SourceRoomNameTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {
            path.stem: Model.from_file(str(path))
            for path in sorted((ROOT / "samples").glob("*/*.hbjson"))
            if path.stem != "Sydney_Opera_House"
        }

    def test_dieting_uses_source_display_names(self):
        model = self.models["Dieting_Spatial_Building"]
        names = hb.room_names(model)
        self.assertEqual(len(names), 84)
        self.assertEqual(names["F1_Circulation"], "首层 · 连续庭院走道")
        self.assertEqual(names["F1_East_Office"], "1层 · 东翼开放办公区")

    def test_resource_display_names_and_identities_keep_their_source(self):
        self.assertEqual(len(self.models), 11)
        self.assertEqual(sum(len(model.rooms) for model in self.models.values()), 341)
        for source, model in self.models.items():
            with self.subTest(source=source):
                original = [(room.identifier, room.display_name) for room in model.rooms]
                names = hb.room_names(model)
                counts = Counter(name for _, name in original)
                self.assertEqual(set(names), {identifier for identifier, _ in original})
                self.assertEqual(len(set(names.values())), len(model.rooms))
                for identifier, name in original:
                    if counts[name] == 1:
                        self.assertEqual(names[identifier], name)
                    else:
                        self.assertTrue(names[identifier].startswith(name + " ["))
                        self.assertTrue(names[identifier][len(name) + 2:-1].isdigit())
                self.assertEqual([(room.identifier, room.display_name) for room in model.rooms], original)


if __name__ == "__main__":
    unittest.main()
