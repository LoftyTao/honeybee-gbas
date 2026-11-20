from __future__ import annotations

import struct
import unittest

from honeybee_gbas.dwg.binary import (
    BitReader,
    BitWriter,
    Drawing,
    crc16,
    extended_data,
    inspect_container,
    modular_char,
    modular_short,
    object_map,
    read_modular_char,
)
from honeybee_gbas.dwg.records import Filer


class BinaryPrimitiveTest(unittest.TestCase):
    def test_primitive_roundtrip_across_unaligned_fields(self):
        writer = BitWriter()
        writer.bits(5, 3)
        for value in (0, 1, 255, 256, 257, 65535):
            writer.bs(value)
        for value in (0, 1, 255, 256, 0xFFFFFFFF):
            writer.bl(value)
        for value in (0.0, 1.0, -550.0, 304.8, 1e-12):
            writer.bd(value)
        writer.text("建筑轮廓 RevitHouse")
        writer.handle(5, 0x123456789ABC)
        reader = BitReader(writer.finish(), limit=writer.position)
        self.assertEqual(reader.bits(3), 5)
        self.assertEqual([reader.bs() for _ in range(6)], [0, 1, 255, 256, 257, 65535])
        self.assertEqual([reader.bl() for _ in range(5)], [0, 1, 255, 256, 0xFFFFFFFF])
        self.assertEqual([reader.bd() for _ in range(5)], [0.0, 1.0, -550.0, 304.8, 1e-12])
        self.assertEqual(reader.text(), "建筑轮廓 RevitHouse")
        self.assertEqual(reader.handle(), (5, 0x123456789ABC))
        self.assertEqual(reader.position, reader.limit)

    def test_crc_reference_vector(self):
        self.assertEqual(crc16(b"123456789", 0xFFFF), 0x4B37)

    def test_extended_string_matches_cad_written_legacy_record(self):
        expected = bytes.fromhex("00170027526f6f6d5f315f66356233336538362e2e4661636530000005002757616c6c00")
        self.assertEqual(extended_data([(1000, "Room_1_f5b33e86..Face0"), (1000, "Wall")]), expected)

    def test_native_point2d_uses_raw_doubles(self):
        writer = BitWriter()
        Filer(writer, []).poly([(2500.0, 6500.0), (-9362.7, -8362.7), (1000.0, -1000.0)])
        reader = BitReader(writer.finish(), limit=writer.position)
        self.assertEqual((reader.rc(), reader.bs()), (1, 3))
        for point in ((2500.0, 6500.0), (-9362.7, -8362.7), (1000.0, -1000.0)):
            self.assertEqual((reader.rd(), reader.rd()), point)
            self.assertEqual(reader.bd(), 0.0)
        self.assertEqual(reader.position, reader.limit)

    def test_xrecord_matches_cad_written_legacy_record(self):
        drawing = Drawing([])
        drawing.add_xrecord(0x8D5, 0x8D4, [(280, 1), (301, "0")])
        _, data = drawing.additional_objects[-1]
        reader = BitReader(data)
        self.assertEqual(reader.bs(), 0x4F)
        end = reader.rl()
        self.assertEqual(reader.handle(), (0, 0x8D5))
        self.assertEqual((reader.bs(), reader.bl()), (0, 0))
        self.assertEqual(reader.raw(reader.bl()), bytes.fromhex("1801012d010200273000"))
        self.assertEqual(reader.bs(), 1)
        self.assertEqual(reader.position, end)
        self.assertEqual([reader.handle() for _ in range(3)], [(4, 0x8D4), (4, 0x8D4), (3, 0)])

    def test_modular_ranges_and_signed_deltas(self):
        for value in (0, 63, 64, 127, 128, 16383, -1, -64, -10000):
            data = modular_char(value)
            self.assertEqual(read_modular_char(data, 0, len(data)), (value, len(data)))
        self.assertEqual(modular_short(32767), b"\xff\x7f")
        self.assertEqual(modular_short(65536), b"\x00\x80\x02\x00")

    def test_bounded_reads_and_finite_writes(self):
        with self.assertRaises(ValueError):
            BitReader(b"\0").bits(9)
        with self.assertRaises(ValueError):
            BitWriter().rd(float("inf"))
        with self.assertRaises(ValueError):
            BitWriter().text("A\0B")
        with self.assertRaises(ValueError):
            BitWriter().bs(65536)


class BinaryContainerTest(unittest.TestCase):
    def drawing(self):
        drawing = Drawing([("建-墙-砖", 7, True)])
        drawing.add_line((0.0, 0.0, -550.0), (3048.0, 0.0, -550.0), "建-墙-砖")
        drawing.add_polyline(((0.0, 0.0), (1000.0, 0.0), (1000.0, 2000.0), (0.0, 2000.0)))
        return drawing

    def test_direct_binary_container(self):
        data = self.drawing().bytes()
        report = inspect_container(data)
        self.assertEqual(data[:6], b"AC1015")
        self.assertEqual(set(report["sections"]), {0, 1, 2})
        self.assertEqual(sum(row["type_code"] == 0x13 for row in report["objects"]), 1)

    def test_crc_detects_object_damage(self):
        data = bytearray(self.drawing().bytes())
        report = inspect_container(data)
        data[report["objects"][-1]["offset"] + 5] ^= 1
        with self.assertRaisesRegex(ValueError, "checksums"):
            inspect_container(data)

    def test_crc_detects_section_damage(self):
        data = bytearray(self.drawing().bytes())
        data[100] ^= 1
        with self.assertRaisesRegex(ValueError, "checksums"):
            inspect_container(data)

    def test_map_pages_reset_deltas(self):
        mapping = object_map([(1 + index * 3, 1000 + index * 41) for index in range(5000)])
        offset = pages = count = 0
        while offset < len(mapping):
            length = struct.unpack_from(">H", mapping, offset)[0]
            self.assertLessEqual(length, 2032)
            self.assertEqual(struct.unpack_from(">H", mapping, offset + length)[0], crc16(mapping[offset:offset + length]))
            cursor = offset + 2
            handle = address = 0
            while cursor < offset + length:
                delta, cursor = read_modular_char(mapping, cursor, offset + length)
                handle += delta
                delta, cursor = read_modular_char(mapping, cursor, offset + length)
                address += delta
                self.assertEqual((handle, address), (1 + count * 3, 1000 + count * 41))
                count += 1
            offset += length + 2
            pages += 1
        self.assertEqual(count, 5000)
        self.assertGreater(pages, 2)


if __name__ == "__main__":
    unittest.main()
