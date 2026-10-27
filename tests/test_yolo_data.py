import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from dr_data import Box, ImageRecord, convert_box
from yolo_loader import (
    combine_train_test_assignments,
    create_split_assignments,
    from_yolo,
    patient_id,
    to_yolo,
)


class BoxConversionTests(unittest.TestCase):
    def test_scale_and_clamp_csv_box(self):
        row = {
            "left_coordinate": "-10",
            "top_coordinate": "20",
            "width": "30",
            "height": "40",
            "image_scale_x": "0.5",
            "image_scale_y": "0.5",
            "label": "Retinal Microaneurysm",
        }
        box, reason = convert_box(row, 100, 100)
        self.assertIsNone(reason)
        self.assertEqual(box, Box(0.0, 40.0, 40.0, 100.0, "Retinal Microaneurysm"))

    def test_reject_nonpositive_and_outside_boxes(self):
        base = {
            "left_coordinate": "10",
            "top_coordinate": "10",
            "width": "5",
            "height": "5",
            "image_scale_x": "1",
            "image_scale_y": "1",
            "label": "Retinal Microaneurysm",
        }
        zero_box, zero_reason = convert_box({**base, "width": "0"}, 100, 100)
        outside_box, outside_reason = convert_box(
            {**base, "left_coordinate": "200"}, 100, 100
        )
        self.assertIsNone(zero_box)
        self.assertEqual(zero_reason, "nonpositive")
        self.assertIsNone(outside_box)
        self.assertEqual(outside_reason, "outside")


class YoloFormatTests(unittest.TestCase):
    def test_numeric_round_trip_is_under_one_pixel(self):
        box = Box(31.25, 52.5, 84.75, 110.125, "Retinal Microaneurysm")
        line = to_yolo(box, 640, 480, 9)
        class_id, restored = from_yolo(line, 640, 480)
        expected = (box.x_min, box.y_min, box.x_max, box.y_max)
        self.assertEqual(class_id, 9)
        self.assertLess(max(abs(left - right) for left, right in zip(expected, restored)), 1)

    def test_split_is_deterministic_and_groups_both_eyes(self):
        with TemporaryDirectory() as directory:
            image_root = Path(directory)
            records = []
            for filename in ("101_left.jpeg", "101_right.jpeg", "102_left.jpeg", "103_right.jpeg"):
                path = image_root / filename
                Image.new("RGB", (10, 10)).save(path)
                records.append(ImageRecord(filename, path, 10, 10, []))

            first, patient_count = create_split_assignments(records, seed=17)
            second, _ = create_split_assignments(records, seed=17)
            self.assertEqual(first, second)
            self.assertEqual(patient_count, 3)
            self.assertEqual(first["101_left.jpeg"], first["101_right.jpeg"])
            self.assertEqual(patient_id("101_left.jpeg"), "101")
            self.assertEqual(set(first.values()), {"train", "val"})

    def test_common_classes_are_distributed_across_splits(self):
        with TemporaryDirectory() as directory:
            image_root = Path(directory)
            records = []
            for patient_number in range(12):
                filename = f"{patient_number}_left.jpeg"
                path = image_root / filename
                Image.new("RGB", (100, 100)).save(path)
                label = "Retinal Microaneurysm" if patient_number < 6 else "Retinal Dot Haemorrhage"
                box = Box(10, 10, 20, 20, label)
                records.append(ImageRecord(filename, path, 100, 100, [box]))

            assignments, patient_count = create_split_assignments(records, seed=23)
            labels_by_split = {"train": set(), "val": set()}
            for record in records:
                labels_by_split[assignments[record.filename]].update(
                    box.label for box in record.boxes
                )

            self.assertEqual(patient_count, 12)
            self.assertTrue(all("Retinal Microaneurysm" in labels for labels in labels_by_split.values()))
            self.assertTrue(all("Retinal Dot Haemorrhage" in labels for labels in labels_by_split.values()))

    def test_official_test_records_are_not_repartitioned(self):
        training = ImageRecord("101_left.jpeg", Path("101_left.jpeg"), 10, 10, [])
        testing = ImageRecord("202_right.jpeg", Path("202_right.jpeg"), 10, 10, [])
        assignments = combine_train_test_assignments([training], [testing], {training.filename: "val"})
        self.assertEqual(assignments, {"101_left.jpeg": "val", "202_right.jpeg": "test"})

    def test_rejects_patient_in_both_source_folders(self):
        training = ImageRecord("101_left.jpeg", Path("101_left.jpeg"), 10, 10, [])
        testing = ImageRecord("101_right.jpeg", Path("101_right.jpeg"), 10, 10, [])
        with self.assertRaisesRegex(ValueError, "Patient IDs occur in both"):
            combine_train_test_assignments([training], [testing], {training.filename: "train"})


if __name__ == "__main__":
    unittest.main()