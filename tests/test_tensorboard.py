"""CPU-only smoke tests with generated audio; no downloaded datasets required."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train


class TensorBoardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.audio = cls.root / "audio"
        # Enough per genre for the actual stratified 70/15/15 split.
        for label, genre in enumerate(train.GTZAN_GENRES):
            folder = cls.audio / genre
            folder.mkdir(parents=True)
            for i in range(10):
                t = np.arange(1103) / 22050
                wav = 0.1 * np.sin(2 * np.pi * (200 + label * 20 + i) * t)
                sf.write(folder / f"{i}.wav", wav, 22050)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def invoke(self, representation, out, epochs=1):
        argv = ["train.py", "--data-root", str(self.audio),
                "--representation", representation, "--epochs", str(epochs),
                "--batch-size", "4", "--out-dir", str(out),
                "--log-dir", str(self.root / "events")]
        with patch.object(sys, "argv", argv), \
             patch.object(train, "pick_device", return_value=torch.device("cpu")), \
             contextlib.redirect_stdout(io.StringIO()):
            train.main()

    def read_metrics(self, out, representation):
        metrics = json.loads((out / f"metrics_{representation}.json").read_text())
        events = EventAccumulator(metrics["tensorboard_run"]).Reload()
        for tag, key in [("loss/train", "train_loss"),
                         ("loss/validation", "val_loss"),
                         ("accuracy/train", "train_acc"),
                         ("accuracy/validation", "val_acc")]:
            records = events.Scalars(tag)
            self.assertEqual([r.step for r in records],
                             [h["epoch"] for h in metrics["history"]])
            for record, history in zip(records, metrics["history"]):
                self.assertAlmostEqual(record.value, history[key], places=5)
        for tag, key in [("loss/test", "test_loss"), ("accuracy/test", "test_acc")]:
            records = events.Scalars(tag)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].step, metrics["best_epoch"])
            self.assertAlmostEqual(records[0].value, metrics[key], places=5)
        self.assertIn("settings/text_summary", events.Tags()["tensors"])
        checkpoint = torch.load(metrics["checkpoint"], weights_only=False)
        self.assertEqual(checkpoint["epoch"], metrics["best_epoch"])
        self.assertEqual(checkpoint["val_acc"], metrics["best_val_acc"])
        return metrics

    def test_real_models_both_representations(self):
        paths = []
        for representation in ["spectrogram", "waveform"]:
            with self.subTest(representation=representation):
                out = self.root / representation
                self.invoke(representation, out)
                metrics = self.read_metrics(out, representation)
                self.assertEqual(metrics["settings"]["train_size"], 70)
                self.assertEqual(metrics["settings"]["val_size"], 15)
                self.assertEqual(metrics["settings"]["test_size"], 15)
                paths.append(metrics["tensorboard_run"])
        self.assertEqual(len(set(paths)), 2)

    def test_best_checkpoint_not_last_and_unique_runs(self):
        original_evaluate = train.evaluate
        saved_state = {}
        calls = 0

        def controlled_evaluate(model, loader, device):
            nonlocal calls
            calls += 1
            loss, acc, y, pred = original_evaluate(model, loader, device)
            if calls == 1:
                saved_state.update({k: v.clone() for k, v in model.state_dict().items()})
                return loss, 0.8, y, pred
            if calls == 2:
                self.assertTrue(any(not torch.equal(v, saved_state[k])
                                    for k, v in model.state_dict().items()))
                return loss, 0.2, y, pred
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, saved_state[k]), k)
            return loss, acc, y, pred

        out = self.root / "best-check"
        with patch.object(train, "evaluate", side_effect=controlled_evaluate):
            self.invoke("spectrogram", out, epochs=2)
        metrics = self.read_metrics(out, "spectrogram")
        self.assertEqual(metrics["best_epoch"], 1)
        self.assertEqual(calls, 3)
        old_run = metrics["tensorboard_run"]
        self.invoke("spectrogram", out)
        newer = self.read_metrics(out, "spectrogram")
        self.assertNotEqual(newer["tensorboard_run"], old_run)
        self.assertTrue(Path(old_run).is_dir())

    def test_failure_closes_writer_without_test_metrics(self):
        for error in (KeyboardInterrupt, RuntimeError):
            with self.subTest(error=error.__name__):
                self.check_failed_run(error)

    def check_failed_run(self, error):
        writers = []
        real_writer = train.SummaryWriter

        def make_writer(*args, **kwargs):
            writer = real_writer(*args, **kwargs)
            writers.append(writer)
            return writer

        out = self.root / error.__name__
        with patch.object(train, "SummaryWriter", side_effect=make_writer), \
             patch.object(train, "evaluate", side_effect=error), \
             self.assertRaises(error):
            self.invoke("spectrogram", out)
        self.assertEqual(len(writers), 1)
        self.assertIsNone(writers[0].file_writer)
        events = EventAccumulator(writers[0].log_dir).Reload()
        self.assertNotIn("accuracy/test", events.Tags()["scalars"])
        self.assertFalse((out / "metrics_spectrogram.json").exists())

    def test_invalid_epochs_rejected_before_loading_checkpoint(self):
        with patch.object(sys, "argv", ["train.py", "--epochs", "0"]), \
             contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit) as caught:
            train.main()
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
