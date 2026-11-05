"""CPU spawn regressions for the real loader, HDF5 crops and batched masking.

Run directly with Python/unittest; pytest is not required. All HDF5 files are
small synthetic fixtures in TemporaryDirectory, never the training dataset.
"""

import functools
import hashlib
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import h5py
import numpy as np
import torch
from upath import UPath

from olmoearth_pretrain.data.collate import collate_double_masked_batched
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data import dataloader as dataloader_module
from olmoearth_pretrain.data.dataloader import (
    OlmoEarthDataLoader,
    OlmoEarthDataLoaderConfig,
)
from olmoearth_pretrain.data.dataset import OlmoEarthDataset
from olmoearth_pretrain.data.transform import FlipAndRotateSpace
from olmoearth_pretrain.train.masking import MaskingConfig


def recording_collator(batch, *, masking_strategy, fail=False):
    """Exercise production crop/transform/masking and return worker observations."""
    if fail:
        raise ValueError("synthetic collator failure")
    info = torch.utils.data.get_worker_info()
    dl = info.dataset.data_loader if info is not None else None
    ids = [int(sample.sentinel2_l2a.min() // 10000) for _, sample in batch]
    crop = [sample.sentinel2_l2a.copy() for _, sample in batch]
    patch, a, b = collate_double_masked_batched(
        batch,
        transform=FlipAndRotateSpace(),
        masking_strategy=masking_strategy,
        masking_strategy_b=None,
    )
    h5_handles = []
    for descriptor in Path("/proc/self/fd").iterdir():
        try:
            target = descriptor.readlink()
        except FileNotFoundError:
            continue
        if target.suffix in (".h5", ".hdf5"):
            h5_handles.append(str(target))
    return (
        {
            "ids": ids,
            "pid": os.getpid(),
            "epoch": dl.epoch if dl else None,
            "seed": dl.seed if dl else None,
            "start": dl.batches_processed if dl else None,
            "shape": [patch, *a.sentinel2_l2a.shape],
            "crop": crop,
            "mask_a": a.sentinel2_l2a_mask,
            "mask_b": b.sentinel2_l2a_mask,
            "random": [
                random.random(),
                float(np.random.random()),
                float(torch.rand(())),
            ],
            "cuda_initialized": torch.cuda.is_initialized(),
            "h5_handles": h5_handles,
        },
    )


class _FakeOSSClient:
    """No-network client placeholder for worker-side prefetch cleanup tests."""

    def close(self):
        pass


def _read_local_h5_as_oss(path, on_client_create):
    """Emulate a complete GET while retaining the production prefetch flow."""
    on_client_create(_FakeOSSClient())
    return Path(str(path)).read_bytes()


def scalar_recording_collator(batch):
    """Keep generator startup tests independent of tensor IPC reconstruction."""
    info = torch.utils.data.get_worker_info()
    return (
        {
            "ids": [int(sample.sentinel2_l2a.min() // 10000) for _, sample in batch],
            "pid": os.getpid(),
            "epoch": info.dataset.data_loader.epoch,
        },
    )


class DataLoaderLifecycleTests(unittest.TestCase):
    """Use the production loader through its public epoch/state interfaces."""

    def setUp(self):
        """Create independent tiny local HDF5 samples without dataset preparation."""
        self.tmp = tempfile.TemporaryDirectory(prefix="olmoearth-lifecycle-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "27"
        self.data.mkdir()
        spatial = np.arange(16 * 16 * 4, dtype=np.float32).reshape(16, 16, 4, 1)
        for idx in range(27):
            # The production filename pattern is used below to avoid duplicating it.
            dataset = self.dataset()
            path = dataset._get_h5_file_path(idx)
            with h5py.File(str(path), "w") as f:
                f["timestamps"] = np.array([[1, m, 2020] for m in range(4)])
                f["sentinel2_l2a"] = np.repeat(
                    spatial + idx * 10000, Modality.SENTINEL2_L2A.num_bands, axis=-1
                )
                f["sentinel1"] = np.repeat(spatial + 1, 2, axis=-1)
                f["worldcover"] = np.ones((16, 16, 1, 1), dtype=np.float32)

    def dataset(self):
        """Use real Dataset.__getitem__, read_h5_file and random cropping."""
        dataset = OlmoEarthDataset(
            h5py_dir=UPath(self.data),
            training_modalities=["sentinel2_l2a", "sentinel1", "worldcover"],
            dtype=np.float32,
            max_sequence_length=4,
            normalize=False,
        )
        dataset.sample_indices = np.arange(27)
        dataset.latlon_distribution = np.zeros((27, 2))
        return dataset

    def loader(
        self,
        *,
        reuse=True,
        workers=2,
        rank=0,
        world=1,
        fail=False,
        inflight=1,
        context="spawn",
    ):
        """Build real spawn workers; None tests the unconfigured default path."""
        masking = MaskingConfig(
            strategy_config={
                "type": "random_time_with_decode",
                "only_decode_modalities": ["worldcover"],
                "encode_ratio": 0.5,
                "decode_ratio": 0.5,
                "random_ratio": 0.5,
            }
        ).build()
        loader = OlmoEarthDataLoader(
            dataset=self.dataset(),
            work_dir=UPath(self.root / f"indices-{rank}"),
            global_batch_size=4,
            min_patch_size=1,
            max_patch_size=2,
            sampled_hw_p_list=[2, 3],
            token_budget=10000,
            max_timesteps=4,
            tile_size=16,
            seed=3622,
            num_workers=workers,
            prefetch_factor=2,
            target_device_type="cpu",
            multiprocessing_context=context,
            dp_world_size=world,
            dp_rank=rank,
            collator=functools.partial(
                recording_collator, masking_strategy=masking, fail=fail
            ),
            masking_strategy=masking,
            max_inflight_oss_gets_per_worker=inflight,
            **({} if reuse is None else {"reuse_workers_across_epochs": reuse}),
        )
        self.addCleanup(lambda: getattr(loader, "close", lambda: None)())
        return loader

    @staticmethod
    def digest(batch):
        """Compare all random outputs without including process identity."""
        record = batch[0]
        h = hashlib.sha256()
        h.update(repr((record["ids"], record["shape"], record["random"])).encode())
        for value in [*record["crop"], record["mask_a"], record["mask_b"]]:
            array = value.numpy() if isinstance(value, torch.Tensor) else value
            h.update(str(array.shape).encode())
            h.update(array.tobytes())
        return h.hexdigest()

    def assert_order(self, loader, batches, start=0):
        """Verify exact global-batch order followed by DP sample striding."""
        indices = loader.get_global_indices().reshape(-1, loader.global_batch_size)
        expected = indices[start:, loader.dp_rank :: loader.dp_world_size].tolist()
        self.assertEqual([b[0]["ids"] for b in batches], expected)
        self.assertFalse(any(b[0]["cuda_initialized"] for b in batches))
        self.assertTrue(all(not b[0]["h5_handles"] for b in batches))

    def workers(self, loader):
        """Hold process objects for post-close liveness/exit-code assertions."""
        return tuple(loader._torch_loader._iterator._workers)

    def assert_closed(self, workers):
        """Every owned worker must have been reaped without forced termination."""
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertTrue(all(worker.exitcode == 0 for worker in workers))

    def test_reuses_workers_and_updates_epoch(self):
        """Epoch two must use the same processes and its own sample sequence."""
        loader = self.loader()
        epochs = []
        for epoch in (1, 2):
            loader.reshuffle(epoch)
            expected = loader.get_global_indices().reshape(-1, 4).tolist()
            batches = list(loader)
            self.assertEqual([b[0]["ids"] for b in batches], expected)
            self.assertEqual({b[0]["epoch"] for b in batches}, {epoch})
            self.assertEqual(loader.batches_processed, 6)
            epochs.append({b[0]["pid"] for b in batches})
            loader.reset()
        self.assertEqual(len(epochs[0]), 2)
        self.assertEqual(epochs[0], epochs[1])

    def test_bounded_oss_prefetch_preserves_order_and_rng(self):
        """Threaded GET lookahead keeps the same CPU tensors and epoch RNG."""
        outputs = {}
        with (
            patch.object(dataloader_module, "_is_oss_full_get_path", return_value=True),
            patch.object(
                dataloader_module, "_download_oss_payload", _read_local_h5_as_oss
            ),
        ):
            for inflight in (1, 2):
                loader = self.loader(workers=2, inflight=inflight, context="fork")
                epochs = []
                for epoch in (1, 2):
                    loader.reshuffle(epoch)
                    batches = list(loader)
                    self.assert_order(loader, batches)
                    epochs.append([self.digest(batch) for batch in batches])
                    loader.reset()
                workers = self.workers(loader)
                loader.close()
                self.assert_closed(workers)
                outputs[inflight] = epochs
        self.assertEqual(outputs[1], outputs[2])

    def test_default_keeps_legacy_path(self):
        """An omitted flag retains construction, sample order and checkpoint keys."""
        loader = self.loader(reuse=None)
        self.assertFalse(loader.reuse_workers_across_epochs)
        pids = []
        for epoch in (1, 2):
            loader.reshuffle(epoch)
            batches = list(loader)
            self.assert_order(loader, batches)
            self.assertIsNone(loader._torch_loader)
            pids.append({b[0]["pid"] for b in batches})
            loader.reset()
        self.assertTrue(pids[0].isdisjoint(pids[1]))
        self.assertEqual(
            set(loader.state_dict()),
            {
                "dataset_fingerprint_version",
                "dataset_fingerprint",
                "batches_processed",
                "seed",
                "epoch",
            },
        )

    def test_uneven_workers_and_dp_striding(self):
        """Six batches over four workers preserve order on both simulated DP ranks."""
        for rank in (0, 1):
            loader = self.loader(workers=4, rank=rank, world=2)
            previous = None
            for epoch in (1, 2):
                loader.reshuffle(epoch)
                batches = list(loader)
                self.assert_order(loader, batches)
                pids = {b[0]["pid"] for b in batches}
                self.assertEqual(len(pids), 4)
                if previous is not None:
                    self.assertEqual(pids, previous)
                previous = pids
                loader.reset()
            loader.close()

    def test_epoch_rng_is_reproducible_and_parent_rng_is_untouched(self):
        """Real crops, transforms and two masked views replay under epoch_v1."""
        runs = []
        for _ in range(2):
            loader = self.loader()
            loader.shuffle = False
            epochs = []
            for epoch in (1, 2):
                loader.reshuffle(epoch)
                parent_python = random.getstate()
                parent_numpy = np.random.get_state()
                parent_torch = torch.get_rng_state().clone()
                batches = list(loader)
                self.assertEqual(random.getstate(), parent_python)
                np.testing.assert_equal(np.random.get_state(), parent_numpy)
                self.assertTrue(torch.equal(torch.get_rng_state(), parent_torch))
                self.assert_order(loader, batches)
                self.assertNotEqual(batches[0][0]["random"], batches[2][0]["random"])
                self.assertTrue(
                    any(
                        not torch.equal(b[0]["mask_a"], b[0]["mask_b"]) for b in batches
                    )
                )
                epochs.append([self.digest(b) for b in batches])
                loader.reset()
            self.assertNotEqual(epochs[0], epochs[1])
            runs.append(epochs)
            loader.close()
        self.assertEqual(runs[0], runs[1])

    def _double_masked_loader(self, *, target_device_type, **kwargs):
        loader = self.loader(**kwargs)
        loader.target_device_type = target_device_type
        loader.collator = functools.partial(
            collate_double_masked_batched,
            transform=FlipAndRotateSpace(),
            masking_strategy=loader.masking_strategy,
            masking_strategy_b=None,
        )
        return loader

    def test_cuda_target_without_accelerator_preserves_batch_interface(self):
        """Both loader modes unwrap the batch when PyTorch skips CUDA pinning."""
        for reuse in (True, False):
            with (
                self.subTest(reuse=reuse),
                torch.random.fork_rng(devices=[]),
                patch.object(torch.accelerator, "is_available", return_value=False),
            ):
                baseline = self._double_masked_loader(
                    target_device_type="cpu", reuse=reuse, workers=1
                )
                fallback = self._double_masked_loader(
                    target_device_type="cuda", reuse=reuse, workers=1
                )
                baseline.reshuffle(1)
                fallback.reshuffle(1)
                torch.manual_seed(42)
                expected_batches = list(baseline)
                torch.manual_seed(42)
                with self.assertWarnsRegex(UserWarning, "no accelerator"):
                    actual_batches = list(fallback)

                self.assertEqual(len(actual_batches), baseline.total_batches)
                for actual, expected in zip(
                    actual_batches, expected_batches, strict=True
                ):
                    self.assertIs(type(actual), list)
                    self.assertEqual(len(actual), 4)
                    for index in (1, 2):
                        self.assertIs(type(actual[index]), type(expected[index]))
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                baseline.close()
                fallback.close()

    def test_pin_shared_inputs_preserves_epoch_rng_and_batch_interface(self):
        """Run the real loader and pin thread with CPU copies in place of CUDA pinning."""
        baseline = self._double_masked_loader(target_device_type="cpu")
        pinned = self._double_masked_loader(target_device_type="cuda")

        pinned_inputs = []

        def copy_for_pin(tensor):
            pinned_inputs.append(tensor)
            return tensor.clone()

        with (
            patch.object(torch.accelerator, "is_available", return_value=True),
            patch.object(
                torch.accelerator,
                "current_accelerator",
                return_value=torch.device("cuda"),
            ),
            patch.object(torch.accelerator, "current_device_index", return_value=0),
            patch.object(torch.accelerator, "set_device_index"),
            patch.object(torch.Tensor, "pin_memory", copy_for_pin),
        ):
            for epoch in (1, 2):
                baseline.reshuffle(epoch)
                pinned.reshuffle(epoch)
                expected_batches = list(baseline)
                pinned_inputs.clear()
                actual_batches = list(pinned)
                self.assertEqual(len(actual_batches), len(expected_batches))
                expected_pin_calls = 0
                for expected, actual in zip(
                    expected_batches, actual_batches, strict=True
                ):
                    self.assertIs(type(actual), type(expected))
                    self.assertEqual((actual[0], actual[3]), (expected[0], expected[3]))
                    view_a, view_b = actual[1:3]
                    for ref, view in zip(expected[1:3], actual[1:3], strict=True):
                        for field, value in ref.as_dict().items():
                            torch.testing.assert_close(
                                getattr(view, field), value, rtol=0, atol=0
                            )
                    for field in ("timestamps", *view_a.modalities):
                        self.assertIs(getattr(view_a, field), getattr(view_b, field))
                    for modality in view_a.modalities:
                        field = view_a.get_masked_modality_name(modality)
                        self.assertIsNot(getattr(view_a, field), getattr(view_b, field))
                    expected_pin_calls += 1 + 3 * len(view_a.modalities)
                self.assertEqual(len(pinned_inputs), expected_pin_calls)
                self.assertEqual(
                    len({id(tensor) for tensor in pinned_inputs}), expected_pin_calls
                )
                baseline.reset()
                pinned.reset()

    def test_shapes_match_legacy_epoch_formula(self):
        """Reuse does not change full-epoch PCG64-driven batch geometry."""
        shapes = []
        for reuse in (False, True):
            loader = self.loader(reuse=reuse)
            epochs = []
            for epoch in (1, 2):
                loader.reshuffle(epoch)
                epochs.append([b[0]["shape"] for b in loader])
                loader.reset()
            shapes.append(epochs)
            loader.close()
        self.assertEqual(shapes[0], shapes[1])

    def test_resume_discards_prefetch_and_invalidates_old_iterator(self):
        """Restoring an active loader cannot yield any batch from the old runtime."""
        loader = self.loader()
        loader.reshuffle(2)
        iterator = iter(loader)
        next(iterator)
        next(iterator)
        workers = self.workers(loader)
        state = json.loads(json.dumps(loader.state_dict()))
        self.assertEqual(state["batches_processed"], 2)
        loader.load_state_dict(state)
        self.assert_closed(workers)
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            next(iterator)
        loader.reshuffle(state["epoch"])
        remainder = list(loader)
        self.assert_order(loader, remainder, start=2)
        self.assertEqual({b[0]["start"] for b in remainder}, {2})
        self.assertEqual(loader.batches_processed, 6)

        # A separate loader with a different configured seed must honor saved seed.
        fresh = self.loader()
        fresh.seed = 777
        fresh.load_state_dict(state)
        fresh.reshuffle(state["epoch"])
        self.assertEqual(fresh.seed, 3622)
        self.assertEqual(
            [self.digest(b) for b in fresh], [self.digest(b) for b in remainder]
        )

    def test_old_checkpoint_and_next_epoch_boundary(self):
        """Legacy checkpoint keys resume both partial and subsequent epochs."""
        old = self.loader(reuse=False)
        old.reshuffle(1)
        iterator = iter(old)
        next(iterator)
        state = json.loads(json.dumps(old.state_dict()))
        iterator.close()
        restored = self.loader()
        restored.load_state_dict(state)
        restored.reshuffle(1)
        self.assert_order(restored, list(restored), start=1)
        restored.reset()
        boundary_state = restored.state_dict()
        restored.load_state_dict(boundary_state)
        restored.reshuffle(2)  # Trainer advances its epoch after reset().
        batches = list(restored)
        self.assert_order(restored, batches)
        self.assertEqual({b[0]["start"] for b in batches}, {0})
        self.assertEqual({b[0]["epoch"] for b in batches}, {2})

    def test_early_close_is_idempotent(self):
        """A partial epoch closes cleanly and a retained iterator cannot restart it."""
        loader = self.loader()
        loader.reshuffle(1)
        iterator = iter(loader)
        next(iterator)
        workers = self.workers(loader)
        loader.close()
        loader.close()
        self.assert_closed(workers)
        self.assertIsNone(loader._torch_loader)
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            next(iterator)

    def test_generator_close_reaps_partial_epoch(self):
        """Closing the public generator also releases its owned persistent runtime."""
        loader = self.loader()
        loader.reshuffle(1)
        iterator = iter(loader)
        next(iterator)
        workers = self.workers(loader)
        iterator.close()
        self.assert_closed(workers)
        self.assertIsNone(loader._torch_loader)

    def test_active_epoch_cannot_be_overwritten(self):
        """Reject overlapping iterators/reshuffle without damaging the active epoch."""
        loader = self.loader()
        loader.reshuffle(1)
        iterator = iter(loader)
        first = next(iterator)
        with self.assertRaisesRegex(RuntimeError, "close"):
            loader.reshuffle(2)
        with self.assertRaisesRegex(RuntimeError, "one iterator"):
            next(iter(loader))
        self.assert_order(loader, [first, *iterator])

    def test_worker_exception_propagates_and_closes(self):
        """A failed production fetch cannot leave persistent children running."""
        baseline = {p.pid for p in mp.active_children()}
        loader = self.loader(fail=True)
        loader.reshuffle(1)
        with self.assertRaisesRegex(ValueError, "synthetic collator failure"):
            next(iter(loader))
        self.assertIsNone(loader._torch_loader)
        self.assertEqual({p.pid for p in mp.active_children()}, baseline)

    def test_in_memory_and_zero_worker_fallback(self):
        """Unsupported reuse modes follow the original index/data loading path."""
        for workers, in_memory in ((0, False), (2, True)):
            loader = self.loader(workers=workers)
            for epoch in (1, 2):
                loader.reshuffle(epoch, in_memory=in_memory)
                self.assert_order(loader, list(loader))
                self.assertIsNone(loader._torch_loader)
                self.assertEqual(loader.state_dict()["worker_rng_policy"], "legacy")
                loader.reset()

    def test_single_worker_reuse_and_exit(self):
        """A one-worker configuration reuses its process and shuts it down cleanly."""
        loader = self.loader(workers=1)
        workers = None
        for epoch in (1, 2):
            loader.reshuffle(epoch)
            batches = list(loader)
            self.assert_order(loader, batches)
            current = self.workers(loader)
            self.assertEqual(len({worker.pid for worker in current}), 1)
            if workers is None:
                workers = current
            else:
                self.assertEqual(current, workers)
            loader.reset()
        loader.close()
        self.assert_closed(workers)

    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA must be visible to test CPU-default mismatch"
    )
    def test_reused_loader_matches_cpu_factory_default_on_cuda_host(self):
        """CUDA availability must not override a CPU PyTorch factory default."""
        previous_default = torch.get_default_device()
        loader = None
        try:
            torch.set_default_device("cpu")
            self.assertEqual(torch.get_default_device(), torch.device("cpu"))
            loader = self.loader(workers=1)
            loader.reshuffle(1)
            batches = list(loader)
            self.assert_order(loader, batches)
        finally:
            if loader is not None:
                loader.close()
            torch.set_default_device(previous_default)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA factory default")
    def test_cuda_factory_default_generator_and_worker_reset(self):
        """Exercise actual loader base-seed creation without tensor IPC payloads."""
        # CUDA initialization cannot be undone; isolate it from CPU parent-state tests.
        if os.environ.get("OLMOEARTH_CUDA_GENERATOR_TEST_CHILD") != "1":
            env = os.environ.copy()
            env["OLMOEARTH_CUDA_GENERATOR_TEST_CHILD"] = "1"
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    f"{type(self).__name__}.{self._testMethodName}",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return
        previous_default = torch.get_default_device()
        loader = None
        try:
            torch.set_default_device("cuda")
            loader = self.loader(workers=1)
            loader.collator = scalar_recording_collator
            previous_pids = None
            for epoch in (1, 2):
                loader.reshuffle(epoch)
                rng_state = torch.cuda.get_rng_state().clone()
                batches = list(loader)
                self.assertTrue(torch.equal(rng_state, torch.cuda.get_rng_state()))
                self.assertEqual(
                    loader._torch_loader.generator.device, torch.get_default_device()
                )
                expected = loader.get_global_indices().reshape(-1, 4).tolist()
                self.assertEqual([batch[0]["ids"] for batch in batches], expected)
                self.assertEqual({batch[0]["epoch"] for batch in batches}, {epoch})
                pids = {batch[0]["pid"] for batch in batches}
                if previous_pids is not None:
                    self.assertEqual(previous_pids, pids)
                previous_pids = pids
                loader.reset()
            workers = self.workers(loader)
            loader.close()
            self.assert_closed(workers)
        finally:
            if loader is not None:
                loader.close()
            torch.set_default_device(previous_default)

    def test_persistent_false_and_empty_remainder(self):
        """Honor an explicit native persistence opt-out and exhausted checkpoints."""
        loader = self.loader()
        loader.persistent_workers = False
        loader.reshuffle(1)
        self.assert_order(loader, list(loader))
        self.assertIsNone(loader._torch_loader)
        restored = self.loader()
        restored.load_state_dict(loader.state_dict())
        restored.reshuffle(1)
        self.assertEqual(list(restored), [])
        self.assertEqual(restored.batches_processed, 6)

    def test_structural_change_rebuilds_workers(self):
        """A supported structural setting change cannot leave stale worker copies."""
        loader = self.loader()
        loader.reshuffle(1)
        list(loader)
        workers = self.workers(loader)
        loader.reset()
        loader.global_batch_size = 8
        loader.reshuffle(2)
        self.assert_order(loader, list(loader))
        self.assert_closed(workers)
        self.assertNotEqual(
            {p.pid for p in workers}, {p.pid for p in self.workers(loader)}
        )

    def test_config_default_and_opt_in(self):
        """Config exposes the opt-in, forwards it and keeps CPU mode explicit."""
        config = OlmoEarthDataLoaderConfig(
            work_dir=str(self.root / "config"),
            global_batch_size=4,
            min_patch_size=1,
            max_patch_size=2,
            sampled_hw_p_list=[2],
            seed=42,
            target_device_type="cpu",
            masking_config=MaskingConfig(strategy_config={"type": "random"}),
        )
        self.assertFalse(config.reuse_workers_across_epochs)
        config.reuse_workers_across_epochs = True
        loader = config.build(self.dataset())
        self.addCleanup(loader.close)
        self.assertTrue(loader.reuse_workers_across_epochs)

    def test_seed_and_epoch_snapshot_are_refreshed(self):
        """Persistent workers observe a new seed and a reset consumed-batch count."""
        loader = self.loader()
        loader.reshuffle(1)
        list(loader)
        processes = self.workers(loader)
        loader.reset()
        loader.seed = 123
        loader.reshuffle(3)
        batches = list(loader)
        self.assert_order(loader, batches)
        self.assertEqual({b[0]["seed"] for b in batches}, {123})
        self.assertEqual({b[0]["epoch"] for b in batches}, {3})
        self.assertEqual({b[0]["start"] for b in batches}, {0})
        self.assertEqual(processes, self.workers(loader))


if __name__ == "__main__":
    unittest.main(verbosity=2)
