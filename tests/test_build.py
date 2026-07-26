import json
import shutil

import boto3
import pytest
import yaml
from moto import mock_aws

from dronesynth.config import ConfigError, load_build_config
from dronesynth.datagen.build import BuildError, DatasetManifest, build_dataset
from dronesynth.datagen.split import SplitError
from dronesynth.ingest.manifest import RunManifest
from dronesynth.storage import LocalStorage, StorageNotPermitted

CONVERSION = {"threshold": 32, "min_box_area": 16, "class_map": {"0": "drone"}}
CONVERTER = {"repo": "drone-synthetic", "commit": "abc1234567", "dirty": False}
GENERATOR = {"repo": "drone-synth-render", "commit": "ffff000011", "dirty": False}
RANDOMIZATION = {"distance_start_cm": 4000, "yaw_deg": 30}


def make_run(
    root,
    run_id,
    *,
    seed,
    frames=3,
    boxes=1,
    randomization=None,
    generator=None,
    conversion=None,
    converter=None,
    claim_frames=None,
):
    """Fabricate one converted run: raw frames + manifest, annotations + provenance.

    The images are text rather than PNGs on purpose. A build never decodes them,
    it only copies them, so tagging each with its run and index makes it possible
    to assert that the right frame landed at the right key.
    """
    raw = root / "raw" / run_id
    (raw / "normal").mkdir(parents=True)

    annotations = []
    for index in range(frames):
        name = f"frame_{index:06d}.png"
        (raw / "normal" / name).write_text(f"{run_id}-image-{index}")
        annotations.append(
            {
                "frame_index": index,
                "normal": name,
                "width": 64,
                "height": 48,
                "boxes": [
                    {
                        "class_id": 0,
                        "x": 10,
                        "y": 12,
                        "w": 8,
                        "h": 6,
                        "mask_area": 48,
                        "fill_ratio": 1.0,
                        "components": 1,
                    }
                ]
                * boxes,
            }
        )

    manifest = RunManifest(
        run_id=run_id,
        captured_at="2026-07-26",
        frame_count=claim_frames if claim_frames is not None else frames,
        ue_map="SkyTestMap",
        drone_model="Quadcopter_A",
        camera_sequence="RunTemplate",
        randomization=RANDOMIZATION if randomization is None else randomization,
        generator=GENERATOR if generator is None else generator,
        seed=seed,
    )
    (raw / "manifest.json").write_text(json.dumps(manifest.to_dict()))

    annotations_dir = root / "datasets" / f"auto-{run_id}" / "annotations"
    annotations_dir.mkdir(parents=True)
    (annotations_dir / f"{run_id}.json").write_text(json.dumps(annotations))
    (annotations_dir / f"{run_id}.provenance.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "converted_at": "2026-07-26T14:30:24+00:00",
                "converter": CONVERTER if converter is None else converter,
                "conversion": CONVERSION if conversion is None else conversion,
            }
        )
    )


def build_config(root, runs, val_runs=(), dataset_root=None, raw_root=None):
    path = root / "build.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "runs": list(runs),
                "split": {"mode": "by_run", "val_runs": list(val_runs)},
                "storage": {
                    "raw_root": raw_root or str(root / "raw"),
                    "dataset_root": dataset_root or str(root / "datasets"),
                },
            }
        )
    )
    return load_build_config(path)


@pytest.fixture
def corpus(tmp_path):
    """Four runs, distinct scenes, ready to build."""
    for offset, run_id in enumerate(["run_a", "run_b", "run_c", "run_d"]):
        make_run(tmp_path, run_id, seed=100 + offset)
    return tmp_path


def test_build_end_to_end(corpus):
    config = build_config(corpus, ["run_a", "run_b", "run_c", "run_d"], val_runs=["run_c"])

    result = build_dataset(version="v002", config=config)

    totals = result.manifest.totals
    assert totals == {
        "runs": 4,
        "frames": 12,
        "boxes": 12,
        "train_runs": 3,
        "val_runs": 1,
        "train_frames": 9,
        "val_frames": 3,
    }

    yolo = corpus / "datasets" / "v002" / "yolo"
    train_images = sorted(p.name for p in (yolo / "images" / "train").iterdir())
    val_images = sorted(p.name for p in (yolo / "images" / "val").iterdir())
    assert len(train_images) == 9
    assert val_images == ["run_c_000000.png", "run_c_000001.png", "run_c_000002.png"]
    assert not any(name.startswith("run_c") for name in train_images)

    # Every image is accompanied by its label, on the same side.
    for subset, images in (("train", train_images), ("val", val_images)):
        labels = sorted(p.name for p in (yolo / "labels" / subset).iterdir())
        assert labels == [name.replace(".png", ".txt") for name in images]


def test_images_are_copied_from_the_run_they_belong_to(corpus):
    config = build_config(corpus, ["run_a", "run_b"], val_runs=["run_b"])
    build_dataset(version="v002", config=config)

    yolo = corpus / "datasets" / "v002" / "yolo"
    assert (yolo / "images" / "train" / "run_a_000001.png").read_text() == "run_a-image-1"
    assert (yolo / "images" / "val" / "run_b_000002.png").read_text() == "run_b-image-2"


def test_labels_are_derived_from_the_annotations(corpus):
    config = build_config(corpus, ["run_a"])
    build_dataset(version="v002", config=config)

    label = corpus / "datasets" / "v002" / "yolo" / "labels" / "train" / "run_a_000000.txt"
    # box x=10 y=12 w=8 h=6 in a 64x48 frame -> centre (14/64, 15/48), size (8/64, 6/48)
    assert label.read_text() == "0 0.218750 0.312500 0.125000 0.125000\n"


def test_dataset_yaml_names_come_from_the_conversion(corpus):
    config = build_config(corpus, ["run_a"])
    build_dataset(version="v002", config=config)

    descriptor = yaml.safe_load(
        (corpus / "datasets" / "v002" / "yolo" / "dataset.yaml").read_text()
    )
    assert descriptor == {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": {0: "drone"},
    }


def test_manifest_embeds_per_run_provenance(corpus):
    config = build_config(corpus, ["run_a", "run_b"], val_runs=["run_b"])

    result = build_dataset(version="v002", config=config)

    written = json.loads((corpus / "datasets" / "v002" / "manifest.json").read_text())
    assert DatasetManifest.from_dict(written) == result.manifest
    assert written["split"] == {"mode": "by_run", "train": ["run_a"], "val": ["run_b"]}
    assert written["conversion"] == CONVERSION

    by_id = {r["run_id"]: r for r in written["runs"]}
    assert by_id["run_b"]["subset"] == "val"
    assert by_id["run_a"]["seed"] == 100
    assert by_id["run_a"]["generator"] == GENERATOR
    assert by_id["run_a"]["converter"] == CONVERTER
    assert by_id["run_a"]["randomization"] == RANDOMIZATION
    assert by_id["run_a"]["frames"] == 3
    assert by_id["run_a"]["boxes"] == 3
    assert by_id["run_a"]["source_version"] == "auto-run_a"


def test_manifest_survives_its_inputs_being_deleted(corpus):
    """The record embeds rather than references, so pruning inputs cannot blind it."""
    config = build_config(corpus, ["run_a"])
    build_dataset(version="v002", config=config)
    shutil.rmtree(corpus / "raw" / "run_a")
    shutil.rmtree(corpus / "datasets" / "auto-run_a")

    written = json.loads((corpus / "datasets" / "v002" / "manifest.json").read_text())
    assert written["runs"][0]["seed"] == 100
    assert written["runs"][0]["generator"] == GENERATOR


def test_rebuilding_a_version_is_refused(corpus):
    config = build_config(corpus, ["run_a"])
    build_dataset(version="v002", config=config)

    with pytest.raises(BuildError, match="already exists"):
        build_dataset(version="v002", config=config)


def test_rebuilding_is_refused_even_when_the_preflight_check_cannot_look(corpus, monkeypatch):
    """The build role holds no ListBucket, so S3 cannot tell an absent key from a
    forbidden one. Immutability has to hold on the manifest write alone."""
    config = build_config(corpus, ["run_a"])
    build_dataset(version="v002", config=config)

    def blind(self, key):
        raise StorageNotPermitted("credentials lack read access")

    monkeypatch.setattr(LocalStorage, "exists", blind)

    with pytest.raises(BuildError, match="already has a manifest"):
        build_dataset(version="v002", config=config)


def test_run_without_provenance_is_refused(corpus):
    (corpus / "datasets" / "auto-run_b" / "annotations" / "run_b.provenance.json").unlink()
    config = build_config(corpus, ["run_a", "run_b"])

    with pytest.raises(BuildError, match="reconvert it before building"):
        build_dataset(version="v002", config=config)


def test_unconverted_run_is_refused(corpus):
    (corpus / "datasets" / "auto-run_b" / "annotations" / "run_b.json").unlink()
    config = build_config(corpus, ["run_a", "run_b"])

    with pytest.raises(BuildError, match="convert it before building"):
        build_dataset(version="v002", config=config)


def test_pruned_run_is_refused(corpus):
    (corpus / "raw" / "run_b" / "manifest.json").unlink()
    config = build_config(corpus, ["run_a", "run_b"])

    with pytest.raises(BuildError, match="never ingested, or it has been pruned"):
        build_dataset(version="v002", config=config)


def test_mixed_conversion_config_is_refused(tmp_path):
    """The live hazard: runs converted either side of the threshold change."""
    make_run(tmp_path, "run_old", seed=1, conversion={**CONVERSION, "threshold": 12})
    make_run(tmp_path, "run_new", seed=2)
    config = build_config(tmp_path, ["run_old", "run_new"])

    with pytest.raises(BuildError, match="2 different conversion configs"):
        build_dataset(version="v002", config=config)


def test_mixed_converter_build_is_refused(tmp_path):
    make_run(tmp_path, "run_a", seed=1)
    make_run(tmp_path, "run_b", seed=2, converter={**CONVERTER, "commit": "9999999999"})
    config = build_config(tmp_path, ["run_a", "run_b"])

    with pytest.raises(BuildError, match="2 different converter builds"):
        build_dataset(version="v002", config=config)


def test_unknown_val_run_is_refused(corpus):
    """split_runs' typo check, finally protecting something."""
    config = build_config(corpus, ["run_a", "run_b"], val_runs=["run_typo"])

    with pytest.raises(SplitError, match="not among the input runs"):
        build_dataset(version="v002", config=config)


def test_duplicate_input_runs_are_refused(corpus):
    with pytest.raises(ConfigError, match="duplicates"):
        build_config(corpus, ["run_a", "run_b", "run_a"])


def test_frame_count_disagreement_is_refused(tmp_path):
    make_run(tmp_path, "run_a", seed=1, frames=3, claim_frames=5)
    config = build_config(tmp_path, ["run_a"])

    with pytest.raises(BuildError, match="stale or the run is corrupt"):
        build_dataset(version="v002", config=config)


def test_same_scene_on_both_sides_is_refused(tmp_path):
    """Two renders of one scene: different run ids, same seed and config."""
    make_run(tmp_path, "run_first", seed=1102)
    make_run(tmp_path, "run_twin", seed=1102)
    config = build_config(tmp_path, ["run_first", "run_twin"], val_runs=["run_twin"])

    with pytest.raises(BuildError, match="both sides of the split"):
        build_dataset(version="v002", config=config)


def test_same_scene_on_one_side_is_fine(tmp_path):
    """A duplicate render is not itself an error; only straddling the split is."""
    make_run(tmp_path, "run_first", seed=1102)
    make_run(tmp_path, "run_twin", seed=1102)
    make_run(tmp_path, "run_other", seed=1103)
    config = build_config(
        tmp_path, ["run_first", "run_twin", "run_other"], val_runs=["run_other"]
    )

    result = build_dataset(version="v002", config=config)
    assert result.manifest.totals["train_runs"] == 2


def test_same_seed_under_a_different_scene_is_not_a_leak(tmp_path):
    """Seed alone would be too strict: the same draw under a different setup
    produces a genuinely different scene."""
    make_run(tmp_path, "run_a", seed=1102)
    make_run(tmp_path, "run_b", seed=1102, randomization={"distance_start_cm": 90000})
    config = build_config(tmp_path, ["run_a", "run_b"], val_runs=["run_b"])

    result = build_dataset(version="v002", config=config)
    assert result.manifest.totals == {
        "runs": 2,
        "frames": 6,
        "boxes": 6,
        "train_runs": 1,
        "val_runs": 1,
        "train_frames": 3,
        "val_frames": 3,
    }


def test_a_different_renderer_build_is_a_different_scene(tmp_path):
    make_run(tmp_path, "run_a", seed=1102)
    make_run(tmp_path, "run_b", seed=1102, generator={**GENERATOR, "commit": "0000abcdef"})
    config = build_config(tmp_path, ["run_a", "run_b"], val_runs=["run_b"])

    assert build_dataset(version="v002", config=config).manifest.totals["val_runs"] == 1


def test_build_on_s3(tmp_path):
    for offset, run_id in enumerate(["run_a", "run_b"]):
        make_run(tmp_path, run_id, seed=200 + offset)

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="synth-bucket")
        for path in sorted((tmp_path / "raw").rglob("*")):
            if path.is_file():
                key = f"raw/{path.relative_to(tmp_path / 'raw').as_posix()}"
                client.put_object(Bucket="synth-bucket", Key=key, Body=path.read_bytes())
        for path in sorted((tmp_path / "datasets").rglob("*")):
            if path.is_file():
                key = f"datasets/{path.relative_to(tmp_path / 'datasets').as_posix()}"
                client.put_object(Bucket="synth-bucket", Key=key, Body=path.read_bytes())

        config = build_config(
            tmp_path,
            ["run_a", "run_b"],
            val_runs=["run_b"],
            raw_root="s3://synth-bucket/raw",
            dataset_root="s3://synth-bucket/datasets",
        )
        result = build_dataset(version="v002", config=config)

        assert result.location == "s3://synth-bucket/datasets/v002"
        keys = {
            entry["Key"]
            for entry in client.list_objects_v2(Bucket="synth-bucket", Prefix="datasets/v002")[
                "Contents"
            ]
        }
        assert "datasets/v002/manifest.json" in keys
        assert "datasets/v002/yolo/dataset.yaml" in keys
        assert "datasets/v002/yolo/images/train/run_a_000000.png" in keys
        assert "datasets/v002/yolo/images/val/run_b_000000.png" in keys
        assert "datasets/v002/yolo/labels/val/run_b_000002.txt" in keys
        assert not any("images/train/run_b" in key for key in keys)

        # The copy is server-side, so the bytes must match the raw frame exactly.
        copied = client.get_object(
            Bucket="synth-bucket", Key="datasets/v002/yolo/images/val/run_b_000001.png"
        )["Body"].read()
        assert copied == b"run_b-image-1"


def test_local_and_s3_builds_agree(tmp_path):
    """Same inputs, same manifest apart from when it was built."""
    local_root = tmp_path / "local"
    local_root.mkdir()
    for offset, run_id in enumerate(["run_a", "run_b"]):
        make_run(local_root, run_id, seed=300 + offset)

    local = build_dataset(
        version="v002",
        config=build_config(local_root, ["run_a", "run_b"], val_runs=["run_b"]),
    ).manifest

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="synth-bucket")
        for top in ("raw", "datasets"):
            for path in sorted((local_root / top).rglob("*")):
                if path.is_file() and "v002" not in path.parts:
                    key = f"{top}/{path.relative_to(local_root / top).as_posix()}"
                    client.put_object(Bucket="synth-bucket", Key=key, Body=path.read_bytes())
        remote = build_dataset(
            version="v002",
            config=build_config(
                local_root,
                ["run_a", "run_b"],
                val_runs=["run_b"],
                raw_root="s3://synth-bucket/raw",
                dataset_root="s3://synth-bucket/datasets",
            ),
        ).manifest

    assert remote.to_dict() | {"built_at": local.built_at} == local.to_dict()
