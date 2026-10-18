import csv

from trainsv_gui import browse, check_dataset, folder_info


def test_folder_info_counts_lazily_across_supported_layouts(tmp_path):
    (tmp_path / "root.wav").touch()
    (tmp_path / "root.lab").touch()
    (tmp_path / "wav").mkdir()
    (tmp_path / "wav" / "a.wav").touch()
    (tmp_path / "lab").mkdir()
    (tmp_path / "lab" / "a.lab").touch()
    (tmp_path / "wavs").mkdir()
    (tmp_path / "wavs" / "b.wav").touch()
    (tmp_path / "labs").mkdir()
    (tmp_path / "labs" / "b.lab").touch()

    result = folder_info(str(tmp_path))

    assert result == {"exists": True, "wav": 3, "lab": 3}


def test_check_dataset_counts_missing_recordings_from_one_wav_scan(tmp_path):
    wavs = tmp_path / "wavs"
    wavs.mkdir()
    (wavs / "present.wav").touch()
    (wavs / "also_present.wav").touch()
    with (tmp_path / "transcriptions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "note_seq", "ph_dur", "style"])
        writer.writeheader()
        writer.writerow({"name": "present", "note_seq": "C4", "ph_dur": "0.2 0.1", "style": ""})
        writer.writerow({"name": "also_present", "note_seq": "", "ph_dur": "0.3", "style": "Solid"})
        writer.writerow({"name": "missing", "note_seq": "", "ph_dur": "0.4", "style": ""})

    result = check_dataset(str(tmp_path))

    assert result["exists"] is True
    assert result["has_csv"] is True
    assert result["has_wavs"] is True
    assert result["rows"] == 3
    assert result["notes"] == 1
    assert result["missing"] == 1
    assert result["styles"] == ["Solid"]
    assert abs(result["seconds"] - 1.0) < 1e-9


def test_browse_preserves_directory_first_sort_and_file_filters(tmp_path):
    (tmp_path / "z_folder").mkdir()
    (tmp_path / "a_folder").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "z.wav").touch()
    (tmp_path / "a.wav").touch()
    (tmp_path / "ignored.txt").touch()

    result = browse(str(tmp_path), want_files=True, exts={"wav"})

    assert [entry["name"] for entry in result["entries"]] == [
        "a_folder", "z_folder", "a.wav", "z.wav"
    ]
    assert [entry["dir"] for entry in result["entries"]] == [True, True, False, False]
