"""
tests/test_io.py
=================
Testes para o módulo de utilitários de IO
(ml_clustering_lab.utils.io).

Cobertura
---------
- ensure_dir: criação de diretório e idempotência
- save_json / load_json: serialização e deserialização
- save_dataframe: salvar CSV
- save_figure: salvar PNG
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ml_clustering_lab.utils.io import (
    ensure_dir,
    load_json,
    save_dataframe,
    save_figure,
    save_json,
)


class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        new_dir = tmp_path / "new" / "nested"
        result = ensure_dir(new_dir)
        assert result.exists()
        assert result.is_dir()

    def test_returns_path_object(self, tmp_path):
        p = ensure_dir(tmp_path / "sub")
        assert isinstance(p, Path)

    def test_idempotent_on_existing_dir(self, tmp_path):
        ensure_dir(tmp_path)
        ensure_dir(tmp_path)  # should not raise
        assert tmp_path.exists()

    def test_accepts_string_path(self, tmp_path):
        new_dir = str(tmp_path / "strdir")
        result = ensure_dir(new_dir)
        assert result.exists()


class TestSaveAndLoadJson:
    def test_save_and_reload(self, tmp_path):
        data = {"a": 1, "b": [1, 2, 3], "c": "hello"}
        path = tmp_path / "test.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_saves_file(self, tmp_path):
        path = tmp_path / "out.json"
        save_json({"x": 42}, path)
        assert path.exists()

    def test_readable_by_stdlib(self, tmp_path):
        path = tmp_path / "out.json"
        save_json({"key": "value"}, path)
        with open(path) as f:
            result = json.load(f)
        assert result["key"] == "value"

    def test_creates_parent_dir(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "out.json"
        save_json({"k": 1}, path)
        assert path.exists()

    def test_load_json_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_json("/tmp/does_not_exist_xyz.json")

    def test_float_nan_roundtrip(self, tmp_path):
        import math

        path = tmp_path / "nan.json"
        save_json({"v": float("nan")}, path)
        loaded = load_json(path)
        # Python's json encodes NaN as a bare NaN literal; when loaded back it's float nan
        # or encoded as string "nan" — either way the value represents not-a-number
        assert math.isnan(loaded["v"]) or loaded["v"] == "nan"


class TestSaveDataframe:
    def test_saves_csv_file(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "out.csv"
        save_dataframe(df, path)
        assert path.exists()

    def test_saved_content_matches(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "out.csv"
        save_dataframe(df, path)
        loaded = pd.read_csv(path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_creates_parent_dir(self, tmp_path):
        df = pd.DataFrame({"x": [1]})
        path = tmp_path / "sub" / "data.csv"
        save_dataframe(df, path)
        assert path.exists()


class TestSaveFigure:
    def test_saves_png_file(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        path = tmp_path / "fig.png"
        save_figure(fig, path)
        assert path.exists()
        plt.close(fig)

    def test_creates_parent_dir(self, tmp_path):
        fig, ax = plt.subplots()
        path = tmp_path / "sub" / "fig.png"
        save_figure(fig, path)
        assert path.exists()
        plt.close(fig)

    def test_file_size_nonzero(self, tmp_path):
        fig, ax = plt.subplots()
        ax.scatter([1, 2], [3, 4])
        path = tmp_path / "scatter.png"
        save_figure(fig, path)
        assert path.stat().st_size > 0
        plt.close(fig)
