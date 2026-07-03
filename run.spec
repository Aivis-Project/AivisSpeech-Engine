# -*- mode: python ; coding: utf-8 -*-
# このファイルは元々 PyInstaller によって自動生成されたもので、それをカスタマイズして使用しています。
import os
import subprocess
import sys
import importlib.util
from pathlib import Path
from shutil import copy2, copytree, ignore_patterns, which

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

datas = []
datas += collect_data_files("e2k")
datas += collect_data_files("pyopenjtalk")
datas += collect_data_files("style_bert_vits2")
datas += collect_data_files("onnxruntime_ep_style_bert_vits2_ggml")

hiddenimports = collect_submodules("onnxruntime_ep_style_bert_vits2_ggml")

# functorch のバイナリを収集
# ONNX に移行したため不要なはずだが、念のため
binaries = collect_dynamic_libs("functorch")

# Windows: Intel MKL 関連の DLL を収集
# これをやらないと PyTorch が CPU 版か CUDA 版かに関わらずクラッシュする…
# ONNX に移行したため不要なはずだが、念のため
if sys.platform == "win32":
    lib_dir_path = Path(sys.prefix) / "Library" / "bin"
    if lib_dir_path.exists():
        mkl_dlls = list(lib_dir_path.glob("*.dll"))
        for dll in mkl_dlls:
            binaries.append((str(dll), "."))

a = Analysis(
    ["run.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    module_collection_mode={
        # Style-Bert-VITS2 内部で使われている TorchScript (@torch.jit) による問題を回避するために必要
        "style_bert_vits2": "pyz+py",
    },
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="run",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory="engine_internal",  # 実行時に sys._MEIPASS が参照するディレクトリ名
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="run",
)


def _env_paths(name):
    value = os.environ.get(name, "")
    return [Path(item) for item in value.split(os.pathsep) if item]


def _package_dir(package_name):
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.submodule_search_locations is None:
        return None
    locations = list(spec.submodule_search_locations)
    if not locations:
        return None
    return Path(locations[0]).resolve()


def _copy_existing_files(candidates, dest_dir):
    copied = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    seen_basenames = set()
    for candidate in candidates:
        if candidate is None or not candidate.exists() or not candidate.is_file():
            continue
        if candidate.name in seen_basenames:
            continue
        seen_basenames.add(candidate.name)
        dest = dest_dir / candidate.name
        if candidate.resolve() == dest.resolve():
            copied.append(dest)
            continue
        copy2(candidate, dest)
        copied.append(dest)
    return copied


def _copy_matching_files(search_dirs, patterns, dest_dir):
    candidates = []
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in patterns:
            candidates.extend(sorted(search_dir.rglob(pattern)))
    return _copy_existing_files(candidates, dest_dir)


def _patch_linux_rpath(libraries):
    if not sys.platform.startswith("linux"):
        return
    patchelf = which("patchelf")
    if patchelf is None:
        print(
            "WARNING: patchelf is not available; ONNX GGML sidecars keep their original rpath."
        )
        return
    for library in libraries:
        if ".so" not in library.name:
            continue
        subprocess.run(
            [patchelf, "--set-rpath", "$ORIGIN", str(library)],
            check=True,
        )


def _macos_rpaths(library):
    result = subprocess.run(
        ["otool", "-l", str(library)],
        check=True,
        capture_output=True,
        text=True,
    )
    rpaths = []
    is_rpath_command = False
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped == "cmd LC_RPATH":
            is_rpath_command = True
            continue
        if is_rpath_command and stripped.startswith("path "):
            rpaths.append(stripped.removeprefix("path ").split(" (offset", 1)[0])
            is_rpath_command = False
    return rpaths


def _patch_macos_rpath(libraries):
    if sys.platform != "darwin":
        return
    install_name_tool = which("install_name_tool")
    if install_name_tool is None:
        print(
            "WARNING: install_name_tool is not available; ONNX GGML sidecars keep their original rpath."
        )
        return

    for library in libraries:
        if library.suffix != ".dylib":
            continue
        rpaths = _macos_rpaths(library)
        if "@loader_path" not in rpaths:
            subprocess.run(
                [install_name_tool, "-add_rpath", "@loader_path", str(library)],
                check=True,
            )
        for rpath in rpaths:
            if rpath == "@loader_path":
                continue
            subprocess.run(
                [install_name_tool, "-delete_rpath", rpath, str(library)],
                check=True,
            )


def _copy_onnx_ggml_runtime(target_dir):
    required = os.environ.get("STYLE_BERT_VITS2_GGML_REQUIRED") == "1"
    bundle_dirs = _env_paths("STYLE_BERT_VITS2_GGML_BUNDLE_DIR")

    package_src = _package_dir("onnxruntime_ep_style_bert_vits2_ggml")
    package_dest = target_dir / "onnxruntime_ep_style_bert_vits2_ggml"
    if package_src is not None and package_src.exists():
        copytree(
            package_src,
            package_dest,
            dirs_exist_ok=True,
            ignore=ignore_patterns("__pycache__", "*.pyc"),
        )

    ep_names = (
        "libstyle_bert_vits2_ggml_onnx_ep.so",
        "libstyle_bert_vits2_ggml_onnx_ep.dylib",
        "style_bert_vits2_ggml_onnx_ep.dll",
    )
    ep_candidates = _env_paths("STYLE_BERT_VITS2_GGML_EP_LIBRARY_PATH")
    ep_candidates += [
        bundle_dir / "onnxruntime_ep_style_bert_vits2_ggml" / "lib" / name
        for bundle_dir in bundle_dirs
        for name in ep_names
    ]
    ep_candidates += [package_dest / "lib" / name for name in ep_names]
    if package_src is not None:
        ep_candidates += [package_src / "lib" / name for name in ep_names]
    copied_ep = _copy_existing_files(
        ep_candidates,
        target_dir / "onnxruntime_ep_style_bert_vits2_ggml" / "lib",
    )

    if sys.platform == "win32":
        tts_patterns = ("tts.dll", "libtts.dll")
        dependency_patterns = ("ggml*.dll", "libggml*.dll")
    elif sys.platform == "darwin":
        tts_patterns = ("libtts.dylib",)
        dependency_patterns = ("libggml*.dylib",)
    else:
        tts_patterns = ("libtts.so", "libtts.so.*")
        dependency_patterns = ("libggml*.so", "libggml*.so.*")

    tts_candidates = _env_paths("STYLE_BERT_VITS2_TTS_CPP_LIBRARY_PATH")
    tts_library_dirs = _env_paths("STYLE_BERT_VITS2_TTS_CPP_LIBRARY_DIRS")
    tts_library_dirs += [bundle_dir / "lib" for bundle_dir in bundle_dirs]
    copied_tts = _copy_existing_files(tts_candidates, target_dir / "lib")
    copied_tts += _copy_matching_files(
        tts_library_dirs, tts_patterns, target_dir / "lib"
    )
    copied_deps = _copy_matching_files(
        tts_library_dirs,
        dependency_patterns,
        target_dir / "lib",
    )
    _patch_linux_rpath(copied_tts + copied_deps)
    _patch_macos_rpath(copied_ep + copied_tts + copied_deps)

    if required:
        if not copied_ep:
            raise RuntimeError(
                "STYLE_BERT_VITS2_GGML_REQUIRED=1 but native Plugin EP was not packaged."
            )
        if not copied_tts:
            raise RuntimeError(
                "STYLE_BERT_VITS2_GGML_REQUIRED=1 but TTS.cpp runtime was not packaged."
            )

    print(f"Packaged ONNX GGML EP sidecars: {[str(path) for path in copied_ep]}")
    print(
        f"Packaged TTS.cpp sidecars: {[str(path) for path in copied_tts + copied_deps]}"
    )


# 実行ファイルのディレクトリに配置するファイルのコピー
target_dir = Path(DISTPATH) / "run"

# リソースをコピー
manifest_file_path = Path("engine_manifest.json")
copy2(manifest_file_path, target_dir)
copytree("resources", target_dir / "resources")

license_file_path = Path("licenses.json")
if license_file_path.is_file():
    copy2(license_file_path, target_dir)

_copy_onnx_ggml_runtime(target_dir)
