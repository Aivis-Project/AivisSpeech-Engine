# -*- mode: python ; coding: utf-8 -*-
# このファイルは元々 PyInstaller によって自動生成されたもので、それをカスタマイズして使用しています。
import sys
from pathlib import Path
from shutil import copy2, copytree

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = []
datas += collect_data_files('e2k')
datas += collect_data_files('pyopenjtalk')
datas += collect_data_files('style_bert_vits2')

# functorch のバイナリを収集
# ONNX に移行したため不要なはずだが、念のため
binaries = collect_dynamic_libs('functorch')

# Windows: Intel MKL 関連の DLL を収集
# これをやらないと PyTorch が CPU 版か CUDA 版かに関わらずクラッシュする…
# ONNX に移行したため不要なはずだが、念のため
if sys.platform == 'win32':
    lib_dir_path = Path(sys.prefix) / 'Library' / 'bin'
    if lib_dir_path.exists():
        mkl_dlls = list(lib_dir_path.glob('*.dll'))
        for dll in mkl_dlls:
            binaries.append((str(dll), '.'))

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    module_collection_mode={
        # Style-Bert-VITS2 内部で使われている TorchScript (@torch.jit) による問題を回避するために必要
        'style_bert_vits2': 'pyz+py',
    },
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='run',
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
    contents_directory='engine_internal',  # 実行時に sys._MEIPASS が参照するディレクトリ名
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='run',
)

# 実行ファイルのディレクトリに配置するファイルのコピー
target_dir = Path(DISTPATH) / 'run'

# リソースをコピー
manifest_file_path = Path('engine_manifest.json')
copy2(manifest_file_path, target_dir)
copytree('resources', target_dir / 'resources')

license_file_path = Path('licenses.json')
if license_file_path.is_file():
    copy2(license_file_path, target_dir)
