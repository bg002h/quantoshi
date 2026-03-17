# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for btc_projections
# Build with:  pyinstaller btc_projections.spec

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['btc_projections.py'],
    pathex=[str(Path('btc_projections.py').parent.resolve())],
    binaries=[],
    datas=[
        # Bundle precomputed model data and price CSV
        ('model_data.pkl',              '.'),
        ('../BitcoinPricesDaily.csv',   '.'),
    ],
    hiddenimports=[
        # statsmodels sub-modules not always auto-detected
        'statsmodels.regression.quantile_regression',
        'statsmodels.tools',
        'statsmodels.base.model',
        'statsmodels.base.wrapper',
        # scipy
        'scipy.stats',
        'scipy.optimize',
        # matplotlib backends
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.backend_agg',
        # PyQt5 modules
        'PyQt5.QtWidgets',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.sip',
        # pandas
        'pandas._libs.tslibs.np_datetime',
        'pandas._libs.tslibs.nattype',
        'pandas._libs.tslibs.timestamps',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Not needed — saves space; unittest/test must stay (scipy dep)
        'tkinter', 'wx', 'gtk', 'PySide2', 'PySide6', 'PyQt6',
        'IPython', 'ipywidgets', 'notebook', 'jupyter',
        'cv2',
        'sklearn', 'torch', 'tensorflow',
        'docutils', 'sphinx',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='btc_projections',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,        # no terminal window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='btc_projections',
)
