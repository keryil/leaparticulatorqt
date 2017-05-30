# -*- mode: python -*-

block_cipher = None


a = Analysis(['leaparticulator/trajectory_plotter/plotter_p2p.py'],
             pathex=['/home/parallels/leaparticulatorqt'],
             binaries=[],
             datas=[('qt_generated/', 'qt_generated/')],
             hiddenimports=['leaparticulator.p2p.server'],
             hookspath=[],
             runtime_hooks=['rthook_pyqt4.py'],
             excludes=['tcl', 'tk', 'Tkinter', 'tkinter'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='plotter_p2p',
          debug=False,
          strip=False,
          upx=True,
          console=True )
