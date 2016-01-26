from distutils.core import setup

setup(
    name='leaparticulatorqt',
    version='0.1',
    packages=['qt_generated', 'leaparticulator', 'leaparticulator.p2p', 'leaparticulator.p2p.ui',
              'leaparticulator.data', 'leaparticulator.test', 'leaparticulator.test.p2p', 'leaparticulator.test.data',
              'leaparticulator.test.analysis', 'leaparticulator.test.benchmark', 'leaparticulator.test.benchmark.data',
              'leaparticulator.browser', 'leaparticulator.drivers', 'leaparticulator.drivers.osx',
              'leaparticulator.drivers.linux', 'leaparticulator.theremin', 'leaparticulator.notebooks',
              'leaparticulator.notebooks.sandbox', 'leaparticulator.trajectory_recorder'],
    url='',
    license='GPL',
    author='Kerem Eryilmaz',
    author_email='keryilmaz@gmail.com',
    description='', requires=['pyaudio', 'jsonpickle', 'pandas',
                              'numpy', 'ghmm', 'PyQt4', 'scipy', 'sip']
)
