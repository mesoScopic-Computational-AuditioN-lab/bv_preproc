# bv_preproc
Collection of high- (and some low-) level functions to preprocess (DICOM) files using BrainVoyager and outside functions. Functions are all bundeled within .py files.

Package is a work in progress, feel free to change scrips and pull request changes.

---

A python implimentation of several preprocessing steps designed to work within BrainVoyager notebooks.
For any support please mail Jorie van Haren <jjg.vanharen@maastrichtuniversity.nl>.
A Demo notebook of one possible preprocessing pipeline is provided to give intuition of how to use the package.

---

## Installation

1. Download the bv_preproc folder to a convinient location
2. Add parent directory of the bv_preproc folder to path using `sys.path.append('folder_where_bv_preproc_is_located')`
3. import bv_preproc

Some optional functions are not loaded directly to enhance system compatibility (most noticable bv_preproc.anatomical_nighres) - These modules should be loaded using e.g. `import bv_preproc.anatomical_nighres`.

---

TODO: Finish implementation of NORDIC (using scrips by Mahdi Enan)
