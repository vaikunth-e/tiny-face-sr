# Third-party sources and artifacts

TinySR's migration code consolidates project implementations. Third-party dependencies are installed separately and retain their own licenses. A project-wide source license has not been selected by this migration; this document does not grant a new blanket license.

- **FFHQ**: public split IDs and derived baseline learning are associated with the FFHQ dataset. No dataset images are bundled. Consult the [FFHQ dataset repository](https://github.com/NVlabs/ffhq-dataset) and source-image terms for dataset use and redistribution.
- **OpenMV**: camera interfaces and the checked Haar binary layout target the compatibility baseline. Consult [OpenMV](https://github.com/openmv/openmv) for firmware, models and applicable licenses. No firmware or built-in FOMO weights are copied into this repository.
- **OpenCV**: the Haar converter accepts compatible externally supplied cascade XML. No OpenCV cascade XML or binary model data is bundled. Preserve upstream notices when distributing converted model data. See [OpenCV](https://github.com/opencv/opencv).
- **PyTorch, TensorFlow/LiteRT, NumPy, Pillow, scikit-image, pyserial and pytest**: runtime/test dependencies are installed separately under their respective licenses.
- **Real-ESRGAN/BasicSR**: reference models are planned, not bundled or represented as tested. Any later checkpoint/source acquisition must retain upstream provenance and terms from the official projects.

The baseline registry identifies both recovered artifact hashes and cleaned distribution hashes. Public models contain no camera photographs, source-image hashes or private run inventories.
