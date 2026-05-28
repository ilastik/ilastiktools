import ilastik_carving_tools
import numpy as np
import numpy.typing as npt
import pytest


@pytest.fixture
def gridseg() -> ilastik_carving_tools.GridSegmentor_3D_UInt32:
    return ilastik_carving_tools.GridSegmentor_3D_UInt32()


@pytest.fixture
def labels() -> npt.NDArray[np.uint32]:
    labels = np.arange(27).reshape((3, 3, 3)).astype(np.uint32)
    labels //= 10
    labels += 1
    return labels


@pytest.fixture
def features() -> npt.NDArray[np.uint32]:
    feats = np.random.random((3, 3, 3)).astype(np.float32)
    return feats


@pytest.fixture
def brush_stroke() -> npt.NDArray[np.uint8]:
    brush_stroke = np.zeros([3, 3, 3], dtype=np.uint8)
    brush_stroke[1, 1, 1] = 1
    brush_stroke[:, :, 2] = 0
    brush_stroke[:, :, 0] = 0
    brush_stroke[0, :, 0] = 0
    return brush_stroke


def test_minimal(
    gridseg: ilastik_carving_tools.GridSegmentor_3D_UInt32,
    labels: npt.NDArray[np.uint32],
    features: npt.NDArray[np.uint32],
    brush_stroke: npt.NDArray[np.uint8],
):
    gridseg.preprocessing(labels, features)
    expected_graph = np.array([3, 2, 3, 1, 1, 2, 2, 3, 1, 1, 0, 2, 2, 2, 0, 1, 1, 3, 3, 1, 1, 2])
    tmpout = gridseg.serializeGraph()
    np.testing.assert_array_equal(tmpout, expected_graph)

    tmp = np.zeros(tmpout.shape, dtype=np.uint32)
    tmpout2 = gridseg.serializeGraph(tmp)
    np.testing.assert_array_equal(tmpout2, tmp)

    gridseg.addSeeds(brushStroke=brush_stroke, roiBegin=[0, 0, 0], roiEnd=[3, 3, 3], maxValidLabel=2)

    gridseg.run(0.5, 0.01)
    seg = gridseg.getSegmentation([0, 0, 0], [3, 3, 1])
    expected_seg_shape = (3, 3, 1)
    assert seg.shape == expected_seg_shape
    np.testing.assert_array_equal(seg, np.ones_like(seg))
    supervoxels = gridseg.getSuperVoxelSeg()
    expected_supervoxels = np.array([0, 1, 1, 1])
    np.testing.assert_array_equal(supervoxels, expected_supervoxels)

    sv_seeds = gridseg.getSuperVoxelSeeds()
    expected_sv_seeds = np.array([0, 0, 1, 0])
    np.testing.assert_array_equal(sv_seeds, expected_sv_seeds)
