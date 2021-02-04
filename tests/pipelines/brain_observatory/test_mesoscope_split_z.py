import pytest
import tifffile
import numpy as np
import h5py
import os
import copy
from ophys_etl.transforms.mesoscope_2p import MesoscopeTiff
from ophys_etl.pipelines.brain_observatory.scripts.run_mesoscope_splitting import split_z


class MesoscopeTiffDummy(MesoscopeTiff):
    """
    A class to allow us to pass in fake metadata by hand
    """

    def __init__(self, source_tiff, cache=False):
        self._n_pages = None
        self._planes = None
        self._volumes = None
        self._source = source_tiff
        self._tiff = tifffile.TiffFile(self._source)
        self._tiff.pages.cache = cache


def generate_fake_z(tmp_filename, frame_zs, n_repeats):
    """
    Writes a fake timeseries TIFF to disk.
    The time series TIFF will contain 100 512x512 TIFFs per ROI.
    Values in the TIFF will be z*10
    Returns frame and ROI metadata needed by MesoscopeTiffDummy.

    Parameters
    -----------
    tmp_filename -- path to file where we should writ the TIFF

    frame_zs -- z values of scans as they should appear
                in SI.hStackManager.zs field, i.e. [[z1, z2], [z3, z4],...]

    n_repeates -- int; how many times to re-scan each z

    Returns
    -------
    frame_metadata -- dict to be assigned to MesoscopeTiffDummy._frame_data

    roi_metadata -- dict to be assigned to MesosocopeTiffDummy._roi_data
    """

    flattened_z = []
    for frame_z_list in frame_zs:
        for zz in frame_z_list:
            flattened_z.append(zz)

    n_z = len(flattened_z)

    # generate fake tiff data and write it to tmp_filename
    tiff_data = np.zeros((n_z*n_repeats, 512, 512), dtype=int)
    for jj in range(n_repeats):
        for ii in range(n_z):
            tiff_data[jj*n_z+ii, :, :] = int(10*flattened_z[ii]*n_repeats + jj)

    tifffile.imwrite(tmp_filename, tiff_data, bigtiff=True)
    del tiff_data

    frame_metadata = {}
    frame_metadata['SI'] = {}
    frame_metadata['SI']['hStackManager'] = {}
    frame_metadata['SI']['hStackManager']['zs'] = copy.deepcopy(frame_zs)
    frame_metadata['SI']['hFastZ'] = {}
    frame_metadata['SI']['hFastZ']['userZs'] = copy.deepcopy(frame_zs)

    frame_metadata['SI']['hChannels'] = {}
    frame_metadata['SI']['hChannels']['channelsActive'] = [[1], [2]]


    # for split z, all that actually matters is whether or not
    # discretePlaneMode is False
    # that is the plane that will be caught with volume_scanned
    _rois = []

    for ii in range(4):
        _rois.append({'zs': -1,
                      'discretePlaneMode': (ii>1),
                      'scanfields': [{'pixelResolutionXY': (512, 512)}]})

    roi_metadata = {}
    roi_metadata['RoiGroups'] = {}
    roi_metadata['RoiGroups']['imagingRoiGroup'] = {}
    roi_metadata['RoiGroups']['imagingRoiGroup']['rois'] = _rois

    return frame_metadata, roi_metadata


def generate_experiments(flattened_z, storage_dir):
    """
    Parameters
    -----------
    flattened_z -- a list of z values for the scan plane

    storage_dir -- path to the parent of each experiment's storage_dir

    Returns
    -------
    A list of experiments suitable for passing into split_timeseries

    """
    # generate some fake experiments based on the data we put in
    # the TIFF file
    experiments = []
    for exp_id, zz in enumerate(flattened_z):

        exp = {}
        local_storage = os.path.join(storage_dir, 'exp_%d' % exp_id)
        if not os.path.exists(local_storage):
            os.makedirs(local_storage)
        exp['storage_directory'] = local_storage
        exp['experiment_id'] = exp_id
        exp['scanfield_z'] = zz
        exp['roi_index'] = (exp_id>1)

        # I do not think that this metadata gets used to split the file;
        # it is just passed through as metadata
        exp['resolution'] = 0
        exp['offset_x'] = 0
        exp['offset_y' ] = 0
        exp['rotation'] = 0
        exp['height'] = 0
        exp['width'] = 0
        experiments.append(exp)

    return experiments


def validate_split_z(experiment_list, storage_dir, frame_zs, n_repeats):
    """
    Actually validate that outputs were correctly written.

    Parameters
    -----------
    experiment_list -- a list of experiments whose h5 files we
                       are validating

    storage_dir -- path to the parent of each experiment's storage_dir

    frame_zs -- the list of lists containing the SI.hStackManager.zs
                metadata from the big tiff file
    """

    # make sure the values in the HDF5 files are what we expect
    for experiment in experiment_list:
        exp_id = experiment['experiment_id']
        zz = experiment['scanfield_z']

        valid_zs = []
        for z_pair in frame_zs:
            if abs(z_pair[0]-zz) < abs(z_pair[1]-zz):
                valid_zs.append(z_pair[0])
            else:
                valid_zs.append(z_pair[1])

        n_z = len(valid_zs)
        assert n_z == len(frame_zs)

        dirname = os.path.join(storage_dir, 'exp_%d' % exp_id)
        fname = os.path.join(dirname, '%d_z_stack_local.h5' % exp_id)
        assert os.path.isfile(fname)
        with h5py.File(fname, 'r') as in_file:
            data = in_file['data'][()]
            print('data ',data.shape)
            assert data.shape == (n_z*n_repeats, 512, 512)
            for jj in range(n_repeats):
                for ii in range(n_z):
                    vz = valid_zs[ii]
                    val = int(10*vz)*n_repeats + jj
                    frame = data[jj*n_z+ii, :, :]
                    assert (frame==val).all()


# flattened_z_expected should include all of the z values from
# frame_zs in the order that they would occur in frame_zs.flatten(),
# excluding any values that do not occur in roi_zs
"""
@pytest.mark.parametrize("frame_zs,roi_zs,flattened_z_expected",
                         [([[22, 33], [44, 55], [66, 77], [88, 99]],
                           [[22, 44, 66, 88], [33, 55, 77, 99]],
                           [22, 33, 44, 55, 66, 77, 88, 99]),
                          ([[22, 33], [44, 55], [66, 77], [88, 99]],
                           [[22, 33, 44, 55], [66, 77, 88, 99]],
                           [22, 33, 44, 55, 66, 77, 88, 99]),
                          ([[22, 44], [66, 88], [33, 55], [77, 99]],
                           [[22, 33, 44, 55], [66, 77, 88, 99]],
                           [22, 44, 66, 88, 33, 55, 77, 99]),
                          ([[22, 0], [44, 0]],
                           [[22],[44]],
                           [22, 44]),
                           ([[44, 0], [22, 0]],
                           [[22],[44]],
                           [44, 22]),
                          ([[22, 9], [44, 6]],
                           [[22],[44]],
                           [22, 44]),
                          ([[44, 1], [22, 4]],
                           [[22],[44]],
                           [44, 22])])
"""
@pytest.mark.parametrize("frame_zs,flattened_z,n_repeats",
                         [([[1.1, 5.1], [1.2, 5.2], [1.3, 5.3]], [1.2, 5.2], 15)
                         ])
def test_split_z(tmpdir, frame_zs, flattened_z, n_repeats):
    storage_dir = os.path.join(tmpdir, 'zstack_storage')
    tiff_fname = os.path.join(tmpdir, 'zstack.tiff')

    # generate mock metadata to be passed directly to
    # MesoscopeTiffDummy

    (frame_metadata,
     roi_metadata) = generate_fake_z(tiff_fname,
                                    frame_zs,
                                    n_repeats)


    # actually read our test data from tmp

    mtiff = MesoscopeTiffDummy(tiff_fname, cache=True)
    mtiff._frame_data = frame_metadata
    mtiff._roi_data = roi_metadata

    experiment_list = generate_experiments(flattened_z, storage_dir)

    for experiment in experiment_list:
        output = split_z(mtiff, experiment, testing=True)

    validate_split_z(experiment_list, storage_dir, frame_zs, n_repeats)
