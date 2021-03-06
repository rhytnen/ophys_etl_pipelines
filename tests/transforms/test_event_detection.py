import pytest
import h5py
import contextlib
import numpy as np
from collections import namedtuple
try:
    import ophys_etl.transforms.event_detection as emod
except ModuleNotFoundError:
    # even though we might skip tests, pytest tries these imports
    from unittest.mock import Mock
    import sys
    sys.modules['FastLZeroSpikeInference'] = Mock()
    import ophys_etl.transforms.event_detection as emod
from ophys_etl.resources import event_decay_lookup_dict as decay_lookup


Events = namedtuple('Events', ['id', 'timestamps', 'magnitudes'])


def make_event(length, index, magnitude, decay_time, rate):
    timestamps = np.arange(length) / rate
    t0 = timestamps[index]
    z = np.zeros(length)
    z[index:] = magnitude * np.exp(-(timestamps[index:] - t0) / decay_time)
    return z


def sum_events(nframes, timestamps, magnitudes, decay_time, rate):
    data = np.zeros(nframes)
    for ts, mag in zip(timestamps, magnitudes):
        data += make_event(nframes, ts, mag, decay_time, rate)
    return data


@pytest.fixture(scope="function")
def dff_hdf5(tmp_path, request):
    sigma = request.param.get("sigma")
    offset = request.param.get("offset")
    decay_time = request.param.get("decay_time")
    nframes = request.param.get("nframes")
    events = request.param.get("events")
    rate = request.param.get("rate")
    noise_mult = request.param.get("noise_multiplier")

    rng = np.random.default_rng(42)
    data = rng.normal(loc=offset, scale=sigma, size=(len(events), nframes))
    for i, event in enumerate(events):
        data[i] += sum_events(nframes, event.timestamps, event.magnitudes,
                              decay_time, rate)

    names = [i.id for i in events]

    h5path = tmp_path / "fake_dff.h5"
    with h5py.File(h5path, "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("roi_names", data=names)

    with h5py.File(h5path, "r") as f:
        yield h5path, decay_time, rate, events, noise_mult


@pytest.mark.event_detect_only
@pytest.mark.parametrize(
        ("rate", "specify_multiplier", "multiplier", "expected"),
        [
            (11.0, True, 12, 12),
            (11.0, False, None, 2.6),
            (31.0, True, 2.4, 2.4),
            (31.0, False, None, 2.0),
            ])
def test_EventDetectionSchema_multiplier(tmp_path, rate, expected,
                                         specify_multiplier, multiplier):
    fpath = tmp_path / "junk_input.h5"
    with h5py.File(fpath, "w") as f:
        f.create_dataset("data", data=[1, 2, 3.14])
        f.create_dataset("roi_names", data=[5, 6, 7, 8])

    args = {
            'movie_frame_rate_hz': rate,
            'ophysdfftracefile': str(fpath),
            'valid_roi_ids': [4, 5, 6],
            'output_event_file': str(tmp_path / "junk_output.hdf5"),
            'decay_time': 1.234
            }
    if specify_multiplier:
        args['noise_multiplier'] = multiplier
    parser = emod.EventDetection(input_data=args, args=[])
    assert parser.args['noise_multiplier'] == expected


@pytest.mark.event_detect_only
@pytest.mark.parametrize(
        "missing_field, context",
        [
            (False, contextlib.nullcontext()),
            (True, pytest.raises(
                emod.EventDetectionException,
                match=r".*does not have the key 'roi_names'.*"))])
def test_EventDetectionSchema_missing_name(tmp_path, missing_field, context):
    fpath = tmp_path / "junk_input.h5"
    with h5py.File(fpath, "w") as f:
        f.create_dataset("data", data=[1, 2, 3.14])
        if not missing_field:
            f.create_dataset("roi_names", data=[5, 6, 7, 8])

    args = {
            'movie_frame_rate_hz': 31.0,
            'ophysdfftracefile': str(fpath),
            'valid_roi_ids': [4, 5, 6],
            'output_event_file': str(tmp_path / "junk_output.hdf5"),
            'decay_time': 1.234
            }
    with context:
        parser = emod.EventDetection(input_data=args, args=[])
        assert 'halflife' in parser.args


@pytest.mark.event_detect_only
def test_EventDetectionSchema_decay_time(tmp_path):
    fpath = tmp_path / "junk_input.h5"
    with h5py.File(fpath, "w") as f:
        f.create_dataset("data", data=[1, 2, 3.14])
        f.create_dataset("roi_names", data=[5, 6, 7, 8])

    # specify decay_time explicitly
    dtime = 1.234
    args = {
            'movie_frame_rate_hz': 31.0,
            'ophysdfftracefile': str(fpath),
            'valid_roi_ids': [4, 5, 6],
            'output_event_file': str(tmp_path / "junk_output.hdf5"),
            'decay_time': dtime
            }
    parser = emod.EventDetection(input_data=args, args=[])
    assert parser.args['decay_time'] == dtime

    # specifying valid genotype rather than decay time
    args.pop('decay_time')
    key = "Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai93(TITL-GCaMP6f)/wt"
    args['full_genotype'] = key
    parser = emod.EventDetection(input_data=args, args=[])
    assert 'decay_time' in parser.args
    assert parser.args['decay_time'] == decay_lookup[key]
    assert 'halflife' in parser.args

    # non-existent genotype exception
    args['full_genotype'] = 'non-existent-genotype'
    with pytest.raises(
            emod.EventDetectionException,
            match=r".*not available.*"):
        parser = emod.EventDetection(input_data=args, args=[])

    # neither arg supplied
    args.pop('full_genotype')
    with pytest.raises(
            emod.EventDetectionException,
            match=r"Must provide either.*"):
        parser = emod.EventDetection(input_data=args, args=[])


@pytest.mark.event_detect_only
@pytest.mark.parametrize(
        "dff_hdf5",
        [
            {
                "sigma": 1.0,
                "decay_time": 0.415,
                "offset": 0.0,
                "nframes": 1000,
                "rate": 31.0,
                "noise_multiplier": 1.0,
                "events": [
                    Events(
                        id=123,
                        timestamps=[145, 212, 280, 310, 430, 600, 890],
                        magnitudes=[4.0, 5.0, 6.0, 5.0, 5.5, 5.0, 7.0]),
                    Events(
                        id=124,
                        timestamps=[45, 112, 232, 410, 490, 650, 850],
                        magnitudes=[4.0, 5.0, 6.0, 5.0, 5.5, 5.0, 7.0])]
                    },
            {
                "sigma": 1.0,
                "decay_time": 0.415,
                "offset": 0.0,
                "nframes": 1000,
                "rate": 11.0,
                "noise_multiplier": 3.0,
                "events": []},
            {
                "sigma": 1.0,
                "decay_time": 0.415,
                "offset": 0.0,
                "nframes": 1000,
                "rate": 11.0,
                "noise_multiplier": 3.0,
                "events": [
                    Events(
                        id=123,
                        timestamps=[145, 212, 280, 310, 430, 600, 890],
                        magnitudes=[4.0, 5.0, 6.0, 5.0, 5.5, 5.0, 7.0]),
                    Events(
                        id=124,
                        timestamps=[45, 112, 232, 410, 490, 650, 850],
                        magnitudes=[4.0, 5.0, 6.0, 5.0, 5.5, 5.0, 7.0])]
                    },
            ], indirect=True)
def test_EventDetection(dff_hdf5, tmp_path):
    """This test runs the actual spike inference on fake data. The fake
    data is constructed by the dff_hdf5 fixture. This is an
    easy test, in that the SNR for the spikes is high, there is neither offset
    nor low frequency background, the fake spikes are not too close together,
    and, some false spikes are ignored at the end.
    """

    dff_path, decay_time, rate, expected_events, noise_mult = dff_hdf5

    args = {
            'movie_frame_rate_hz': rate,
            'ophysdfftracefile': str(dff_path),
            'valid_roi_ids': [123, 124],
            'output_event_file': str(tmp_path / "junk_output.h5"),
            'decay_time': decay_time,
            'noise_multiplier': noise_mult
            }
    ed = emod.EventDetection(input_data=args, args=[])
    ed.run()

    with h5py.File(args['output_event_file'], "r") as f:
        keys = list(f.keys())
        for k in ['events', 'roi_names', 'noise_stds', 'lambdas']:
            assert k in keys
        events = f['events'][()]
        if expected_events == []:
            assert "warning" in keys
            assert "No valid ROIs in" in str(f['warning'][()])

    for result, expected in zip(events, expected_events):
        nresult = np.count_nonzero(result)
        result_index = np.argwhere(result != 0).flatten()
        # check that the number of events match the expectation:
        print(result_index)
        print(expected.timestamps)
        print(ed.args['noise_multiplier'])
        assert nresult == len(expected.timestamps)

        # check that they are in the right place:
        np.testing.assert_array_equal(result_index, expected.timestamps)


@pytest.mark.event_detect_only
@pytest.mark.parametrize("rate", [11.0, 31.0])
def test_fast_lzero(rate):
    decay_time = 0.4
    nframes = 1000
    timestamps = [45, 112, 232, 410, 490, 700, 850]
    magnitudes = [4.0, 5.0, 6.0, 5.0, 5.5, 5.0, 7.0]
    data = sum_events(nframes, timestamps, magnitudes, decay_time, rate)

    halflife = emod.calculate_halflife(decay_time)
    gamma = emod.calculate_gamma(halflife, rate)
    f = emod.fast_lzero(1.0, data, gamma, True)

    assert f.shape == data.shape


@pytest.mark.event_detect_only
@pytest.mark.parametrize(
        "frac_outliers, threshold",
        [
            (0.025, 0.1),
            (0.05, 0.1),
            ])
def test_trace_noise_estimate(frac_outliers, threshold):
    """makes a low-frequency signal with noise and outliers and
    checks that the trace noise estimate meets some threshold
    """
    filt_length = 31
    npts = 10000
    sigma = 1.0
    rng = np.random.default_rng(42)
    x = 0.2 * np.cos(2.0 * np.pi * np.arange(npts) / (filt_length * 10))
    x += rng.standard_normal(npts) * sigma
    inds = rng.integers(0, npts, size=int(frac_outliers * npts))
    x[inds] *= 100
    rstd = emod.trace_noise_estimate(x, filt_length)
    assert np.abs(rstd - sigma) < threshold
