import argschema
import tempfile
import tifffile
import json
import marshmallow
import h5py
import os
import numpy as np
from pathlib import Path
from ophys_etl.schemas.fields import H5InputFile
from ophys_etl.transforms.suite2p_wrapper import (Suite2PWrapper,
                                                  Suite2PWrapperSchema)


class Suite2PRegistrationInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.Str(default="INFO")
    suite2p_args = argschema.fields.Nested(Suite2PWrapperSchema,
                                           required=True)
    motion_corrected_output = argschema.fields.OutputFile(
        required=True,
        description="destination path for hdf5 motion corrected video.")
    motion_diagnostics_output = argschema.fields.OutputFile(
        required=True,
        description=("hdf5 format of diagnostics harvested from Suite2P "
                     "ops.npy"))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tmpdir = None

    @marshmallow.pre_load
    def setup_default_suite2p_args(self, data: dict, **kwargs) -> dict:
        data["suite2p_args"]["log_level"] = data["log_level"]
        data['suite2p_args']['roidetect'] = False
        data['suite2p_args']['do_registration'] = 1
        data['suite2p_args']['reg_tif'] = True
        data['suite2p_args']['retain_files'] = ["*.tif", "ops.npy"]
        if "output_dir" not in data["suite2p_args"]:
            # send suite2p results to a temporary directory
            # the results of this pipeline will be formatted versions anyway
            self.tmpdir = tempfile.TemporaryDirectory()
            data["suite2p_args"]["output_dir"] = self.tmpdir.name
        if "output_json" not in data["suite2p_args"]:
            Suite2p_output = (Path(data["suite2p_args"]["output_dir"])
                              / "Suite2P_output.json")
            data["suite2p_args"]["output_json"] = str(Suite2p_output)
        # we are not doing registration here, but the wrapper schema wants
        # a value:
        data['suite2p_args']['nbinned'] = 1000
        return data


class Suite2PRegistrationOutputSchema(argschema.ArgSchema):
    motion_corrected_output = H5InputFile(
        required=True,
        description="destination path for hdf5 motion corrected video.")
    motion_diagnostics_output = H5InputFile(
        required=True,
        description=("hdf5 format of diagnostics harvested from Suite2P "
                     "ops.npy"))


class Suite2PRegistration(argschema.ArgSchemaParser):
    default_schema = Suite2PRegistrationInputSchema
    default_output_schema = Suite2PRegistrationOutputSchema

    def run(self):
        self.logger.name = type(self).__name__
        self.logger.setLevel(self.args.pop('log_level'))
        ophys_etl_commit_sha = os.environ.get("OPHYS_ETL_COMMIT_SHA",
                                              "local build")
        self.logger.info(f"OPHYS_ETL_COMMIT_SHA: {ophys_etl_commit_sha}")

        # register with Suite2P
        suite2p_args = self.args['suite2p_args']
        register = Suite2PWrapper(input_data=suite2p_args, args=[])
        register.run()

        # get paths to Suite2P outputs
        with open(suite2p_args["output_json"], "r") as f:
            outj = json.load(f)
        tif_paths = [Path(i) for i in outj['output_files']["*.tif"]]
        ops_path = Path(outj['output_files']['ops.npy'][0])

        # assemble tifs into an hdf5
        data = []
        for fname in tif_paths:
            with tifffile.TiffFile(fname) as f:
                nframes = len(f.pages)
                for i, page in enumerate(f.pages):
                    arr = page.asarray()
                    if i == 0:
                        data.append(
                                np.zeros((nframes, *arr.shape), dtype='int16'))
                    data[-1][i] = arr
        data = np.concatenate(data, axis=0)
        data[data < 0] = 0
        data = np.uint16(data)

        # write the hdf5
        with h5py.File(self.args['motion_corrected_output'], "w") as f:
            f.create_dataset("data", data=data)
        self.logger.info("concatenated Suite2P tiff output to "
                         f"{self.args['motion_corrected_output']}")

        ops_keys = ["Lx", "Ly", "nframes", "xrange", "yrange", "xoff", "yoff",
                    "corrXY", "meanImg"]
        ops = np.load(ops_path, allow_pickle=True)
        with h5py.File(self.args['motion_diagnostics_output'], "w") as f:
            for key in ops_keys:
                f.create_dataset(key, data=ops.item()[key])
        self.logger.info("transferred registration metrics from `ops.npy` to "
                         f"{self.args['motion_diagnostics_output']}. "
                         f"Metrics are {ops_keys}.")

        # Clean up temporary directories and/or files created during
        # Schema invocation
        if self.schema.tmpdir is not None:
            self.schema.tmpdir.cleanup()

        outj = {k: self.args[k]
                for k in ['motion_corrected_output',
                          'motion_diagnostics_output']}
        self.output(outj, indent=2)


if __name__ == "__main__":  # pragma: nocover
    s2preg = Suite2PRegistration()
    s2preg.run()