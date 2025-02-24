import os
import numpy as np
import json
import zarr
import s3fs
from xradio._utils.zarr.common import _get_file_system_and_items

from numcodecs.compat import (
    ensure_text,
    ensure_ndarray_like,
    ensure_bytes,
    ensure_contiguous_ndarray_like,
)

full_dims_lm = ["time", "frequency", "polarization", "l", "m"]
full_dims_uv = ["time", "frequency", "polarization", "l", "m"]
norm_dims = ["frequency", "polarization"]

image_data_variables_and_dims_double_precision = {
    "aperture": {"dims": full_dims_uv, "dtype": "<c16", "name": "APERTURE"},
    "aperture_normalization": {
        "dims": norm_dims,
        "dtype": "<c16",
        "name": "APERTURE_NORMALIZATION",
    },
    "primary_beam": {"dims": full_dims_lm, "dtype": "<f8", "name": "PRIMARY_BEAM"},
    "uv_sampling": {"dims": full_dims_uv, "dtype": "<c16", "name": "UV_SAMPLING"},
    "uv_sampling_normalization": {
        "dims": norm_dims,
        "dtype": "<c16",
        "name": "UV_SAMPLING_NORMALIZATION",
    },
    "point_spread_function": {
        "dims": full_dims_lm,
        "dtype": "<f8",
        "name": "POINT_SPREAD_FUNCTION",
    },
    "visibility": {"dims": full_dims_uv, "dtype": "<c16", "name": "VISIBILITY"},
    "visibility_normalization": {
        "dims": norm_dims,
        "dtype": "<c16",
        "name": "VISIBILITY_NORMALIZATION",
    },
    "sky": {"dims": full_dims_lm, "dtype": "<f8", "name": "SKY"},
}

image_data_variables_and_dims_single_precision = {
    "aperture": {"dims": full_dims_uv, "dtype": "<c8", "name": "APERTURE"},
    "aperture_normalization": {
        "dims": norm_dims,
        "dtype": "<c16",
        "name": "APERTURE_NORMALIZATION",
    },
    "primary_beam": {"dims": full_dims_lm, "dtype": "<f4", "name": "PRIMARY_BEAM"},
    "uv_sampling": {"dims": full_dims_uv, "dtype": "<c8", "name": "UV_SAMPLING"},
    "uv_sampling_normalization": {
        "dims": norm_dims,
        "dtype": "<c16",
        "name": "UV_SAMPLING_NORMALIZATION",
    },
    "point_spread_function": {
        "dims": full_dims_lm,
        "dtype": "<f8",
        "name": "POINT_SPREAD_FUNCTION",
    },
    "visibility": {"dims": full_dims_uv, "dtype": "<c8", "name": "VISIBILITY"},
    "visibility_normalization": {
        "dims": norm_dims,
        "dtype": "<c16",
        "name": "VISIBILITY_NORMALIZATION",
    },
    "sky": {"dims": full_dims_lm, "dtype": "<f4", "name": "SKY"},
}


def pad_array_with_nans(input_array, output_shape, dtype):
    """
    Pad an integer array with NaN values to match the specified output shape.

    Parameters:
    - input_array: The input NumPy array to be padded.
    - output_shape: A tuple specifying the desired output shape (e.g., (rows, columns)).

    Returns:
    - A NumPy array with NaN padding to match the specified output shape.
    """
    # Get the input shape
    input_shape = input_array.shape

    # Calculate the padding dimensions
    padding_shape = tuple(max(0, o - i) for i, o in zip(input_shape, output_shape))

    # Create a new array filled with NaN values
    padded_array = np.empty(output_shape, dtype=dtype)
    padded_array[:] = np.nan

    # Copy the input array to the appropriate position within the padded array
    padded_array[: input_shape[0], : input_shape[1], : input_shape[2]] = input_array

    return padded_array


def write_binary_blob_to_disk(arr, file_path, compressor):
    """
    Compress a NumPy array using Blosc and save it to disk.

    Parameters:
    - arr: The NumPy array to compress and save.
    - file_path: The path to the output file where the compressed array will be saved.
    - compressor:

    Returns:
    - None
    """
    import toolviper.utils.logger as logger

    # Encode the NumPy array using the codec
    logger.debug("1. Before compressor " + file_path)
    compressed_arr = compressor.encode(np.ascontiguousarray(arr))

    logger.debug("2. Before makedir")
    # Ensure the directory exists before saving the file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    arr_len = len(compressed_arr)
    logger.debug("3. Before write the len is: " + str(arr_len))
    # Save the compressed array to disk
    # with open(file_path, "wb") as file:
    #     file.write(compressed_arr)

    logger.debug("4. Using new writer: " + str(arr_len))
    write_to_lustre_chunked(file_path, compressed_arr)

    # /.lustre/aoc/sciops/pford/CHILES/cube_image/uid___A002_Xee7674_X2844_Cube_3.img.zarr/SKY/0.0.110.0.0
    # 348192501 bytes
    # 332.0622453689575 M

    # from io import BufferedWriter
    # # Calculate buffer size based on compressed_arr size (adjust multiplier)
    # buffer_size = min(len(compressed_arr), 1024 * 1024 * 4)  # Max 4 MB buffer
    # with BufferedWriter(open(file_path, "wb"), buffer_size) as f:
    #     f.write(compressed_arr)
    #     f.flush()  # Ensure data gets written to disk

    logger.debug("4. Write completed")

    # print(f"Compressed array saved to {file_path}")


def write_to_lustre_chunked(
    file_path,
    compressed_arr,
    chunk_size=1024 * 1024 * 128,
):  # 128 MiB chunks
    """
    Writes compressed data to a Lustre file path with chunking.

    Args:
        file_path: Path to the file for writing.
        compressed_arr: Compressed data array to write.
        chunk_size: Size of each data chunk in bytes (default: 128 MiB).
    """
    fs, items = _get_file_system_and_items(file_path.rsplit("/", 1)[0])

    if isinstance(fs, s3fs.core.S3FileSystem):
        with fs.open(file_path, "wb") as f:
            for i in range(0, len(compressed_arr), chunk_size):
                chunk = compressed_arr[i : i + chunk_size]
                f.write(chunk)
    else:
        with open(file_path, "wb") as f:
            for i in range(0, len(compressed_arr), chunk_size):
                chunk = compressed_arr[i : i + chunk_size]
                f.write(chunk)


def read_binary_blob_from_disk(file_path, compressor, dtype=np.float64):
    """
    Read a compressed binary blob from disk and decode it using Blosc.

    Parameters:
    - file_path: The path to the compressed binary blob file.
    - compressor: The Blosc compressor to use (e.g., 'zstd', 'lz4', 'blosclz', etc.).

    Returns:
    - The decoded NumPy array.
    """

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None

    # Read the compressed binary blob from disk
    with open(file_path, "rb") as file:
        compressed_arr = file.read()

    # Decode the compressed data using the Blosc compressor
    decoded_bytes = compressor.decode(compressed_arr)

    decoded_arr = np.frombuffer(decoded_bytes, dtype)  # Adjust dtype as needed

    return decoded_arr


def read_json_file(file_path):
    """
    Read a JSON file and return its contents as a Python dictionary.

    Parameters:
    - file_path: The path to the JSON file to be read.

    Returns:
    - A Python dictionary containing the JSON data.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{file_path}': {e}")
        return None


class NumberEncoder(json.JSONEncoder):
    def default(self, o):
        # See json.JSONEncoder.default docstring for explanation
        # This is necessary to encode numpy dtype
        if isinstance(o, numbers.Integral):
            return int(o)
        if isinstance(o, numbers.Real):
            return float(o)
        return json.JSONEncoder.default(self, o)


def write_json_file(data, file_path):
    """
    Write a Python dictionary to a JSON file.

    Parameters:
    - data: The Python dictionary to be written to the JSON file.
    - file_path: The path to the JSON file to be created or overwritten.

    Returns:
    - None
    """

    with open(file_path, "w") as file:
        json.dump(
            data,
            file,
            indent=4,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ": "),
            cls=NumberEncoder,
        )


def create_data_variable_meta_data(
    zarr_group_name, data_variables_and_dims, xds_dims, parallel_coords, compressor
):
    zarr_meta = data_variables_and_dims

    fs, items = _get_file_system_and_items(zarr_group_name)

    for data_variable_key, dims_dtype_name in data_variables_and_dims.items():
        # print(data_variable_key, dims_dtype_name)

        dims = dims_dtype_name["dims"]
        dtype = dims_dtype_name["dtype"]
        data_variable_name = dims_dtype_name["name"]

        data_variable_path = os.path.join(zarr_group_name, data_variable_name)
        if isinstance(fs, s3fs.core.S3FileSystem):
            # N.b.,stateful "folder creation" is not a well defined concept for S3 objects and URIs
            # see https://github.com/fsspec/s3fs/issues/401
            # nor is a path specifier (cf. "URI")
            fs.mkdir(data_variable_path)
        else:
            # default to assuming we can use the os module and mkdir system call
            os.system("mkdir " + data_variable_path)
        # Create .zattrs
        zattrs = {
            "_ARRAY_DIMENSIONS": dims,
            # "coordinates": "time declination right_ascension"
        }

        shape = []
        chunks = []
        for d in dims:
            shape.append(xds_dims[d])
            if d in parallel_coords:
                chunks.append(len(parallel_coords[d]["data_chunks"][0]))
            else:
                chunks.append(xds_dims[d])

        # print(chunks,shape)
        # assuming data_variable_path has been set compatibly
        zattrs_file = os.path.join(data_variable_path, ".zattrs")

        if isinstance(fs, s3fs.core.S3FileSystem):
            with fs.open(zattrs_file, "w") as file:
                json.dump(
                    zattrs,
                    file,
                    indent=4,
                    sort_keys=True,
                    ensure_ascii=True,
                    separators=(",", ": "),
                    cls=NumberEncoder,
                )
        else:
            # default to assuming we can use primitives
            write_json_file(zattrs, zattrs_file)

        # Create .zarray
        from zarr import n5

        compressor_config = n5.compressor_config_to_zarr(
            n5.compressor_config_to_n5(compressor.get_config())
        )

        if "f" in dtype:
            zarray = {
                "chunks": chunks,
                "compressor": compressor_config,
                "dtype": dtype,
                "fill_value": "NaN",
                "filters": None,
                "order": "C",
                "shape": shape,
                "zarr_format": 2,
            }

        else:
            zarray = {
                "chunks": chunks,
                "compressor": compressor_config,
                "dtype": dtype,
                "fill_value": None,
                "filters": None,
                "order": "C",
                "shape": shape,
                "zarr_format": 2,
            }

        zarr_meta[data_variable_key]["chunks"] = chunks
        zarr_meta[data_variable_key]["shape"] = shape

        # again, assuming data_variable_path has been set compatibly
        zarray_file = os.path.join(data_variable_path, ".zarray")

        if isinstance(fs, s3fs.core.S3FileSystem):
            with fs.open(zarray_file, "w") as file:
                json.dump(
                    zarray,
                    file,
                    indent=4,
                    sort_keys=True,
                    ensure_ascii=True,
                    separators=(",", ": "),
                    cls=NumberEncoder,
                )
        else:
            # default to assuming we can use primitives
            write_json_file(zarray, zarray_file)

    return zarr_meta


def write_chunk(img_xds, meta, parallel_dims_chunk_id, compressor, image_file):
    dims = meta["dims"]
    dtype = meta["dtype"]
    data_variable_name = meta["name"]
    chunks = meta["chunks"]
    shape = meta["shape"]
    chunk_name = ""
    if data_variable_name in img_xds:
        for d in img_xds[data_variable_name].dims:
            if d in parallel_dims_chunk_id:
                chunk_name = chunk_name + str(parallel_dims_chunk_id[d]) + "."
            else:
                chunk_name = chunk_name + "0."
        chunk_name = chunk_name[:-1]

        if list(img_xds[data_variable_name].shape) != list(chunks):
            array = pad_array_with_nans(
                img_xds[data_variable_name].values,
                output_shape=chunks,
                dtype=dtype,
            )
        else:
            array = img_xds[data_variable_name].values

        write_binary_blob_to_disk(
            array,
            file_path=os.path.join(image_file, data_variable_name, chunk_name),
            compressor=compressor,
        )

        # z_chunk = zarr.open(
        #     os.path.join(image_file, data_variable_name, chunk_name),
        #     mode="a",
        #     shape=meta["shape"],
        #     chunks=meta["chunks"],
        #     dtype=meta["dtype"],
        #     compressor=compressor,
        # )

        # return z_chunk
