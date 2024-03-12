import os
import numpy as np
import json

from numcodecs.compat import (
    ensure_text,
    ensure_ndarray_like,
    ensure_bytes,
    ensure_contiguous_ndarray_like,
)

full_dims_lm = ["time", "polarization", "frequency", "l", "m"]
full_dims_uv = ["time", "polarization", "frequency", "l", "m"]
norm_dims = ["polarization", "frequency"]

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
    # Encode the NumPy array using the codec
    compressed_arr = compressor.encode(np.ascontiguousarray(arr))

    # Ensure the directory exists before saving the file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the compressed array to disk
    with open(file_path, "wb") as file:
        file.write(compressed_arr)

    # print(f"Compressed array saved to {file_path}")


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


def create_data_variable_meta_data_on_disk(
    zarr_group_name, data_variables_and_dims, xds_dims, parallel_coords, compressor
):
    zarr_meta = data_variables_and_dims

    for data_variable_key, dims_dtype_name in data_variables_and_dims.items():
        # print(data_variable_key, dims_dtype_name)

        dims = dims_dtype_name["dims"]
        dtype = dims_dtype_name["dtype"]
        data_variable_name = dims_dtype_name["name"]
        data_variable_path = os.path.join(zarr_group_name, data_variable_name)
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
        write_json_file(zattrs, os.path.join(data_variable_path, ".zattrs"))

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

        write_json_file(zarray, os.path.join(data_variable_path, ".zarray"))
    return zarr_meta



def write_chunk(img_xds,meta,parallel_dims_chunk_id,compressor,image_file):
    dims = meta["dims"]
    dtype = meta["dtype"]
    data_varaible_name = meta["name"]
    chunks = meta["chunks"]
    shape = meta["shape"]
    chunk_name = ""
    if data_varaible_name in img_xds:
        for d in img_xds[data_varaible_name].dims:
            if d in parallel_dims_chunk_id:
                chunk_name = chunk_name + str(parallel_dims_chunk_id[d]) + "."
            else:
                chunk_name = chunk_name + "0."
        chunk_name = chunk_name[:-1]

        if list(img_xds[data_varaible_name].shape) != list(chunks):
            array = pad_array_with_nans(
                img_xds[data_varaible_name].values,
                output_shape=chunks,
                dtype=dtype,
            )
        else:
            array = img_xds[data_varaible_name].values

        write_binary_blob_to_disk(
            array,
            file_path=os.path.join(
                image_file, data_varaible_name, chunk_name
            ),
            compressor=compressor,
        )