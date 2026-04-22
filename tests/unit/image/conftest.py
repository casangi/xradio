"""pytest fixtures shared across image unit tests."""

import pytest
from toolviper.dask.client import local_client


@pytest.fixture(scope="module")
def dask_client_module():
    """Set up and tear down a Dask client for the test module.

    Starts a local Dask cluster with limited resources before any test in
    the module runs and ensures both the client and the cluster are closed
    after all tests complete.

    Returns
    -------
    distributed.Client
        A Dask client connected to a local cluster, shared across all tests
        in the module.
    """
    print("\nSetting up Dask client for the test module...")
    client = local_client(
        cores=2, memory_limit="3GB"
    )  # Do not increase size – GitHub macOS runners will hang.
    try:
        yield client
    finally:
        print("\nTearing down Dask client for the test module...")
        if client is not None:
            client.close()
            cluster = getattr(client, "cluster", None)
            if cluster is not None:
                cluster.close()
