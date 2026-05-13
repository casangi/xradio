import warnings as _warnings

# Zarr v3 marks some on-disk specifications as not yet finalized and emits
# UnstableSpecificationWarning whenever they are written or read. XRADIO
# knowingly opts in to those formats, so silence the warning package-wide.
try:
    from zarr.errors import UnstableSpecificationWarning as _UnstableSpecificationWarning

    _warnings.filterwarnings("ignore", category=_UnstableSpecificationWarning)
except ImportError:
    # Older/newer zarr versions may not expose the class at this path; fall
    # back to matching by qualified name so the filter still works.
    _warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*",
        module=r"zarr\..*",
    )
