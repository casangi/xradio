from xradio.vis.model import *

time_axis = TimeAxis.new(numpy.arange(10))
bl_axis = BaselineAxis.new(numpy.arange(10))
freq_axis = FrequencyAxis.new(numpy.arange(100e6, 150e6, 2.5e6), spectral_coord=None)
pol_axis = PolarizationAxis.new(["I"])

vis = VisibilityXds.new(
    VISIBILITY=VisibilityArray.empty(
        dict(time=10, baseline_id=10, channel=20, polarization=1),
        time=time_axis,
        baseline_id=bl_axis,
        frequency=freq_axis,
        polarization=pol_axis,
    ),
    FLAG=FlagArray.empty(
        dict(time=10, baseline_id=10, channel=20, polarization=1),
        time=time_axis,
        baseline_id=bl_axis,
        frequency=freq_axis,
        polarization=pol_axis,
    ),
    time=time_axis,
    baseline_id=bl_axis,
    frequency=freq_axis,
    polarization=pol_axis,
    field_info=None,
    source_info=None,
    antenna_xds=None,
)
