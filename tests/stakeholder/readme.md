# Single Dish Dataset for `xradio`/MSv4 Development

## Selection Strategy

Data were chosen to cover as much observing modes as possible. The following observing properties were taken into account.

* number of target: single target/multiple targets
* strategy of OFF position: absolute/relative/horizontal
    * absolute OFF position has its own field entry in FIELD table
    * others do not have explicit field entry in FIELD table
    * relative OFF is fixed position specified as offset coordinate relative to ON position
    * horizontal OFF is also an offset position relative to ON position but specified so that its elevation value is always close to ON position (meaning that OFF position is moving with time)
* ephemeris/non-ephemeris
* scan pattern: raster scan/fast scan
    * raster scan is, as name indicates, raster pattern with relatively long integration time (0.1-1sec) and channelized spw
    * fast scan is either Lissajous or double-circle pattern with short integration time (1msec) and single channel spw

## Description of Selected Dataset

### uid___A002_Xced5df_Xf9d9.small.ms

* single target
* horizontal OFF
* non-ephemeris
* raster scan
* from MOUS uid://A001/X1284/X2893
* public data
* original size: 2.3GB
* reduced size: 50MB
* reduction commands:
    ```
    importasdm(asdm='uid___A002_Xced5df_Xf9d9', vis='uid___A002_Xced5df_Xf9d9.ms',
           createmms=False, ocorr_mode='ao', lazy=False, process_caldevice=False, savecmds=True,
           outfile='uid___A002_Xced5df_Xf9d9.flagonline.txt',
           overwrite=False,
           bdfflags=True, with_pointing_correction=True)
    split(vis='uid___A002_Xced5df_Xf9d9.ms',
          outputvis='uid___A002_Xced5df_Xf9d9.small.ms',
          spw='17:2047~2049,19:63~65,21:511~513,23:511~513',
          antenna='0~1&&&',datacolumn='float_data',
          keepflags=True,width=1)
    tb.open('uid___A002_Xced5df_Xf9d9.small.ms/POINTING')
    tsel = tb.query('ANTENNA_ID IN [0,1]')
    tcpy = tsel.copy('uid___A002_Xced5df_Xf9d9.small.ms/POINTING.small', deep=True)
    tcpy.close()
    tsel.close()
    tb.close()
    !rm -rf uid___A002_Xced5df_Xf9d9.small.ms/POINTING
    !mv uid___A002_Xced5df_Xf9d9.small.ms/POINTING.small uid___A002_Xced5df_Xf9d9.small.ms/POINTING
    ```

### uid___A002_X1015532_X1926f.ms

* single target
* relative OFF
* ephemeris (Jupiter)
* raster scan
* from MOUS uid://A001/X2d1f/Xa0
* **proprietary until November 2024**
* original size: 709MB
* reduced size: 22MB
* reduction commands:
    ```
    importasdm(asdm='uid___A002_X1015532_X1926f',
           vis='uid___A002_X1015532_X1926f.ms', createmms=False,
           ocorr_mode='ao', lazy=False,
           process_caldevice=False, savecmds=True,
           outfile='uid___A002_X1015532_X1926f.flagonline.txt', overwrite=False,
           bdfflags=True, with_pointing_correction=True)
    split(vis='uid___A002_X1015532_X1926f.ms',
          outputvis='uid___A002_X1015532_X1926f.small.ms',
          spw='17:2047~2050,19:63~66,21:2047~2050,23:2047~2050',
          datacolumn='float_data',keepflags=True,width=1)
    ```

### uid___A002_Xe3a5fd_Xe38e.small.ms

* multiple targets
* absolute OFF
* non-ephemeris
* raster scan
* from MOUS uid://A001/X1465/X2015
* public data
* original size: 937MB
* reduced size: 49MB
* reduction commands:
    ```
    importasdm(asdm='uid___A002_Xe3a5fd_Xe38e', vis='uid___A002_Xe3a5fd_Xe38e.ms',
               createmms=False, ocorr_mode='ao', lazy=False,    process_caldevice=False, savecmds=True,
               outfile='uid___A002_Xe3a5fd_Xe38e.flagonline.txt', overwrite=False,
               bdfflags=True, with_pointing_correction=True)
    split(vis='uid___A002_Xe3a5fd_Xe38e.ms',
          outputvis='uid___A002_Xe3a5fd_Xe38e.small.ms',
          spw='17:1856~2143,19:960~1039,21:960~1039,23:960~1039',
          datacolumn='float_data',keepflags=True,width=8)
    ```

### uid___A002_Xae00c5_X2e6b.small.ms

* single target
* relative OFF
* ephemeris (Sun)
* fast scan
* public
    * cf. [CASAguides: Sunspot Band6 SingleDish](https://casaguides.nrao.edu/index.php?title=Sunspot_Band6_SingleDish_for_CASA_6.5.4)
* original size: 455MB
* reduced size: 47MB
* reduction commands:
    ```
    importasdm(asdm='uid___A002_Xae00c5_X2e6b', vis='uid___A002_Xae00c5_X2e6b.ms',
               createmms=False, ocorr_mode='ao', lazy=False,
               process_caldevice=False, savecmds=True,
               outfile='uid___A002_Xae00c5_X2e6b.flagonline.txt',
               overwrite=False, bdfflags=True, with_pointing_correction=True)
    split(vis='uid___A002_Xae00c5_X2e6b.ms',
          outputvis='uid___A002_Xae00c5_X2e6b.small.ms',keepmms=True,
          spw='0~1', scan='2', antenna='0~1&&&', timerange='20:12:14~20:14:34',
          datacolumn='float_data', width=1)
    tb.open('uid___A002_Xae00c5_X2e6b.small.ms/POINTING')
    tsel = tb.query('ANTENNA_ID IN [0,1]')
    tcpy = tsel.copy('uid___A002_Xae00c5_X2e6b.small.ms/POINTING.small', deep=True)
    tcpy.close()
    tsel.close()
    tb.close()
    !rm -rf uid___A002_Xae00c5_X2e6b.small.ms/POINTING
    !mv uid___A002_Xae00c5_X2e6b.small.ms/POINTING.small uid___A002_Xae00c5_X2e6b.small.ms/POINTING
    ```

## Data Property Spreadsheet

| dataset | target | OFF | ephemeris | scan pattern |
|---------|--------|-----|-----------|--------------|
| uid___A002_Xced5df_Xf9d9small.ms | single | horizontal | N | raster |
| uid___A002_Xe3a5fd_Xe38e.small.ms | single | relative | Y | raster |
| uid___A002_X1015532_X1926f.small.ms | multi | absolute | N | raster |
| uid___A002_Xae00c5_X2e6b.small.ms | single | relative | Y | fast |


