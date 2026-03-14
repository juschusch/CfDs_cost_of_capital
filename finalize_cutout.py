import atlite
from atlite import Cutout

import logging
logging.getLogger("atlite").setLevel(logging.DEBUG)
logging.getLogger("cdsapi").setLevel(logging.DEBUG)

import glob, os
print(os.path.abspath("ERA5_weather_data_2000-24"))
print(glob.glob("ERA5_weather_data_2000-24/**/*", recursive=True)[:50]) 

cutout = Cutout(
    "cutouts/de-2000_24-era5",
    module="era5",
    time=slice("2000-01-01", "2024-12-31"),
    x=slice(5.0, 16.0),
    y=slice(47.0, 56.0),
    dx=1.0,
    dy=1.0,
)

cutout.prepare(
    features=None,
    tmpdir="ERA5_weather_data_2000-24",
    data_format="grib",
    monthly_requests=False,
    show_progress=True,
)


import xarray as xr
fn = r"ERA5_weather_data_2000-24/tmpx02sc4aode-2000_24-era5.nc"
ds_tmp = xr.open_dataset(fn, chunks={'time': 100})  # adjust chunk dims to memory/CPU
print(ds_tmp)
print("coords:", list(ds_tmp.coords))
print("vars:", list(ds_tmp.data_vars))
print("time range:", ds_tmp.coords['time'].values[0], ds_tmp.coords['time'].values[-1])


fn_2 = r"cutouts/de-2004_12-era5.nc"
ds_tmp_2 = xr.open_dataset(fn_2)
print(ds_tmp_2)
print("coords:", list(ds_tmp_2.coords))
print("vars:", list(ds_tmp_2.data_vars))
print("time range:", ds_tmp_2.coords['time'].values[0], ds_tmp_2.coords['time'].values[-1])