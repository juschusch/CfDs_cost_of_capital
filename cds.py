import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_dewpoint_temperature",
        "2m_temperature",
        "100m_u_component_of_wind",
        "100m_v_component_of_wind",
        "surface_net_solar_radiation",
        "surface_solar_radiation_downwards",
        "toa_incident_solar_radiation",
        "total_sky_direct_solar_radiation_at_surface",
        "runoff",
        "soil_temperature_level_1",
        "forecast_surface_roughness",
        "geopotential"
    ],
    "year": ["2004"],
    "month": [
        "01", "02"#, "03",
        #  "04", "05", "06",
        # "07", "08", "09",
        # "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [56, 5, 47, 16]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
