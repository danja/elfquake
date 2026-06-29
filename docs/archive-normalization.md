# Archive Normalization

Archive normalizers live in `src/elfquake/normalize/space_weather.py`.

Current normalizers:

* GFZ Kp/Ap text to `date,slot,kp,ap,source_file`
* Kyoto Dst text to `date,hour,dst_nt,source_file`
* daily F10.7 JSON or simple text to `date,f107,source_file`
* GOES XRS NetCDF to `time_utc,variable,value,units,source_file`

GOES XRS extraction requires `netCDF4`.

Installed choices:

* Debian/Ubuntu system Python: `sudo apt install python3-netcdf4`
* virtualenv: `python3 -m pip install netCDF4`

The package also requires the native NetCDF/HDF5 stack. Prefer the OS package on long-running hosts unless the project is deployed inside a managed virtualenv.
