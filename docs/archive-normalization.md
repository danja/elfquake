# Archive Normalization

Archive normalizers live in `src/elfquake/normalize/space_weather.py`.

Current stubs:

* GFZ Kp/Ap text to `date,slot,kp,ap,source_file`
* Kyoto Dst text to `date,hour,dst_nt,source_file`
* daily F10.7 JSON or simple text to `date,f107,source_file`
* GOES XRS NetCDF boundary row marked `requires_netcdf_decoder`

The GOES XRS archive needs a NetCDF reader before numeric flux extraction. Do not add that dependency without an explicit install step.
