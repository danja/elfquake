# ELFQuake

predicting earthquakes with radio

documentation will be at https://elfquake.org/

## Status 2024-08-03

A new beginning!

I've shifted the previous repo to [elfquake-old](https://github.com/danja/elfquake-old)

---

current material is at

I'm returning to this after a few years absence. Again.

I've accumulated a lot of junk in this repo, so I'll retire this, start a new one.

See [ELFQuake blog](https://elfquake.wordpress.com/)

**Status 2023-05-29**

I'm returning to this after a few years absence.

Currently reorganising things a bit. More material to follow.

### Pre-2023 :

Collect seismic data from INGV, train a PredNet network

ingv/get-ingv-data.py

- pulls INGV data, filters, dumps to CSV files (working)

ingv/aggregate.py

- filter/aggregate CSV data, dumps to HDF5 file (in progress)
