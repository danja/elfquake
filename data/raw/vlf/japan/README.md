# Japan passive ELF/VLF captures

This directory follows the Cumiana manifest and capture layout. It is reserved
for **passive broadband natural-radio** data, not transmitter amplitude/phase
paths.

The manifest is intentionally empty until a reproducible machine-fetchable
Japan station or archive endpoint is confirmed. The station registry records
two strong candidates from Nagoya University ISEE: Moshiri (MSR) and Kagoshima
(KAG). Their public pages provide plots and data inventories, while raw digital
data currently require a request. WALDO is an additional archive candidate;
record the exact station URL, frequency range, license, and overlap with the
seismic catalogue before adding it to `manifest.csv`.

Once an endpoint is verified, add one row with `receiver_mode` set to
`passive_broadband_elf_vlf`, then run:

```sh
./scripts/capture-japan-vlf-loop.sh
```

Station registry: `stations.csv`.
