# Reynolds-number split

The source data consist of separate training, validation, and test HDF5 files with 80, 10, and 10 Reynolds-number groups, respectively. The HDF5 files are available from the corresponding author upon request. Exact group names should be exported from those files using:

```bash
python -m data.inspect_split --train /path/to/train.hdf5 --validation /path/to/validation.hdf5 --test /path/to/test.hdf5 --output re_split.json
```

The script verifies that the three case lists are disjoint. The archive does not invent a case list without reading the underlying files.
