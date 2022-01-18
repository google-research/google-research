# Examples

This directory contains examples of how to access the SMU databases with Python.

It assumes you have followed [the setup instructions](../README.md)

The following examples are provided
- `field_access.py` : basic access to different kinds of fields
- `missing_fields.py` : how to check for missing fields
- `indices.py` : using the various indices in the database for faster lookups
- `csv.py` : generates a CSV file with a few fields
- `dataframe.py` : generates a [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) of some field values
- `rdkit.py` : converts Conformers to RDKit molecules
- `multiple_bond_topology.py` : illustrates some points around multiple BondTopology for each Conformer
- `special_cases.py` : in the complete database, there are a number of records which have a quite different set of information in them

To run them, after you have activated your virtual environment, go to the directory where you have the database files (the .sqlite files) and run a command like this (note no `.py` extension)

        python -m smu.examples.field_access
