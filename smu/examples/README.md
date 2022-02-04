# Examples

This directory contains examples of how to access the SMU databases with Python.

It assumes you have followed [the setup instructions](../README.md)

The following examples are provided
- `field_access.py` : basic access to different kinds of fields
- `indices.py` : using the various indices in the database for faster lookups
- `to_csv.py` : generates a CSV file with a few fields
- `dataframe.py` : generates a [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) of some field values
- `to_rdkit.py` : converts Conformers to RDKit molecules
- `missing_fields.py` : how to check for missing fields
- `multiple_bond_topology.py` : illustrates some points around multiple BondTopology for each Conformer
- `special_cases.py` : in the complete database, there are a number of records which have a quite different set of information in them
- `c6h6.py` : a more complex example bringing together many of the concepts from other examples

To run them, after you have activated your virtual environment, go to the directory where you have the database files (the .sqlite files) and run a command like this (note no `.py` extension)

        python -m smu.examples.field_access

You can also execute the examples with

        python <path to the examples directory>/field_access.py

In addition, there is a `.out` file for each `.py` file which gives the expected standard output for that script
