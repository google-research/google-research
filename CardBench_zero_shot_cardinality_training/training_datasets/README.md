## Training Datasets

This directory contains the training datasets of CardBench, Single Table and Binary Join.
Each training instance is a SQL query represented as an annotated graph. By running the queries on Google Big Query we obtained the query cardinality and execution time that are included as context in the graph.

CardBench includes two training datasets, Single Table and Binary Join. The two datasets differ in the operations their queries include. Single Table contains queries that apply 1-4 filter predicates on one table, Binary Join contains queries that join two tables and apply 1-3 filter predicates per table.

Example Single Table Query:

```
SELECT count(*) FROM tpch_10G.nation as nation  WHERE nation.n_nationkey <= 6 AND nation.n_comment IS NULL AND nation.n_regionkey >= 1;
```

Example Binary Join Query:

```
SELECT count(*) FROM tpch_10G.region as region JOIN tpch_10G.nation as nation ON region.r_regionkey = nation.n_regionkey  WHERE nation.n_comment IS NOT NULL AND nation.n_nationkey != 5;
```

The training datasets are stored in the single_table and binary_join directories. The query graphs of each database are saved in separate files. For example, the query graphs of single tables queries for the consumer database are stored in the files:
```
consumer_single_table.npz-0-to-800
consumer_single_table.npz-800-to-1600
...
consumer_single_table.npz-5600-to-5961
```
The naming scheme for the files is: database_name_\<single_table/binary_join\>.npz-X-to-Y. The figure below lists the database names, table count for each database and the number of queries graphs. A ~ in the database name marks databases that we have down sampled.

![training datasets](figures/training_datasets.png)


### Training Data Schema

The picture below illustrates the graph structure. On the right of the figure we visualize the graph for the example query shown under it. On the left of the figure we list the node features and their data type per graph node.

![graph figure](figures/graph_figure.png)


### How to read the training data

The training data are encoded using the Graph Struct defined by the [Sparse Deferred](https://github.com/google-research/google-research/tree/master/sparse_deferred). Sparse Deferred provides an easy way to write, read and store the training data and also serialized them in a TF/JAX friendly format.

The following code provides an example on how to read the npz files and how to access nodes, node featured and top level information (cardinality, query, etc). 
To run the code you need python >= 3.10, tpqm, glob and numpy.

```
  import glob
  import tqdm

  from sparse_deferred.structs import graph_struct

  GraphStruct = graph_struct.GraphStruct
  InMemoryDB = graph_struct.InMemoryDB

  # the filename is the filename without the shard numbers
  # here consumer_single_table.npz instead of consumer_single_table.npz-0-to-800
  filename = "single_table/consumer_single_table.npz"
  filenames = glob.glob(filename + '-*')
  filenames.sort(key=lambda f: int(f.split('-')[-1]))
  db = InMemoryDB()

  for file in tqdm.tqdm(filenames):
    db_temp = InMemoryDB.from_file(file)
    for i in range(db_temp.size):
      db.add(db_temp.get_item(i))
  db.finalize()

  # print the number of training instances
  print("Number of training instances:", db.size)

  # print the schema of each training instance
  print("Schema:", db.schema)

  # print the first training example
  first_training_example = db.get_item(0)
  print("First training example:", first_training_example)

  # print node types
  print("Node types:", first_training_example.nodes.keys())

  # print table nodes features
  print("Table node features:", db.get_item(0).nodes["tables"].keys())

  # print edge types
  print("Edge types:", first_training_example.edges.keys())

  # print number of rows and name of the first table node
  print(
      "First table number of rows:", db.get_item(0).nodes["tables"]["rows"][0]
  )
  print("First table name:", db.get_item(0).nodes["tables"]["name"][0])

  # print query cardinality, query, query_id, execution_time
  # these are graph level features
  print(
      "Query cardinality:",
      first_training_example.nodes["g"]["cardinality"][0],
  )
  print("Execution time:", first_training_example.nodes["g"]["exec_time"][0])
  print("Query id:", first_training_example.nodes["g"]["query_id"][0])
  print("Query:", first_training_example.nodes["g"]["query"][0])

```

The schema of the annotated graphs as printed by the code above shows the types of edges and the node types they connect:

```
{'table_to_attr': ('tables', 'attributes'), 'attr_to_pred': ('attributes', 'predicates'), 'pred_to_pred': ('predicates', 'predicates'), 'attr_to_op': ('attributes', 'ops'), 'op_to_op': ('ops', 'ops'), 'pred_to_op': ('predicates', 'ops'), 'attr_to_corr': ('attributes', 'correlations'), 'corr_to_pred': ('correlations', 'predicates')}
```


### Notes

* Depending on the type of an attribute the corresponding feature is populated. For example percentiles_str are populated for an attributes with type string and percentiles_num are populated for numeric attributes. The empty feature is filled with the value -1.
* To download the training datasets please follow the [instructions here](https://github.com/google-research/google-research/tree/master?tab=readme-ov-file#google-research)