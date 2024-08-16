## Zero Shot Cardinality Training

This repository contains the infrastructure to produce training datasets for
cardinality estimation. This process is a multi-step process:

* Create down sampled versions of databases, if needed
* Calculate table (collect table/column information, calculate statistics table/column statistics)
* Generate training sql queries
* Run queries to collect actual cardinalities
* Create training datasets (querygraps)

![CardBench](training_datasets/figures/cardbench.png)


As the cost of running this pipeline is significant we plan to release the final
product (the training dataset) in addition to the code that performs all the steps.

The training datasets and their documentation can be found in the training_datasets directory.

<span style="color:red">
We are releasing the code to create the CardBench training datasets incrementally.
</span>


## General Information & Setup

Any statistic and information that is collected or calculated is stored in a set of database tables. The ``statistics_sql_tables_definition.sql`` script generates all the necessary tables. After the tables are created 
please update the ``configuration.py`` file with the ids of the tables. 

The code queries the tables of the "data" database (these are the tables for which we collect cardinalities) and the "metadata" (or statistics) database (these tables store the statistics we calculate). These two databases can be stored in separate systems.

The code was initially designed to work with Big Query as the database backend for both the data and metadata databases. In ``database_connector.py`` we provide an extensible database connector that can be extended to work with 
any database. Changes in the rest of the code will be needed to support the functionality needed to calculate some statistics (for example percentiles requires a percientile SQL function or discovering the schema of a table requires calling the database specific API that returns the column names and types of a table.)

### Calculate Statistics

The first step is to calculate statistics and collect information about the databases. 
The ``calculate_statistics.py`` file runs this step. The ``calculate_statistics_library.py`` contains the relevant code.


<span style="color:red">
As this step requires some effort to replicate, we will also release the collected statistics in this repository.
</span>

### Generate Queries
TODO

### Execute Queries
TODO

### Generate Annotated Query Graphs
TODO

