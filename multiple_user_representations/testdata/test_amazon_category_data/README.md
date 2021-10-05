# Data Format for real-data experiments.

## The directory for preprocessed data should contain the following

+   <b> User item interactions: </b> A text file containing user item
    interactions. The file name should be `user_item_mapped.txt`. The expected
    format for each line is `<user_id> <item_id> <timestamp>`. Each line must be
    sorted with respect to the timestamp. The ids for users and items should be
    integers starting with `1`.
+   <b> User negative items: </b> A text file named `user_neg_items.txt`
    containing candidate items for evaluating metrics. See the
    [proposal](https://docs.google.com/document/d/1yGNtBrRIExcP4Fr_SZrX7kplm_C4DXiY_dbmYF1uTpg/edit#bookmark=id.4dm5yebd3f21)
    doc for the evaluation protocol followed for real data experiments. The
    expected format for each line is `<user_id>: <item_id_1>...<item_id_N>`.

A template is provided in the `test_amazon_category_data` directory.
