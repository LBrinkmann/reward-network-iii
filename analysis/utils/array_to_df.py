import pandas as pd


def using_multiindex(A, columns, value_columns=None, value_name='value'):
    """
    Turns a multi-dimension numpy array into a single dataframe.

    Args:
        A: multi-dimensional numpy array.
        columns: the column name for each dimension.
        value_columns: the last dimension can be turned into separate columns
            filled with the individual values;
            if not None: needs to be list matching in length the size of the
            last dimension of A.
            if 'None': all values will be stored in a single column
        value_name: name for column storing the matrix values (only relevant if
        value_columns == None)
    """
    shape = A.shape
    if value_columns is not None:
        assert len(columns) == len(shape) - 1
        new_shape = (-1, len(value_columns))
    else:
        assert len(columns) == len(shape)
        new_shape = (-1,)
        value_columns = [value_name]

    index = pd.MultiIndex.from_product(
        [range(s) for s, c in zip(shape, columns)], names=columns)
    df = pd.DataFrame(A.reshape(*new_shape), columns=value_columns, index=index)
    df = df.reset_index()
    return df