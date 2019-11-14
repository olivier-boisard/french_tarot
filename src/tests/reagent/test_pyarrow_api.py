import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def test_write_state_action_df_to_table(request):
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))
    table = pa.Table.from_pandas(df)

    tmp_filename = "tmp.parquet"
    pq.write_table(table, tmp_filename)
    request.addfinalizer(lambda: os.remove(tmp_filename))

    df_read = pq.read_table(tmp_filename).to_pandas()
    assert (df == df_read).values.all()
