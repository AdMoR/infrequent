import pyspark
from pyspark.ml.fpm import FPGrowth
import pandas as pd
out = list()
with open("set1.train.txt") as f:
    lines = f.readlines()
    llines = map(lambda x: x.split(" "), lines)
    for i, x in enumerate(llines):
        if i > 10000:
            break
        out.append({y.split(":")[0] if ":" in y else None for y in x})
ref_df = pd.DataFrame(out)
spark = pyspark.sql.SparkSession.builder.getOrCreate()
df = spark.createDataFrame(ref_df)
fpGrowth = FPGrowth(itemsCol="items", minSupport=0.0, minConfidence=0.0)
model = fpGrowth.fit(df)