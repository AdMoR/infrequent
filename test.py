from unittest import TestCase
import pyspark
from pyspark.sql.types import IntegerType, StringType
import os
from pyspark.ml.fpm import FPGrowth
import pandas as pd

class Test(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        out = list()
        with open("set1.train.txt") as f:
            lines = f.readlines()
            llines = map(lambda x: x.split(" "), lines)
            for i, x in enumerate(llines):
                if i > 1000:
                    break
                out.append([str(y.split(":")[0]) if ":" in y else "" for y in x])
        cls.out = [pyspark.Row(id=i, items=x) for i, x in enumerate(out)]
        cls.spark = pyspark.sql.SparkSession.builder.master("local[*]").getOrCreate()

    def test(self):
        df = self.spark.createDataFrame(self.out)
        fpGrowth = FPGrowth(itemsCol="items", minSupport=0.0, minConfidence=0.0)
        model = fpGrowth.fit(df)
        print(model.associationRules.show())