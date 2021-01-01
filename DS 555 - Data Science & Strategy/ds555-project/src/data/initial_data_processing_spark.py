from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pandas as pd
from pyspark.sql import Row

# Import data types
from pyspark.sql.types import StringType, StructType, StructField


def initial_data_preparation(spark):
    df = spark.read.csv("../../data/raw/survey_results_public.csv", header=True)
    print(df.count())
    # Drop unnecessary columns.
    columns_to_drop = ["CurrencySymbol", "CompFreq", "CurrencyDesc", "CompTotal", "Respondent"]
    df = df.drop(*columns_to_drop)

    # Process Columns
    df = (df.withColumn("Age1stCode", F.regexp_replace(F.col("Age1stCode"), 'Younger than 5 years', "3"))
          .withColumn("Age1stCode", F.regexp_replace(F.col("Age1stCode"), 'Older than 85', "90"))
          .withColumn("Age1stCode", F.col("Age1stCode").cast(T.FloatType()))
          .withColumn("YearsCode", F.regexp_replace(F.col("YearsCode"), 'Less than 1 year', "0.5"))
          .withColumn("YearsCode", F.regexp_replace(F.col("YearsCode"), 'More than 50 years', "55"))
          .withColumn("YearsCode", F.col("YearsCode").cast(T.FloatType()))
          .withColumn("YearsCodePro", F.regexp_replace(F.col("YearsCodePro"), 'Less than 1 year', "0.5"))
          .withColumn("YearsCodePro", F.regexp_replace(F.col("YearsCodePro"), 'More than 50 years', "55"))
          .withColumn("YearsCodePro", F.col("YearsCodePro").cast(T.FloatType()))
          )

    # Detect multi-choice columns.
    multi_choice_counts = {col_: df.select(col_).filter(F.col(col_).contains(";")).count() for col_ in df.columns}
    multi_choice_columns = [key for key, value in multi_choice_counts.items() if value > 0]

    def get_all_distinct_choices(col_name):
        distinct_choices = df.select(col_name).distinct().collect()
        distinct_choices = [row[col_name] for row in distinct_choices]

        list_of_choices = [str(item).split(";") for item in distinct_choices]
        all_choices = []
        for ch in list_of_choices:
            all_choices += ch

        if 'NA' in all_choices:
            all_choices.remove('NA')
        all_choices = pd.Series(all_choices).unique().tolist()
        return all_choices

    distinct_choice_lists_for_each_columns = {column: get_all_distinct_choices(column) for column in
                                              multi_choice_columns}

    # Process multi-choice columns.
    def sep_multi_choice(row_):
        row = row_.asDict()
        for column in multi_choice_columns:
            for choise in distinct_choice_lists_for_each_columns[column]:
                if type(row[column]) is str:
                    if choise in row[column]:
                        row[column + "_" + choise] = 1
                    else:
                        row[column + "_" + choise] = 0
        new_row = Row(**row)
        return new_row

    print("Multi choice columns processing is started.")
    new_rdd = df.rdd
    new_rdd = new_rdd.map(sep_multi_choice)
    df = spark.createDataFrame(new_rdd)
    # df = df.drop(*multi_choice_columns)
    print("Multi choice columns processed.")
    print(df.count())
    print(df.columns)
    df = df.toPandas()
    df.to_csv("../../data/interim/spark_processed_data.csv", index=False, sep="|")
    pd.Series(multi_choice_columns).to_csv("../../data/interim/spark_multi_choice_columns.csv", index=False, sep="|")
    pd.DataFrame.from_dict(distinct_choice_lists_for_each_columns, orient="index").T.to_csv("../../data/interim/spark_multi_choice_options.csv", index=False, sep="|")
    return df


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .master("local[8]") \
        .appName("Python Spark SQL basic example") \
        .getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    df = initial_data_preparation(spark)

    # spark.stop()
