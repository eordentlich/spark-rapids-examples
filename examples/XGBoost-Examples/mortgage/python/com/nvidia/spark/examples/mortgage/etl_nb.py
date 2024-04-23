#
# Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from .consts_nb import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from sys import exit

def _get_quarter_from_csv_file_name():
    return substring_index(substring_index(input_file_name(), ".", 1), "/", -1)

def load_data(spark, paths, schema, args, extra_csv_opts={}):
    reader = (spark
        .read
        .format(args.format)
        .option('asFloats', args.asFloats)
        .option('maxRowsPerChunk', args.maxRowsPerChunk))
    if args.format == 'csv':
        (reader
            .schema(schema)
            .option('delimiter', '|')
            .option('header', False))
        for k, v in extra_csv_opts.items():
            reader.option(k, v)
    return reader.load(paths).withColumn('quarter', _get_quarter_from_csv_file_name())

def prepare_rawDf(spark, args):
    extra_csv_options = {
        'nullValue': '',
        'parserLib': 'univocity',
    }
    paths = extract_paths(args.dataPaths, 'data::')
    rawDf = load_data(spark, paths, csv_raw_schema, args, extra_csv_options)

    return rawDf

def extract_perf_columns(rawDf):
    perfDf = rawDf.select(
      col("loan_id"),
      date_format(to_date(col("monthly_reporting_period"),"MMyyyy"), "MM/dd/yyyy").alias("monthly_reporting_period"),
      upper(col("servicer")).alias("servicer"),
      col("interest_rate"),
      col("current_actual_upb"),
      col("loan_age"),
      col("remaining_months_to_legal_maturity"),
      col("adj_remaining_months_to_maturity"),
      date_format(to_date(col("maturity_date"),"MMyyyy"), "MM/yyyy").alias("maturity_date"),
      col("msa"),
      col("current_loan_delinquency_status"),
      col("mod_flag"),
      col("zero_balance_code"),
      date_format(to_date(col("zero_balance_effective_date"),"MMyyyy"), "MM/yyyy").alias("zero_balance_effective_date"),
      date_format(to_date(col("last_paid_installment_date"),"MMyyyy"), "MM/dd/yyyy").alias("last_paid_installment_date"),
      date_format(to_date(col("foreclosed_after"),"MMyyyy"), "MM/dd/yyyy").alias("foreclosed_after"),
      date_format(to_date(col("disposition_date"),"MMyyyy"), "MM/dd/yyyy").alias("disposition_date"),
      col("foreclosure_costs"),
      col("prop_preservation_and_repair_costs"),
      col("asset_recovery_costs"),
      col("misc_holding_expenses"),
      col("holding_taxes"),
      col("net_sale_proceeds"),
      col("credit_enhancement_proceeds"),
      col("repurchase_make_whole_proceeds"),
      col("other_foreclosure_proceeds"),
      col("non_interest_bearing_upb"),
      col("principal_forgiveness_upb"),
      col("repurchase_make_whole_proceeds_flag"),
      col("foreclosure_principal_write_off_amount"),
      col("servicing_activity_indicator"),
      col('quarter')
    )
    return perfDf.select("*").filter("current_actual_upb != 0.0")

def extract_acq_columns(rawDf):
    acqDf = rawDf.select(
      col("loan_id"),
      col("orig_channel"),
      upper(col("seller_name")).alias("seller_name"),
      col("orig_interest_rate"),
      col("orig_upb"),
      col("orig_loan_term"),
      date_format(to_date(col("orig_date"),"MMyyyy"), "MM/yyyy").alias("orig_date"),
      date_format(to_date(col("first_pay_date"),"MMyyyy"), "MM/yyyy").alias("first_pay_date"),
      col("orig_ltv"),
      col("orig_cltv"),
      col("num_borrowers"),
      col("dti"),
      col("borrower_credit_score"),
      col("first_home_buyer"),
      col("loan_purpose"),
      col("property_type"),
      col("num_units"),
      col("occupancy_status"),
      col("property_state"),
      col("zip"),
      col("mortgage_insurance_percent"),
      col("product_type"),
      col("coborrow_credit_score"),
      col("mortgage_insurance_type"),
      col("relocation_mortgage_indicator"),
      dense_rank().over(Window.partitionBy("loan_id").orderBy(to_date(col("monthly_reporting_period"),"MMyyyy"))).alias("rank"),
      col('quarter')
      )

    return acqDf.select("*").filter(col("rank")==1)

def _parse_dates(perf):
    return perf \
            .withColumn("monthly_reporting_period", to_date(col("monthly_reporting_period"), "MM/dd/yyyy")) \
            .withColumn("monthly_reporting_period_month", month(col("monthly_reporting_period"))) \
            .withColumn("monthly_reporting_period_year", year(col("monthly_reporting_period"))) \
            .withColumn("monthly_reporting_period_day", dayofmonth(col("monthly_reporting_period"))) \
            .withColumn("last_paid_installment_date", to_date(col("last_paid_installment_date"), "MM/dd/yyyy")) \
            .withColumn("foreclosed_after", to_date(col("foreclosed_after"), "MM/dd/yyyy")) \
            .withColumn("disposition_date", to_date(col("disposition_date"), "MM/dd/yyyy")) \
            .withColumn("maturity_date", to_date(col("maturity_date"), "MM/yyyy")) \
            .withColumn("zero_balance_effective_date", to_date(col("zero_balance_effective_date"), "MM/yyyy"))

def _create_perf_deliquency(spark, perf):
    aggDF = perf.select(
            col("quarter"),
            col("loan_id"),
            col("current_loan_delinquency_status"),
            when(col("current_loan_delinquency_status") >= 1, col("monthly_reporting_period")).alias("delinquency_30"),
            when(col("current_loan_delinquency_status") >= 3, col("monthly_reporting_period")).alias("delinquency_90"),
            when(col("current_loan_delinquency_status") >= 6, col("monthly_reporting_period")).alias("delinquency_180")) \
            .groupBy("quarter", "loan_id") \
            .agg(
                max("current_loan_delinquency_status").alias("delinquency_12"),
                min("delinquency_30").alias("delinquency_30"),
                min("delinquency_90").alias("delinquency_90"),
                min("delinquency_180").alias("delinquency_180")) \
            .select(
                col("quarter"),
                col("loan_id"),
                (col("delinquency_12") >= 1).alias("ever_30"),
                (col("delinquency_12") >= 3).alias("ever_90"),
                (col("delinquency_12") >= 6).alias("ever_180"),
                col("delinquency_30"),
                col("delinquency_90"),
                col("delinquency_180"))
    joinedDf = perf \
            .withColumnRenamed("monthly_reporting_period", "timestamp") \
            .withColumnRenamed("monthly_reporting_period_month", "timestamp_month") \
            .withColumnRenamed("monthly_reporting_period_year", "timestamp_year") \
            .withColumnRenamed("current_loan_delinquency_status", "delinquency_12") \
            .withColumnRenamed("current_actual_upb", "upb_12") \
            .select("quarter", "loan_id", "timestamp", "delinquency_12", "upb_12", "timestamp_month", "timestamp_year") \
            .join(aggDF, ["loan_id", "quarter"], "left_outer")

    # calculate the 12 month delinquency and upb values
    months = 12
    monthArray = [lit(x) for x in range(0, 12)]
    # explode on a small amount of data is actually slightly more efficient than a cross join
    testDf = joinedDf \
            .withColumn("month_y", explode(array(monthArray))) \
            .select(
                    col("quarter"),
                    floor(((col("timestamp_year") * 12 + col("timestamp_month")) - 24000) / months).alias("josh_mody"),
                    floor(((col("timestamp_year") * 12 + col("timestamp_month")) - 24000 - col("month_y")) / months).alias("josh_mody_n"),
                    col("ever_30"),
                    col("ever_90"),
                    col("ever_180"),
                    col("delinquency_30"),
                    col("delinquency_90"),
                    col("delinquency_180"),
                    col("loan_id"),
                    col("month_y"),
                    col("delinquency_12"),
                    col("upb_12")) \
            .groupBy("quarter", "loan_id", "josh_mody_n", "ever_30", "ever_90", "ever_180", "delinquency_30", "delinquency_90", "delinquency_180", "month_y") \
            .agg(max("delinquency_12").alias("delinquency_12"), min("upb_12").alias("upb_12")) \
            .withColumn("timestamp_year", floor((lit(24000) + (col("josh_mody_n") * lit(months)) + (col("month_y") - 1)) / lit(12))) \
            .selectExpr("*", "pmod(24000 + (josh_mody_n * {}) + month_y, 12) as timestamp_month_tmp".format(months)) \
            .withColumn("timestamp_month", when(col("timestamp_month_tmp") == lit(0), lit(12)).otherwise(col("timestamp_month_tmp"))) \
            .withColumn("delinquency_12", ((col("delinquency_12") > 3).cast("int") + (col("upb_12") == 0).cast("int")).alias("delinquency_12")) \
            .drop("timestamp_month_tmp", "josh_mody_n", "month_y")

    return perf.withColumnRenamed("monthly_reporting_period_month", "timestamp_month") \
            .withColumnRenamed("monthly_reporting_period_year", "timestamp_year") \
            .join(testDf, ["quarter", "loan_id", "timestamp_year", "timestamp_month"], "left") \
            .drop("timestamp_year", "timestamp_month")

def _create_acquisition(spark, acq):
    nameMapping = spark.createDataFrame(name_mapping, ["from_seller_name", "to_seller_name"])
    return acq.join(nameMapping, col("seller_name") == col("from_seller_name"), "left") \
      .drop("from_seller_name") \
      .withColumn("old_name", col("seller_name")) \
      .withColumn("seller_name", coalesce(col("to_seller_name"), col("seller_name"))) \
      .drop("to_seller_name") \
      .withColumn("orig_date", to_date(col("orig_date"), "MM/yyyy")) \
      .withColumn("first_pay_date", to_date(col("first_pay_date"), "MM/yyyy")) 

def _gen_dictionary(etl_df, col_names):
    cnt_table = etl_df.select(posexplode(array([col(i) for i in col_names])))\
                    .withColumnRenamed("pos", "column_id")\
                    .withColumnRenamed("col", "data")\
                    .filter("data is not null")\
                    .groupBy("column_id", "data")\
                    .count()
    windowed = Window.partitionBy("column_id").orderBy(desc("count"))
    return cnt_table.withColumn("id", row_number().over(windowed)).drop("count")

def _cast_string_columns_to_numeric(spark, input_df):
    cached_dict_df = _gen_dictionary(input_df, cate_col_names).cache()
    output_df = input_df
    #  Generate the final table with all columns being numeric.
    for col_pos, col_name in enumerate(cate_col_names):
        col_dict_df = cached_dict_df.filter(col("column_id") == col_pos)\
                                    .drop("column_id")\
                                    .withColumnRenamed("data", col_name)
        
        output_df = output_df.join(broadcast(col_dict_df), col_name, "left")\
                        .drop(col_name)\
                        .withColumnRenamed("id", col_name)
    return output_df  

def run_mortgage(spark, perf, acq):
    parsed_perf = _parse_dates(perf)
    perf_deliqency = _create_perf_deliquency(spark, parsed_perf)
    cleaned_acq = _create_acquisition(spark, acq)
    clean_df = perf_deliqency.join(cleaned_acq, ["loan_id", "quarter"], "inner").drop("quarter")
    casted_clean_df = _cast_string_columns_to_numeric(spark, clean_df)\
                    .select(all_col_names)\
                    .withColumn(label_col_name, when(col(label_col_name) > 0, 1).otherwise(0))\
                    .fillna(float(0))
    return casted_clean_df      
    
def extract_paths(paths, prefix):
    results = [ path[len(prefix):] for path in paths if path.startswith(prefix) ]
    if not results:
        print('-' * 80)
        print('Usage: {} data path required'.format(prefix))
        exit(1)
    return results

def etl(spark, args):
    rawDf = prepare_rawDf(spark, args)
    rawDf.write.parquet(extract_paths(args.dataPaths, 'tmp::')[0], mode='overwrite')
    rawDf = spark.read.parquet(extract_paths(args.dataPaths, 'tmp::')[0])

    acq = extract_acq_columns(rawDf)
    perf = extract_perf_columns(rawDf)
    out = run_mortgage(spark, perf, acq)
    return out
    
    
