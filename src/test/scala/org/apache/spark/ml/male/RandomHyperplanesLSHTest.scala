package org.apache.spark.ml.male_hw8

import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer}
import org.apache.spark.ml.male.{RandomHyperplanesLSH}
import org.apache.spark.sql.SparkSession
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should

class RandomHyperplanesLSHTest extends AnyFlatSpec with should.Matchers {
  //  добавляю программно переменную среды в винду
  System.setProperty("hadoop.home.dir", "C:\\Program Files (x86)\\Hadoop")
  val spark = SparkSession.builder()
    .appName("Simaple App")
    .master("local[4]")
    .getOrCreate()

  val sqlc = spark.sqlContext
  val df = spark.read.option("inferSchema", "true").option("header", "true").csv("C:\\Users\\Xiaomi\\tripadvisor_hotel_reviews.csv")
  df.show(10)
  import sqlc.implicits._


  "RandomHyperplanesLSH" should "work" in {
    val preprocessingPipe = new Pipeline()
      .setStages(Array(
        new RegexTokenizer()
          .setInputCol("Review")
          .setOutputCol("tokenized")
          .setPattern("\\W+"),
        new HashingTF()
          .setInputCol("tokenized")
          .setOutputCol("tf")
          .setBinary(true)
          .setNumFeatures(1000),
        new HashingTF()
          .setInputCol("tokenized")
          .setOutputCol("tf2")
          .setNumFeatures(1000),
        new IDF()
          .setInputCol("tf2")
          .setOutputCol("tfidf")
      ))

    val Array(train, test) = df.randomSplit(Array(0.8, 0.2))


    val pipe = preprocessingPipe.fit(train)

    val trainFeatures = pipe.transform(train).cache()
    val testFeatures = pipe.transform(test)
    val testFeaturesWithIndex = testFeatures.withColumn("id", monotonicallyIncreasingId()).cache()
    testFeaturesWithIndex.show(10)

    val mh = new RandomHyperplanesLSH()
      .setInputCol("tfidf")
      .setOutputCol("brpBuckets")
      .setNumHashTables(3)

//    val mh = new BucketedRandomProjectionLSH()
//      .setInputCol("tfidf")
//      .setOutputCol("brpBuckets")
//      .setBucketLength(5)
//      .setNumHashTables(3)

    val mhModel = mh.fit(trainFeatures)

    val euqlidNeigh = mhModel.approxSimilarityJoin(trainFeatures, testFeaturesWithIndex, 0.6)

    euqlidNeigh.show

    val predictions = euqlidNeigh
      .withColumn("similarity", (lit(1) / (col("distCol") + lit(0.0000001))))
      .groupBy("datasetB.id")
      .agg((sum(col("similarity") * col("datasetA.Rating")) / sum(col("similarity"))).as("predict"))

    val forMetric = testFeaturesWithIndex.join(predictions, Seq("id"))
    val metrics = new RegressionEvaluator()
      .setLabelCol("Rating")
      .setPredictionCol("predict")
      .setMetricName("rmse")
    println(metrics.evaluate(forMetric))

  }


}
