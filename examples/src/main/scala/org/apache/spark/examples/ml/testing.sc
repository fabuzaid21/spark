import org.apache.spark.mllib.linalg.{Vectors, Vector, SparseVector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.VectorSlicer

import scala.util.Random

def loadData(
        sc: SparkContext,
        path: String,
        format: String,
        expectedNumFeatures: Option[Int] = None): RDD[LabeledPoint] = {
  format match {
    case "dense" => MLUtils.loadLabeledPoints(sc, path)
    case "libsvm" => expectedNumFeatures match {
      case Some(numFeatures) => MLUtils.loadLibSVMFile(sc, path, numFeatures)
      case None => MLUtils.loadLibSVMFile(sc, path)
    }
    case _ => throw new IllegalArgumentException(s"Bad data format: $format")
  }
}

def loadDatasets(
        sc: SparkContext,
        input: String,
        dataFormat: String,
        testInput: String,
        algo: String,
        fracTest: Double): (DataFrame, DataFrame) = {
  val sqlContext = new SQLContext(sc)
  import sqlContext.implicits._

  // Load training data
  val origExamples: RDD[LabeledPoint] = loadData(sc, input, dataFormat)

  // Load or create test set
  val splits: Array[RDD[LabeledPoint]] = if (testInput != "") {
    // Load testInput.
    val numFeatures = origExamples.take(1)(0).features.size
    val origTestExamples: RDD[LabeledPoint] =
      loadData(sc, testInput, dataFormat, Some(numFeatures))
    Array(origExamples, origTestExamples)
  } else {
    // Split input into training, test.
    origExamples.randomSplit(Array(1.0 - fracTest, fracTest), seed = 12345)
  }

  // For classification, convert labels to Strings since we will index them later with
  // StringIndexer.
  def labelsToStrings(data: DataFrame): DataFrame = {
    algo.toLowerCase match {
      case "classification" =>
        data.withColumn("labelString", data("label").cast(StringType))
      case "regression" =>
        data
      case _ =>
        throw new IllegalArgumentException("Algo ${params.algo} not supported.")
    }
  }
  val dataframes = splits.map(_.toDF()).map(labelsToStrings)
  val training = dataframes(0).cache()
  val test = dataframes(1).cache()

  val numTraining = training.count()
  val numTest = test.count()
  val numFeatures = training.select("features").first().getAs[Vector](0).size
  println("Loaded data:")
  println(s"  numTraining = $numTraining, numTest = $numTest")
  println(s"  numFeatures = $numFeatures")

  (training, test)
}

val (training, test) = loadDatasets(sc, "data/1700_500", "libsvm", "", "classification", 0.2)
val numFeatures = 300
val fracInitialTraining = 0.6
val fracIncrementalUpdate = 0.1

// import org.apache.spark.sql.functions._
// val toSparse = udf((v: Vector) => v.toSparse)
// val sparseTraining = training.withColumn("features", toSparse(training("features")))

val (initialStep, stepSize) = (numFeatures * fracInitialTraining,
  numFeatures * fracIncrementalUpdate)

val shuffledFeatureIndices = Random.shuffle(Range(0, numFeatures).iterator).toArray
val(firstSplit, rest) = shuffledFeatureIndices.splitAt(initialStep.toInt)
val remainingSplits = rest.sliding(stepSize.toInt, stepSize.toInt)
val indicesArr = Array(firstSplit) ++ remainingSplits
import org.apache.spark.sql.functions._

val trainSplits = indicesArr.map { indices =>
  //      val slicer = new VectorSlicer().setInputCol("features").setOutputCol("slicedFeatures")
  //      slicer.setIndices(indices)
  //      val output = slicer.transform(sparseTraining)
  val sliceToSparse = udf { (v: Vector) =>
    val vals = indices.map(v.apply)
    Vectors.sparse(indices.max + 1, indices, vals)
  }
  training.withColumn("features", sliceToSparse(training("features")))
}

// val splitWeights = Array(0.6, 0.1, 0.1, 0.1)
// val trainSplits = training.randomSplit(splitWeights)
// val slicer = new VectorSlicer().setInputCol("features").setOutputCol("reducedFeatures")
// slicer.setIndices(Array(0, 1))
// val output = slicer.transform(training)

