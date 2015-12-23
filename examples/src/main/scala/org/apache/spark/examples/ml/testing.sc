import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

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

val (training, test) = loadDatasets(sc, "data/10K_2K", "libsvm", "", "classification", 0.2)


