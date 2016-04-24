package org.apache.spark.ml.tree.impl

import org.apache.spark.ml.tree.impl.AltDT.AltDTMetadata
import org.apache.spark.ml.tree.{CategoricalSplit, ContinuousSplit, ImpurityAggregatorSingle, Split}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.tree.model.ImpurityStats
import org.apache.spark.util.collection.{SortDataFormat, Sorter}

import scala.collection.mutable.ArrayBuffer

/**
 * Feature vector types are based on (feature type, representation).
 * The feature type can be continuous or categorical.
 *
 * Features are sorted by value, so we must store indices + values.
 * These values are currently stored in a dense representation only.
 * TODO: Support sparse storage (to optimize deeper levels of the tree), and maybe compressed
 *       storage (to optimize upper levels of the tree).
 */
sealed trait FeatureVector extends Serializable {

  val featureIndex: Int
  var values: Array[Double]
  val indices: Array[Int]
  val metadata: AltDTMetadata

  def isCategorical: Boolean

  def isOrdered: Boolean

  /**
   * Choose the best split for a feature at a node.
   * TODO: Return null or None when the split is invalid, such as putting all instances on one
   *       child node.
   *
   * @return  (best split, statistics for split)  If the best split actually puts all instances
   *          in one leaf node, then it will be set to None.
   */
  def chooseSplit(from: Int, to: Int, labels: Array[Double]): (Option[Split], ImpurityStats)

  /**
   * Only called AFTER values have already sorted
   */
  def compress(isSparse: Boolean = false)

  def deepCopy(): FeatureVector

  def toString: String

  def equals(other: Any): Boolean

}

final class ContinuousFeatureVector(
                               override val featureIndex: Int,
                               override var values: Array[Double],
                               override val indices: Array[Int],
                               override val metadata: AltDTMetadata) extends FeatureVector {

  override def isCategorical: Boolean = false

  override def isOrdered: Boolean = false

  /**
   * Choose splitting rule: feature value <= threshold
   * @return  (best split, statistics for split)  If the best split actually puts all instances
   *          in one leaf node, then it will be set to None.  The impurity stats maybe still be
   *          useful, so they are returned.
   */
  override def chooseSplit(from: Int, to: Int, labels: Array[Double]): (Option[Split], ImpurityStats) = {
    val leftImpurityAgg = metadata.createImpurityAggregator()
    val rightImpurityAgg = metadata.createImpurityAggregator()
    var i = from
    while (i < to) {
      rightImpurityAgg.update(labels(indices(i)), 1.0)  // can we pass each label once with a weight equal to count?
      i += 1
    }

    var bestThreshold: Double = Double.NegativeInfinity
    val bestLeftImpurityAgg = leftImpurityAgg.deepCopy()
    var bestGain: Double = 0.0
    val fullImpurity = rightImpurityAgg.getCalculator.calculate()
    var leftCount: Double = 0.0
    var rightCount: Double = rightImpurityAgg.getCount
    val fullCount: Double = rightCount
    var currentThreshold = values.headOption.getOrElse(bestThreshold)
    var j = from
    while (j < to) {
      val value = values(j)
      val label = labels(indices(j))
      if (value != currentThreshold) {
        // Check gain
        val leftWeight = leftCount / fullCount
        val rightWeight = rightCount / fullCount
        val leftImpurity = leftImpurityAgg.getCalculator.calculate()
        val rightImpurity = rightImpurityAgg.getCalculator.calculate()
        val gain = fullImpurity - leftWeight * leftImpurity - rightWeight * rightImpurity
        if (leftCount != 0 && rightCount != 0 && gain > bestGain && gain > metadata.minInfoGain) {
          bestThreshold = currentThreshold
          System.arraycopy(leftImpurityAgg.stats, 0, bestLeftImpurityAgg.stats, 0, leftImpurityAgg.stats.length)
          bestGain = gain
        }
        currentThreshold = value
      }
      // Move this instance from right to left side of split.
      leftImpurityAgg.update(label, 1.0)
      rightImpurityAgg.update(label, -1.0)
      leftCount += 1.0
      rightCount -= 1.0
      j += 1
    }

    val fullImpurityAgg = leftImpurityAgg.deepCopy().add(rightImpurityAgg)
    val bestRightImpurityAgg = fullImpurityAgg.deepCopy().subtract(bestLeftImpurityAgg)
    val bestImpurityStats = new ImpurityStats(bestGain, fullImpurity, fullImpurityAgg.getCalculator,
      bestLeftImpurityAgg.getCalculator, bestRightImpurityAgg.getCalculator)
    val split = if (bestThreshold != Double.NegativeInfinity && bestThreshold != values.last) {
      Some(new ContinuousSplit(featureIndex, bestThreshold))
    } else {
      None
    }
    (split, bestImpurityStats)
  }

  protected var compressedRLEVals: Array[(Double, Int)] = _
  protected var compressedDeltaVals: Array[Double] = _

  override def compress(isSparse: Boolean = false) = {
    if (isSparse) {
      compressedRLEVals = FeatureVector.runLengthEncoding(values)
    } else {
      compressedDeltaVals = FeatureVector.deltaEncoding(values)
    }
//    values = null
  }

  def deepCopy(): FeatureVector =
    new ContinuousFeatureVector(featureIndex, values.clone(), indices.clone(), metadata)

  /** For debugging */
  override def toString: String = {
    "  ContinuousFeatureVector(" +
      s"    featureIndex: $featureIndex,\n" +
      s"    values: ${values.mkString(", ")},\n" +
      s"    indices: ${indices.mkString(", ")},\n" +
      "  )"
  }

  override def equals(other: Any): Boolean = {
    other match {
      case o: ContinuousFeatureVector =>
        featureIndex == o.featureIndex &&
          values.sameElements(o.values) && indices.sameElements(o.indices)
      case _ => false
    }
  }
}

abstract class CategoricalFeatureVector(val featureArity: Int) extends FeatureVector {
  require(featureArity > 0)
  def isCategorical: Boolean = true

  protected var compressedVals: Array[(Double, Int)] = _

  // Run-length encoding
  override def compress(isSparse: Boolean = false) = {
    compressedVals = FeatureVector.runLengthEncoding(values)
//    values = null
  }
}

final class OrderedCategoricalFeatureVector(
                                             override val featureIndex: Int,
                                             override val featureArity: Int,
                                             override var values: Array[Double],
                                             override val indices: Array[Int],
                                             override val metadata: AltDTMetadata)
  extends CategoricalFeatureVector(featureArity) {
  require(!metadata.isUnorderedFeature(featureIndex))

  override def isOrdered: Boolean = true

  /**
   * Find the best split for an ordered categorical feature at a single node.
   *
   * Algorithm:
   *  - For each category, compute a "centroid."
   *     - For multiclass classification, the centroid is the label impurity.
   *     - For binary classification and regression, the centroid is the average label.
   *  - Sort the centroids, and consider splits anywhere in this order.
   *    Thus, with K categories, we consider K - 1 possible splits.
   *
   * @param labels  Labels corresponding to values, in the same order.
   * @return  (best split, statistics for split)  If the best split actually puts all instances
   *          in one leaf node, then it will be set to None.  The impurity stats maybe still be
   *          useful, so they are returned.
   */
  override def chooseSplit(from: Int, to: Int, labels: Array[Double]): (Option[Split], ImpurityStats) = {
    // TODO: Support high-arity features by using a single array to hold the stats.

    // aggStats(category) = label statistics for category
    val aggStats = Array.tabulate[ImpurityAggregatorSingle](featureArity)(
      _ => metadata.createImpurityAggregator())
    var i = from
    while (i < to) {
      val cat = values(i)
      val label = labels(indices(i))
      aggStats(cat.toInt).update(label)
      i += 1
    }

    // Compute centroids.  centroidsForCategories is a list: (category, centroid)
    val centroidsForCategories: Seq[(Int, Double)] = if (metadata.isMulticlass) {
      // For categorical variables in multiclass classification,
      // the bins are ordered by the impurity of their corresponding labels.
      Range(0, featureArity).map { case featureValue =>
        val categoryStats = aggStats(featureValue)
        val centroid = if (categoryStats.getCount != 0) {
          categoryStats.getCalculator.calculate()
        } else {
          Double.MaxValue
        }
        (featureValue, centroid)
      }
    } else if (metadata.isClassification) { // binary classification
      // For categorical variables in binary classification,
      // the bins are ordered by the centroid of their corresponding labels.
      Range(0, featureArity).map { case featureValue =>
        val categoryStats = aggStats(featureValue)
        val centroid = if (categoryStats.getCount != 0) {
          assert(categoryStats.stats.length == 2)
          (categoryStats.stats(1) - categoryStats.stats(0)) / categoryStats.getCount
        } else {
          Double.MaxValue
        }
        (featureValue, centroid)
      }
    } else { // regression
      // For categorical variables in regression,
      // the bins are ordered by the centroid of their corresponding labels.
      Range(0, featureArity).map { case featureValue =>
        val categoryStats = aggStats(featureValue)
        val centroid = if (categoryStats.getCount != 0) {
          categoryStats.getCalculator.predict
        } else {
          Double.MaxValue
        }
        (featureValue, centroid)
      }
    }

//    logDebug("Centroids for categorical variable: " + centroidsForCategories.mkString(","))

    val categoriesSortedByCentroid: List[Int] = centroidsForCategories.toList.sortBy(_._2).map(_._1)

    // Cumulative sums of bin statistics for left, right parts of split.
    val leftImpurityAgg = metadata.createImpurityAggregator()
    val rightImpurityAgg = metadata.createImpurityAggregator()
    var j = 0
    val length = aggStats.length
    while (j < length) {
      rightImpurityAgg.add(aggStats(j))
      j += 1
    }

    var bestSplitIndex: Int = -1  // index into categoriesSortedByCentroid
    val bestLeftImpurityAgg = leftImpurityAgg.deepCopy()
    var bestGain: Double = 0.0
    val fullImpurity = rightImpurityAgg.getCalculator.calculate()
    var leftCount: Double = 0.0
    var rightCount: Double = rightImpurityAgg.getCount
    val fullCount: Double = rightCount

    // Consider all splits. These only cover valid splits, with at least one category on each side.
    val numSplits = categoriesSortedByCentroid.length - 1
    var sortedCatIndex = 0
    while (sortedCatIndex < numSplits) {
      val cat = categoriesSortedByCentroid(sortedCatIndex)
      // Update left, right stats
      val catStats = aggStats(cat)
      leftImpurityAgg.add(catStats)
      rightImpurityAgg.subtract(catStats)
      leftCount += catStats.getCount
      rightCount -= catStats.getCount
      // Compute impurity
      val leftWeight = leftCount / fullCount
      val rightWeight = rightCount / fullCount
      val leftImpurity = leftImpurityAgg.getCalculator.calculate()
      val rightImpurity = rightImpurityAgg.getCalculator.calculate()
      val gain = fullImpurity - leftWeight * leftImpurity - rightWeight * rightImpurity
      if (leftCount != 0 && rightCount != 0 && gain > bestGain && gain > metadata.minInfoGain) {
        bestSplitIndex = sortedCatIndex
        System.arraycopy(leftImpurityAgg.stats, 0, bestLeftImpurityAgg.stats, 0, leftImpurityAgg.stats.length)
        bestGain = gain
      }
      sortedCatIndex += 1
    }

    val categoriesForSplit =
      categoriesSortedByCentroid.slice(0, bestSplitIndex + 1).map(_.toDouble)
    val bestFeatureSplit =
      new CategoricalSplit(featureIndex, categoriesForSplit.toArray, featureArity)
    val fullImpurityAgg = leftImpurityAgg.deepCopy().add(rightImpurityAgg)
    val bestRightImpurityAgg = fullImpurityAgg.deepCopy().subtract(bestLeftImpurityAgg)
    val bestImpurityStats = new ImpurityStats(bestGain, fullImpurity, fullImpurityAgg.getCalculator,
      bestLeftImpurityAgg.getCalculator, bestRightImpurityAgg.getCalculator)

    if (bestSplitIndex == -1 || bestGain == 0.0) {
      (None, bestImpurityStats)
    } else {
      (Some(bestFeatureSplit), bestImpurityStats)
    }
  }

  override def deepCopy(): FeatureVector =
    new OrderedCategoricalFeatureVector(featureIndex, featureArity, values.clone(), indices.clone(), metadata)

    /** For debugging */
    override def toString: String = {
      "  OrderedCategoricalFeatureVector(" +
        s"    featureIndex: $featureIndex,\n" +
        s"    featureArity: $featureArity,\n" +
        s"    values: ${values.mkString(", ")},\n" +
        s"    indices: ${indices.mkString(", ")},\n" +
        "  )"
    }

    override def equals(other: Any): Boolean = {
      other match {
        case o: OrderedCategoricalFeatureVector =>
          featureIndex == o.featureIndex && featureArity == o.featureArity &&
            values.sameElements(o.values) && indices.sameElements(o.indices)
        case _ => false
      }
    }
}

final class UnorderedCategoricalFeatureVector(
                                             override val featureIndex: Int,
                                             override val featureArity: Int,
                                             override var values: Array[Double],
                                             override val indices: Array[Int],
                                             override val metadata: AltDTMetadata)
  extends CategoricalFeatureVector(featureArity) {
  require(metadata.isUnorderedFeature(featureIndex))

  private val splits = metadata.getUnorderedSplits(featureIndex)

  override def isOrdered: Boolean = false

  /**
   * Find the best split for an unordered categorical feature at a single node.
   *
   * Algorithm:
   *  - Considers all possible subsets (exponentially many)
   *
   * @param from  start index of sub-array for the data that belongs to the node
   * @param to  end index of sub-array for the data that belongs to the node
   * @param labels  Labels corresponding to values, in the same order.
   * @return  (best split, statistics for split)  If the best split actually puts all instances
   *          in one leaf node, then it will be set to None.  The impurity stats maybe still be
   *          useful, so they are returned.
   */
  override def chooseSplit(from: Int, to: Int, labels: Array[Double]): (Option[Split], ImpurityStats) = {
    // Label stats for each category
    val aggStats = Array.tabulate[ImpurityAggregatorSingle](featureArity)(
      _ => metadata.createImpurityAggregator())
    var i = from
    while (i < to) {
      val cat = values(i)
      val label = labels(indices(i))
      // NOTE: we assume the values for categorical features are Ints in [0,featureArity)
      aggStats(cat.toInt).update(label)
      i += 1
    }

    // Aggregated statistics for left part of split and entire split.
    val leftImpurityAgg = metadata.createImpurityAggregator()
    val fullImpurityAgg = metadata.createImpurityAggregator()
    aggStats.foreach(fullImpurityAgg.add)
    val fullImpurity = fullImpurityAgg.getCalculator.calculate()

    if (featureArity == 1) {
      // All instances go right
      val impurityStats = new ImpurityStats(0.0, fullImpurityAgg.getCalculator.calculate(),
        fullImpurityAgg.getCalculator, leftImpurityAgg.getCalculator,
        fullImpurityAgg.getCalculator)
      (None, impurityStats)
    } else {
      //  TODO: We currently add and remove the stats for all categories for each split.
      //  A better way to do it would be to consider splits in an order such that each iteration
      //  only requires addition/removal of a single category and a single add/subtract to
      //  leftCount and rightCount.
      //  TODO: Use more efficient encoding such as gray codes
      var bestSplit: Option[CategoricalSplit] = None
      val bestLeftImpurityAgg = leftImpurityAgg.deepCopy()
      var bestGain: Double = -1.0
      val fullCount: Double = to - from
      for (split <- splits) {
        // Update left, right impurity stats
        split.leftCategories.foreach(c => leftImpurityAgg.add(aggStats(c.toInt)))
        val rightImpurityAgg = fullImpurityAgg.deepCopy().subtract(leftImpurityAgg)
        val leftCount = leftImpurityAgg.getCount
        val rightCount = rightImpurityAgg.getCount
        // Compute impurity
        val leftWeight = leftCount / fullCount
        val rightWeight = rightCount / fullCount
        val leftImpurity = leftImpurityAgg.getCalculator.calculate()
        val rightImpurity = rightImpurityAgg.getCalculator.calculate()
        val gain = fullImpurity - leftWeight * leftImpurity - rightWeight * rightImpurity
        if (leftCount != 0 && rightCount != 0 && gain > bestGain && gain > metadata.minInfoGain) {
          bestSplit = Some(split)
          System.arraycopy(leftImpurityAgg.stats, 0, bestLeftImpurityAgg.stats, 0, leftImpurityAgg.stats.length)
          bestGain = gain
        }
        // Reset left impurity stats
        leftImpurityAgg.clear()
      }

      val bestFeatureSplit = bestSplit match {
        case Some(split) => Some(
          new CategoricalSplit(featureIndex, split.leftCategories, featureArity))
        case None => None

      }
      val bestRightImpurityAgg = fullImpurityAgg.deepCopy().subtract(bestLeftImpurityAgg)
      val bestImpurityStats = new ImpurityStats(bestGain, fullImpurity,
        fullImpurityAgg.getCalculator, bestLeftImpurityAgg.getCalculator,
        bestRightImpurityAgg.getCalculator)
      (bestFeatureSplit, bestImpurityStats)
    }
  }

  override def deepCopy(): FeatureVector =
    new UnorderedCategoricalFeatureVector(featureIndex, featureArity, values.clone(), indices.clone(), metadata)

  /** For debugging */
  override def toString: String = {
    "  UnorderedCategoricalFeatureVector(" +
      s"    featureIndex: $featureIndex,\n" +
      s"    featureArity: $featureArity,\n" +
      s"    values: ${values.mkString(", ")},\n" +
      s"    indices: ${indices.mkString(", ")},\n" +
      "  )"
  }

  override def equals(other: Any): Boolean = {
    other match {
      case o: UnorderedCategoricalFeatureVector =>
        featureIndex == o.featureIndex && featureArity == o.featureArity &&
          values.sameElements(o.values) && indices.sameElements(o.indices)
      case _ => false
    }
  }
}

object FeatureVector {

  def deltaEncoding(values: Array[Double]): Array[Double] = {
    val deltaEncoded = new ArrayBuffer[Double]()
    var prev = values(0)
    deltaEncoded += prev
    var i = 1
    while (i < values.length) {
      val next = values(i)
      deltaEncoded += (next - prev)
      prev = next
      i += 1
    }
    deltaEncoded.toArray
  }

  def runLengthEncoding(values: Array[Double]): Array[(Double, Int)] = {
    val rle = new ArrayBuffer[(Double, Int)]()
    var currVal = values(0)
    var i = 1
    var count = 1
    while (i < values.length) {
      val nextVal = values(i)
      if (nextVal != currVal) {
        rle += ((currVal.toInt, count))
        currVal = nextVal
        count = 1
      } else {
        count += 1
      }
      i += 1
    }
    // add the last one
    rle += ((currVal.toInt, count))
    rle.toArray
  }

  // create unsorted FeatureVector
  def createUnsortedVector(featureIndex: Int, featureArity: Int, values: Array[Double],
                           indices: Array[Int], metadata: AltDTMetadata): FeatureVector = {
    val isOrdered = !metadata.isUnorderedFeature(featureIndex)
    (featureArity, isOrdered) match {
      case (0, _) => new ContinuousFeatureVector(featureIndex, values, indices, metadata)
      case (_, true) => new OrderedCategoricalFeatureVector(featureIndex, featureArity, values, indices, metadata)
      case (_, false) => new UnorderedCategoricalFeatureVector(featureIndex, featureArity, values, indices, metadata)
    }
  }

  def createUnsortedVector(featureIndex: Int, featureArity: Int, vector: Vector, metadata: AltDTMetadata): FeatureVector = {
    val values = vector.toArray
    val indices = values.indices.toArray
    createUnsortedVector(featureIndex, featureArity, values, indices, metadata)
  }

  /** Store column sorted by feature values. */
  def fromOriginal(
                    featureIndex: Int,
                    featureArity: Int,
                    vector: Vector,
                    metadata: AltDTMetadata): FeatureVector = {

    val fv = createUnsortedVector(featureIndex, featureArity, vector, metadata)
    val sorter = new Sorter(new FeatureVectorSortByValue(featureIndex, featureArity, metadata))
    // don't pass in fv.size, because that could be larger than fv.values.length (if the vector is sparse)
    sorter.sort(fv, 0, fv.values.length, Ordering[KeyWrapper[Double]])

    if (!fv.isCategorical) {
      /**
       * boolean argument to [[FeatureVector.compress]] only matters for
       * continuous vectors; boolean check borrowed from [[Vector.compressed]]
       */
      fv.compress(1.5 * (vector.numNonzeros + 1.0) < vector.size)
    } else {
      fv.compress()
    }
    fv
  }
}

/**
 * Sort FeatureVector by values column; @see [[FeatureVector.fromOriginal()]]
 * @param featureIndex @param featureArity Passed in so that, if a new
 *                     FeatureVector is allocated during sorting, that new object
 *                     also has the same featureIndex and featureArity
 */
private class FeatureVectorSortByValue(featureIndex: Int, featureArity: Int,
                                       metadata: AltDTMetadata)(implicit ord: Ordering[Double])
  extends SortDataFormat[KeyWrapper[Double], FeatureVector] {

  override def newKey(): KeyWrapper[Double] = new KeyWrapper()

  override def getKey(data: FeatureVector,
                      pos: Int,
                      reuse: KeyWrapper[Double]): KeyWrapper[Double] = {
    if (reuse == null) {
      new KeyWrapper().setKey(data.values(pos))
    } else {
      reuse.setKey(data.values(pos))
    }
  }

  override def getKey(data: FeatureVector,
                      pos: Int): KeyWrapper[Double] = {
    getKey(data, pos, null)
  }

  private def swapElements[T](data: Array[T],
                              pos0: Int,
                              pos1: Int): Unit = {
    val tmp = data(pos0)
    data(pos0) = data(pos1)
    data(pos1) = tmp
  }

  override def swap(data: FeatureVector, pos0: Int, pos1: Int): Unit = {
    swapElements(data.values, pos0, pos1)
    swapElements(data.indices, pos0, pos1)
  }

  override def copyRange(src: FeatureVector,
                         srcPos: Int,
                         dst: FeatureVector,
                         dstPos: Int,
                         length: Int): Unit = {
    System.arraycopy(src.values, srcPos, dst.values, dstPos, length)
    System.arraycopy(src.indices, srcPos, dst.indices, dstPos, length)
  }

  override def allocate(length: Int): FeatureVector = {
    FeatureVector.createUnsortedVector(featureIndex, featureArity, new Array[Double](length),
      new Array[Int](length), metadata)
  }

  override def copyElement(src: FeatureVector,
                           srcPos: Int,
                           dst: FeatureVector,
                           dstPos: Int): Unit = {
    dst.values(dstPos) = src.values(srcPos)
    dst.indices(dstPos) = src.indices(srcPos)
  }
}

/**
 * A wrapper that holds a primitive key – borrowed from [[org.apache.spark.ml.recommendation.ALS.KeyWrapper]]
 */
private class KeyWrapper[Double](implicit ord: Ordering[Double])
  extends Ordered[KeyWrapper[Double]] {

  var key: Double = _

  override def compare(that: KeyWrapper[Double]): Int = {
    ord.compare(key, that.key)
  }

  def setKey(key: Double): this.type = {
    this.key = key
    this
  }
}

