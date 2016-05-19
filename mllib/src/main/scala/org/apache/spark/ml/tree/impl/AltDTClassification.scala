package org.apache.spark.ml.tree.impl

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.AltDT.{AltDTMetadata, FeatureVector, PartitionInfo}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.ImpurityStats
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

object AltDTClassification {

  def trainImpl(
                 input: RDD[LabeledPoint],
                 colStoreInit: RDD[(Int, Array[Double])],
                 metadata: AltDTMetadata,
                 numRows: Int,
                 maxDepth: Int): Node = {

    val labels = new Array[Byte](numRows)
    input.map(_.label).zipWithIndex().collect().foreach { case (label: Double, rowIndex: Long) =>
      labels(rowIndex.toInt) = label.toByte
    }
    val labelsBc = input.sparkContext.broadcast(labels)
    // NOTE: Labels are not sorted with features since that would require 1 copy per feature,
    //       rather than 1 copy per worker. This means a lot of random accesses.
    //       We could improve this by applying first-level sorting (by node) to labels.

    // Sort each column by feature values.
    val colStore: RDD[FeatureVector] = colStoreInit.map { case (featureIndex, col) =>
      val featureArity: Int = metadata.categoricalFeaturesInfo.getOrElse(featureIndex, 0)
      if (featureArity > 0) {
        // aggStats(category) = label statistics for category
        val aggStats = FeatureVector.initAggStats(col, labelsBc.value, featureArity, metadata)
        FeatureVector.fromOriginal(featureIndex, featureArity, col, Array(aggStats))
      } else {
        FeatureVector.fromOriginal(featureIndex, featureArity, col)
      }
    }
    // Group columns together into one array of columns per partition.
    // TODO: Test avoiding this grouping, and see if it matters.
    val groupedColStore: RDD[Array[FeatureVector]] = colStore.mapPartitions {
      iterator: Iterator[FeatureVector] =>
        if (iterator.nonEmpty) Iterator(iterator.toArray) else Iterator()
    }
    groupedColStore.persist(StorageLevel.MEMORY_AND_DISK)

    // Initialize partitions with 1 node (each instance at the root node).
    val fullImpurityAgg = metadata.createImpurityAggregator()
    var i = 0
    while (i < labels.length) {
      fullImpurityAgg.update(labels(i))
      i += 1
    }

    var partitionInfos: RDD[PartitionInfo] = groupedColStore.map(new PartitionInfo(_, Array(fullImpurityAgg)))

    // Initialize model.
    // Note: We do not use node indices.
    var numActiveNodes = 1
    val rootNode = LearningNode.emptyNode(1) // TODO: remove node id
    // Active nodes (still being split), updated each iteration
    var activeNodePeriphery: Array[LearningNode] = Array(rootNode)

    // Iteratively learn, one level of the tree at a time.
    var currentLevel = 0
    var doneLearning = false
    while (currentLevel < maxDepth && !doneLearning) {
      // Compute best split for each active node.
      val bestSplitsAndGains = computeBestSplits(partitionInfos, numActiveNodes, labelsBc, metadata)

      // Update current model and node periphery.
      // Note: This flatMap has side effects (on the model).
      activeNodePeriphery =
        AltDT.computeActiveNodePeriphery(activeNodePeriphery, bestSplitsAndGains, metadata.minInfoGain)
      val activeNodeMap = AltDT.computeActiveNodeMap(bestSplitsAndGains, metadata.minInfoGain)
      assert(activeNodeMap.size == activeNodePeriphery.length)

      // Filter active node periphery by impurity.
      val estimatedRemainingActive = activeNodePeriphery.count(_.stats.impurity > 0.0)

      doneLearning = currentLevel + 1 >= maxDepth || estimatedRemainingActive == 0
      if (!doneLearning) {
        val splits: Array[Option[Split]] = bestSplitsAndGains.map(_._1)
        // Aggregate bit vector (1 bit/instance) indicating whether each instance goes left/right
        val aggBitVector = AltDT.aggregateBitVector(partitionInfos, splits)

        val newPartitionInfos = partitionInfos.map { partitionInfo =>
          partitionInfo.update(aggBitVector, activeNodeMap, labelsBc.value, metadata)
        }

        newPartitionInfos.cache().count()
        partitionInfos = newPartitionInfos
        numActiveNodes = activeNodeMap.size
      }
      currentLevel += 1
    }

    // Done with learning
    groupedColStore.unpersist()
    labelsBc.unpersist()
    rootNode.toNode
  }

  /**
   * Find the best splits for all active nodes.
   * - On each partition, for each feature on the partition, select the best split for each node.
   * Each worker returns: For each active node, best split + info gain
   * - The splits across workers are aggregated to the driver.
   * @return  Array over active nodes of (best split, impurity stats for split),
   *          where the split is None if no useful split exists
   */
  private[impl] def computeBestSplits(
                                       partitionInfos: RDD[PartitionInfo],
                                       numActiveNodes: Int,
                                       labelsBc: Broadcast[Array[Byte]],
                                       metadata: AltDTMetadata) = {
    // On each partition, for each feature on the partition, select the best split for each node.
    // This will use:
    //  - groupedColStore (the features)
    //  - partitionInfos (the node -> instance mapping)
    //  - labelsBc (the labels column)
    // Each worker returns:
    //   for each active node, best split + info gain,
    //     where the best split is None if no useful split exists
    val partBestSplitsAndGains: RDD[Array[(Option[Split], ImpurityStats)]] = partitionInfos.map {
      case PartitionInfo(columns: Array[FeatureVector], fullImpurityAggs: Array[ImpurityAggregatorSingle]) =>
        val localLabels = labelsBc.value
        val toReturn = chooseSplitsForActiveNodes(columns(0), numActiveNodes, localLabels, fullImpurityAggs, metadata)
        columns.drop(1).foreach { col =>
          val splitsForCol = chooseSplitsForActiveNodes(col, numActiveNodes, localLabels, fullImpurityAggs, metadata)
          splitsForCol.zipWithIndex.foreach { case (splitAndStats, idx) =>
            if (splitAndStats._2.gain > toReturn(idx)._2.gain) {
              toReturn(idx) = splitAndStats
            }
          }
        }
        toReturn
    }

    // Aggregate best split for each active node.
    partBestSplitsAndGains.treeReduce { case (splitsGains1, splitsGains2) =>
      splitsGains1.zip(splitsGains2).map { case ((split1, gain1), (split2, gain2)) =>
        if (gain1.gain >= gain2.gain) {
          (split1, gain1)
        } else {
          (split2, gain2)
        }
      }
    }
  }

  /**
   * Choose the best split for a feature at a node.
   * TODO: Return null or None when the split is invalid, such as putting all instances on one
   * child node.
   *
   * @return  (best split, statistics for split)  If the best split actually puts all instances
   *          in one leaf node, then it will be set to None.
   */
  private[impl] def chooseSplitsForActiveNodes(
                                                col: FeatureVector,
                                                numActiveNodes: Int,
                                                labels: Array[Byte],
                                                fullImpurityAggs: Array[ImpurityAggregatorSingle],
                                                metadata: AltDTMetadata): Array[(Option[Split], ImpurityStats)] = {
    if (col.isCategorical) {
      if (metadata.isUnorderedFeature(col.featureIndex)) {
        val splits: Array[CategoricalSplit] = metadata.getUnorderedSplits(col.featureIndex)
        Range(0, numActiveNodes).toArray.map { nodeIndex =>
          chooseUnorderedCategoricalSplit(col.featureIndex, col.aggStats(nodeIndex), fullImpurityAggs(nodeIndex),
            metadata, col.featureArity, splits)
        }
      } else {
        Range(0, numActiveNodes).toArray.map { nodeIndex =>
          chooseOrderedCategoricalSplit(col.featureIndex, col.aggStats(nodeIndex), fullImpurityAggs(nodeIndex),
            metadata, col.featureArity)
        }
      }
    } else {
      if (col.sparse) {
        chooseContinuousSplitsSparseForActiveNodes(col.featureIndex, numActiveNodes, col.compressedVals,
          col.indices, labels, col.nodeIndices, fullImpurityAggs, metadata)
      } else {
        chooseContinuousSplitsForActiveNodes(col.featureIndex, numActiveNodes, col.values, col.indices, labels,
          col.nodeIndices, fullImpurityAggs, metadata)
      }
    }
  }

  /**
   * Find the best split for an ordered categorical feature at a single node.
   *
   * Algorithm:
   * - For each category, compute a "centroid."
   * - For multiclass classification, the centroid is the label impurity.
   * - For binary classification and regression, the centroid is the average label.
   * - Sort the centroids, and consider splits anywhere in this order.
   * Thus, with K categories, we consider K - 1 possible splits.
   *
   * @param featureIndex  Index of feature being split.
   * @return  (best split, statistics for split)  If the best split actually puts all instances
   *          in one leaf node, then it will be set to None.  The impurity stats maybe still be
   *          useful, so they are returned.
   */
  // TODO: Support high-arity features by using a single array to hold the stats.
  private[impl] def chooseOrderedCategoricalSplit(
                                                   featureIndex: Int,
                                                   // aggStats(category) = label statistics for category
                                                   aggStats: Array[ImpurityAggregatorSingle],
                                                   fullImpurityAgg: ImpurityAggregatorSingle,
                                                   metadata: AltDTMetadata,
                                                   featureArity: Int): (Option[Split], ImpurityStats) = {

    // Compute centroids.  centroidsForCategories is a list: (category, centroid)
    val centroidsForCategories = if (metadata.isMulticlass) {
      // For categorical variables in multiclass classification,
      // the bins are ordered by the impurity of their corresponding labels.
      Range(0, featureArity).toArray.map { case featureValue =>
        val categoryStats = aggStats(featureValue)
        val centroid = if (categoryStats.getCount != 0) {
          categoryStats.getCalculator.calculate()
        } else {
          Double.MaxValue
        }
        (featureValue, centroid)
      }
    } else if (metadata.isClassification) {
      // binary classification
      // For categorical variables in binary classification,
      // the bins are ordered by the centroid of their corresponding labels.
      Range(0, featureArity).toArray.map { case featureValue =>
        val categoryStats = aggStats(featureValue)
        val centroid = if (categoryStats.getCount != 0) {
          assert(categoryStats.stats.length == 2)
          (categoryStats.stats(1) - categoryStats.stats(0)) / categoryStats.getCount
        } else {
          Double.MaxValue
        }
        (featureValue, centroid)
      }
    } else {
      // regression
      // For categorical variables in regression,
      // the bins are ordered by the centroid of their corresponding labels.
      Range(0, featureArity).toArray.map { case featureValue =>
        val categoryStats = aggStats(featureValue)
        val centroid = if (categoryStats.getCount != 0) {
          categoryStats.getCalculator.predict
        } else {
          Double.MaxValue
        }
        (featureValue, centroid)
      }
    }

    val categoriesSortedByCentroid = centroidsForCategories.sortBy(_._2).map(_._1)

    // Cumulative sums of bin statistics for left, right parts of split.
    val leftImpurityAgg = metadata.createImpurityAggregator()
    val rightImpurityAgg = metadata.createImpurityAggregator()
    var j = 0
    val length = aggStats.length
    while (j < length) {
      rightImpurityAgg.add(aggStats(j))
      j += 1
    }

    var bestSplitIndex: Int = -1 // index into categoriesSortedByCentroid
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

  /**
   * Find the best split for an unordered categorical feature at a single node.
   *
   * Algorithm:
   * - Considers all possible subsets (exponentially many)
   *
   * @param featureIndex  Index of feature being split.
   * @return  (best split, statistics for split)  If the best split actually puts all instances
   *          in one leaf node, then it will be set to None.  The impurity stats maybe still be
   *          useful, so they are returned.
   */
  private[impl] def chooseUnorderedCategoricalSplit(
                                                     featureIndex: Int,
                                                     // aggStats(category) = label statistics for category
                                                     aggStats: Array[ImpurityAggregatorSingle],
                                                     fullImpurityAgg: ImpurityAggregatorSingle,
                                                     metadata: AltDTMetadata,
                                                     featureArity: Int,
                                                     splits: Array[CategoricalSplit]): (Option[Split], ImpurityStats) = {

    // Aggregated statistics for left part of split and entire split.
    val leftImpurityAgg = metadata.createImpurityAggregator()
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
      val fullCount: Double = fullImpurityAgg.getCount
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

  /**
   * Choose splitting rule: feature value <= threshold
   * @return  (best split, statistics for split)  If the best split actually puts all instances
   *          in one leaf node, then it will be set to None.  The impurity stats maybe still be
   *          useful, so they are returned.
   */
  private[impl] def chooseContinuousSplitsForActiveNodes(
                                                          featureIndex: Int,
                                                          numActiveNodes: Int,
                                                          values: Array[Double],
                                                          indices: Array[Int],
                                                          labels: Array[Byte],
                                                          nodeIndices: Array[Int],
                                                          fullImpurityAggs: Array[ImpurityAggregatorSingle],
                                                          metadata: AltDTMetadata): Array[(Option[Split], ImpurityStats)] = {
    val leftImpurityAggs = Array.fill[ImpurityAggregatorSingle](numActiveNodes)(metadata.createImpurityAggregator())
    val bestLeftImpurityAggs = Array.fill[ImpurityAggregatorSingle](numActiveNodes)(metadata.createImpurityAggregator())
    val rightImpurityAggs = fullImpurityAggs.map(agg => agg.deepCopy())
    val fullImpurities = fullImpurityAggs.map(agg => agg.getCalculator.calculate())

    val bestThresholds = Array.fill[Double](numActiveNodes)(Double.NegativeInfinity)
    val currentThresholds = Array.fill[Double](numActiveNodes)(Double.NegativeInfinity)
    val bestGains = new Array[Double](numActiveNodes)
    val leftCounts = new Array[Int](numActiveNodes)
    val rightCounts = fullImpurityAggs.map(agg => agg.getCount)
    val fullCounts = rightCounts.clone()

    var i = 0
    while (i < labels.length) {
      val nodeIdx = nodeIndices(i)
      if (nodeIdx >= 0) {
        val value = values(i)
        val label = labels(indices(i))
        val leftImpurityAgg = leftImpurityAggs(nodeIdx)
        val rightImpurityAgg = rightImpurityAggs(nodeIdx)

        val currentThreshold = currentThresholds(nodeIdx)
        if (value != currentThreshold) {
          val leftCount = leftCounts(nodeIdx)
          val rightCount = rightCounts(nodeIdx)
          val fullCount = fullCounts(nodeIdx)
          val fullImpurity = fullImpurities(nodeIdx)
          val bestGain = bestGains(nodeIdx)
          // Check gain
          val leftWeight = leftCount / fullCount
          val rightWeight = rightCount / fullCount
          val leftImpurity = leftImpurityAgg.getCalculator.calculate()
          val rightImpurity = rightImpurityAgg.getCalculator.calculate()
          val gain = fullImpurity - leftWeight * leftImpurity - rightWeight * rightImpurity
          if (leftCount != 0 && rightCount != 0 && gain > bestGain && gain > metadata.minInfoGain) {
            bestThresholds(nodeIdx) = currentThreshold
            val bestLeftImpurityAgg = bestLeftImpurityAggs(nodeIdx)
            System.arraycopy(leftImpurityAgg.stats, 0, bestLeftImpurityAgg.stats, 0, leftImpurityAgg.stats.length)
            bestGains(nodeIdx) = gain
          }
          currentThresholds(nodeIdx) = value
        }
        // Move this instance from right to left side of split.
        leftImpurityAgg.update(label, 1)
        rightImpurityAgg.update(label, -1)
        leftCounts(nodeIdx) += 1
        rightCounts(nodeIdx) -= 1
      }
      i += 1
    }

    val splitsAndStats = Range(0, numActiveNodes).toArray.map { nodeIdx =>
      val bestThreshold = bestThresholds(nodeIdx)
      val bestRightImpurityAgg = fullImpurityAggs(nodeIdx).deepCopy().subtract(bestLeftImpurityAggs(nodeIdx))
      val split: Option[Split] = {
        if (bestThreshold != Double.NegativeInfinity && bestThreshold != currentThresholds(nodeIdx)) {
          Some(new ContinuousSplit(featureIndex, bestThreshold))
        } else {
          None
        }
      }
      (split, new ImpurityStats(bestGains(nodeIdx), fullImpurities(nodeIdx),
        fullImpurityAggs(nodeIdx).getCalculator, bestLeftImpurityAggs(nodeIdx).getCalculator,
        bestRightImpurityAgg.getCalculator))
    }
    splitsAndStats
  }

  def chooseContinuousSplitSparseForActiveNodes(
                                                 featureIndex: Int,
                                                 compressedVals: Array[(Double, Int)],
                                                 indices: Array[Int],
                                                 labels: Array[Byte],
                                                 from: Int,
                                                 to: Int,
                                                 nodeOffsets: Map[Int, Int],
                                                 initImpurityAgg: ImpurityAggregatorSingle,
                                                 metadata: AltDTMetadata): (Option[Split], ImpurityStats) = {
    val leftImpurityAgg = metadata.createImpurityAggregator()
    val rightImpurityAgg = initImpurityAgg.deepCopy()

    var bestThreshold: Double = Double.NegativeInfinity
    val bestLeftImpurityAgg = leftImpurityAgg.deepCopy()
    var bestGain: Double = 0.0
    val fullImpurity = rightImpurityAgg.getCalculator.calculate()
    var leftCount: Double = 0.0
    var rightCount: Double = rightImpurityAgg.getCount
    val fullCount: Double = rightCount

    var valuesJ = nodeOffsets(from)
    var indicesJ = from
    val valuesTo = nodeOffsets(to)
    while (valuesJ < valuesTo) {
      val (value, count) = compressedVals(valuesJ)
      var j = 0
      while (j < count) {
        val label = labels(indices(indicesJ))
        // Move this instance from right to left side of split.
        leftImpurityAgg.update(label, 1.0)
        rightImpurityAgg.update(label, -1.0)
        leftCount += 1.0
        rightCount -= 1.0
        indicesJ += 1
        j += 1
      }
      // Check gain
      val leftWeight = leftCount / fullCount
      val rightWeight = rightCount / fullCount
      val leftImpurity = leftImpurityAgg.getCalculator.calculate()
      val rightImpurity = rightImpurityAgg.getCalculator.calculate()
      val gain = fullImpurity - leftWeight * leftImpurity - rightWeight * rightImpurity
      if (leftCount != 0 && rightCount != 0 && gain > bestGain && gain > metadata.minInfoGain) {
        bestThreshold = value
        System.arraycopy(leftImpurityAgg.stats, 0, bestLeftImpurityAgg.stats, 0, leftImpurityAgg.stats.length)
        bestGain = gain
      }
      valuesJ += 1
    }

    val fullImpurityAgg = leftImpurityAgg.deepCopy().add(rightImpurityAgg)
    val bestRightImpurityAgg = fullImpurityAgg.deepCopy().subtract(bestLeftImpurityAgg)
    val bestImpurityStats = new ImpurityStats(bestGain, fullImpurity, fullImpurityAgg.getCalculator,
      bestLeftImpurityAgg.getCalculator, bestRightImpurityAgg.getCalculator)
    val split = if (bestThreshold != Double.NegativeInfinity && bestThreshold != compressedVals.last._1) {
      Some(new ContinuousSplit(featureIndex, bestThreshold))
    } else {
      None
    }
    (split, bestImpurityStats)
  }

  def chooseContinuousSplitsSparseForActiveNodes(
                                                  featureIndex: Int,
                                                  numActiveNodes: Int,
                                                  compressedVals: Array[(Double, Int)],
                                                  indices: Array[Int],
                                                  labels: Array[Byte],
                                                  nodeIndices: Array[Int],
                                                  fullImpurityAggs: Array[ImpurityAggregatorSingle],
                                                  metadata: AltDTMetadata): Array[(Option[Split], ImpurityStats)] = {

    val leftImpurityAggs = Array.fill[ImpurityAggregatorSingle](numActiveNodes)(metadata.createImpurityAggregator())
    val bestLeftImpurityAggs = Array.fill[ImpurityAggregatorSingle](numActiveNodes)(metadata.createImpurityAggregator())
    val rightImpurityAggs = fullImpurityAggs.map(agg => agg.deepCopy())
    val fullImpurities = fullImpurityAggs.map(agg => agg.getCalculator.calculate())

    val bestThresholds = Array.fill[Double](numActiveNodes)(Double.NegativeInfinity)
    val currentThresholds = Array.fill[Double](numActiveNodes)(Double.NegativeInfinity)
    val bestGains = new Array[Double](numActiveNodes)
    val leftCounts = new Array[Int](numActiveNodes)
    val fullCounts = fullImpurityAggs.map(agg => agg.getCount)
    val rightCounts = fullImpurityAggs.map(agg => agg.getCount.toInt)

    var i = 0
    var valIndex = 0
    var valCountIndex = 0
    var currVal = Double.NegativeInfinity
    var currCount = 0
    while (i < labels.length) {
      if (valCountIndex == currCount) {
        val valAndCount = compressedVals(valIndex)
        currVal = valAndCount._1
        currCount = valAndCount._2
        valCountIndex = 0
        valIndex += 1
      }

      val nodeIdx = nodeIndices(i)
      if (nodeIdx >= 0) {
        val label = labels(indices(i))
        val leftImpurityAgg = leftImpurityAggs(nodeIdx)
        val rightImpurityAgg = rightImpurityAggs(nodeIdx)

        val currentThreshold = currentThresholds(nodeIdx)
        if (currVal != currentThreshold) {
          val leftCount = leftCounts(nodeIdx)
          val rightCount = rightCounts(nodeIdx)
          val fullCount = fullCounts(nodeIdx)
          val fullImpurity = fullImpurities(nodeIdx)
          val bestGain = bestGains(nodeIdx)
          // Check gain
          val leftWeight = leftCount / fullCount
          val rightWeight = rightCount / fullCount
          val leftImpurity = leftImpurityAgg.getCalculator.calculate()
          val rightImpurity = rightImpurityAgg.getCalculator.calculate()
          val gain = fullImpurity - leftWeight * leftImpurity - rightWeight * rightImpurity
          if (leftCount != 0 && rightCount != 0 && gain > bestGain && gain > metadata.minInfoGain) {
            bestThresholds(nodeIdx) = currentThreshold
            val bestLeftImpurityAgg = bestLeftImpurityAggs(nodeIdx)
            System.arraycopy(leftImpurityAgg.stats, 0, bestLeftImpurityAgg.stats, 0, leftImpurityAgg.stats.length)
            bestGains(nodeIdx) = gain
          }
          currentThresholds(nodeIdx) = currVal
        }
        // Move this instance from right to left side of split.
        leftImpurityAgg.update(label, 1)
        rightImpurityAgg.update(label, -1)
        leftCounts(nodeIdx) += 1
        rightCounts(nodeIdx) -= 1
      }
      valCountIndex += 1
      i += 1
    }

    val splitsAndStats = Range(0, numActiveNodes).toArray.map { nodeIdx =>
      val bestThreshold = bestThresholds(nodeIdx)
      val bestRightImpurityAgg = fullImpurityAggs(nodeIdx).deepCopy().subtract(bestLeftImpurityAggs(nodeIdx))
      val split: Option[Split] = {
        if (bestThreshold != Double.NegativeInfinity && bestThreshold != currentThresholds(nodeIdx)) {
          Some(new ContinuousSplit(featureIndex, bestThreshold))
        } else {
          None
        }
      }
      (split, new ImpurityStats(bestGains(nodeIdx), fullImpurities(nodeIdx),
        fullImpurityAggs(nodeIdx).getCalculator, bestLeftImpurityAggs(nodeIdx).getCalculator,
        bestRightImpurityAgg.getCalculator))
    }
    splitsAndStats
  }
}
