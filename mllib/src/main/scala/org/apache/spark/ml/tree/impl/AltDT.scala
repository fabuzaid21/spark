/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.tree.impl

import org.apache.spark.Logging
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.TreeUtil._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.impl.DecisionTreeMetadata
import org.apache.spark.mllib.tree.impurity.{Entropy, Gini, Impurity, Variance}
import org.apache.spark.mllib.tree.model.ImpurityStats
import org.apache.spark.rdd.RDD
import org.apache.spark.util.collection.{BitSet, SortDataFormat, Sorter}
import org.roaringbitmap.RoaringBitmap
import java.util.{HashMap => JavaHashMap}

import scala.collection.mutable.ArrayBuffer


/**
 * DecisionTree which partitions data by feature.
 *
 * Algorithm:
 *  - Repartition data, grouping by feature.
 *  - Prep data (sort continuous features).
 *  - On each partition, initialize instance--node map with each instance at root node.
 *  - Iterate, training 1 new level of the tree at a time:
 *     - On each partition, for each feature on the partition, select the best split for each node.
 *     - Aggregate best split for each node.
 *     - Aggregate bit vector (1 bit/instance) indicating whether each instance splits
 *       left or right.
 *     - Broadcast bit vector.  On each partition, update instance--node map.
 *
 * TODO: Update to use a sparse column store.
 */
private[ml] object AltDT extends Logging {

  private[impl] class AltDTMetadata(
      val numClasses: Int,
      val maxBins: Int,
      val minInfoGain: Double,
      val impurity: Impurity,
      val categoricalFeaturesInfo: Map[Int, Int]) extends Serializable {

    private val unorderedSplits = {
      /**
       * borrowed from [[DecisionTreeMetadata.buildMetadata]]
       */
      if (numClasses > 2) {
        // Multiclass classification
        val maxCategoriesForUnorderedFeature =
          ((math.log(maxBins / 2 + 1) / math.log(2.0)) + 1).floor.toInt
        categoricalFeaturesInfo.filter { case (featureIndex, numCategories) =>
            numCategories > 1 && numCategories <= maxCategoriesForUnorderedFeature
        }.map { case (featureIndex, numCategories) =>
          // Hack: If a categorical feature has only 1 category, we treat it as continuous.
          // TODO(SPARK-9957): Handle this properly by filtering out those features.
          // Decide if some categorical features should be treated as unordered features,
          //  which require 2 * ((1 << numCategories - 1) - 1) bins.
          // We do this check with log values to prevent overflows in case numCategories is large.
          // The next check is equivalent to: 2 * ((1 << numCategories - 1) - 1) <= maxBins
              featureIndex -> findSplits(featureIndex, numCategories)
            }
      } else {
        Map.empty[Int, Array[CategoricalSplit]]
      }
    }

    /**
     * Returns all possible subsets of features for categorical splits.
     * Borrowed from [[RandomForest.findSplits]]
     */
    private def findSplits(
                            featureIndex: Int,
                            featureArity: Int): Array[CategoricalSplit] = {
      // Unordered features
      // 2^(featureArity - 1) - 1 combinations
      val numSplits = (1 << (featureArity - 1)) - 1
      val splits = new Array[CategoricalSplit](numSplits)

      var splitIndex = 0
      while (splitIndex < numSplits) {
        val categories: List[Double] =
          RandomForest.extractMultiClassCategories(splitIndex + 1, featureArity)
        splits(splitIndex) =
          new CategoricalSplit(featureIndex, categories.toArray, featureArity)
        splitIndex += 1
      }
      splits
    }

    def getUnorderedSplits(featureIndex: Int): Array[CategoricalSplit] = unorderedSplits(featureIndex)

    def isClassification: Boolean = numClasses >= 2

    def isMulticlass: Boolean = numClasses > 2

    def isUnorderedFeature(featureIndex: Int): Boolean = unorderedSplits.contains(featureIndex)

    def createImpurityAggregator(): ImpurityAggregatorSingle = {
      impurity match {
        case Entropy => new EntropyAggregatorSingle(numClasses)
        case Gini => new GiniAggregatorSingle(numClasses)
        case Variance => new VarianceAggregatorSingle
      }
    }
  }

  private[impl] object AltDTMetadata {
    def fromStrategy(strategy: Strategy): AltDTMetadata = new AltDTMetadata(strategy.numClasses,
      strategy.maxBins, strategy.minInfoGain, strategy.impurity, strategy.categoricalFeaturesInfo)
  }

  /**
   * Method to train a decision tree model over an RDD.
   */
  def train(
      input: RDD[LabeledPoint],
      strategy: Strategy,
      colStoreInput: Option[RDD[(Int, Array[Double])]] = None,
      parentUID: Option[String] = None): DecisionTreeModel = {
    // TODO: Check validity of params
    // TODO: Check for empty dataset
    val numFeatures = input.first().features.size
    val rootNode = trainImpl(input, strategy, colStoreInput)
    impl.RandomForest.finalizeTree(rootNode, strategy.algo, strategy.numClasses, numFeatures,
      parentUID)
  }

  private[impl] def trainImpl(
      input: RDD[LabeledPoint],
      strategy: Strategy,
      colStoreInput: Option[RDD[(Int, Array[Double])]]): Node = {

    val metadata = AltDTMetadata.fromStrategy(strategy)

    // The case with 1 node (depth = 0) is handled separately.
    // This allows all iterations in the depth > 0 case to use the same code.
    // TODO: Check that learning works when maxDepth > 0 but learning stops at 1 node (because of
    //       other parameters).
    if (strategy.maxDepth == 0) {
      val impurityAggregator: ImpurityAggregatorSingle =
        input.aggregate(metadata.createImpurityAggregator())(
          (agg, lp) => agg.update(lp.label, 1.0),
          (agg1, agg2) => agg1.add(agg2))
      val impurityCalculator = impurityAggregator.getCalculator
      return new LeafNode(impurityCalculator.getPredict.predict, impurityCalculator.calculate(),
        impurityCalculator)
    }

    // Prepare column store.
    val colStoreInit: RDD[(Int, Array[Double])] = colStoreInput.getOrElse(
      rowToColumnStoreDense(input.map(_.features)))
    val numRows: Int = colStoreInit.first()._2.length
    if (metadata.numClasses > 1 && metadata.numClasses <= 32) {
      AltDTClassification.trainImpl(input, colStoreInit, metadata, numRows, strategy.maxDepth)
    } else {
      AltDTRegression.trainImpl(input, colStoreInit, metadata, numRows, strategy.maxDepth)
    }
  }

  private[impl] def computeActiveNodeMap(
                                          bestSplitsAndGains: Array[(Option[Split], ImpurityStats)],
                                          minInfoGain: Double): JavaHashMap[Int, Int] = {
    val activeNodeMap = new JavaHashMap[Int, Int]()
    var newNodeIdx = 0
    bestSplitsAndGains.zipWithIndex.foreach { case ((split, stats), nodeIdx) =>
      if (split.nonEmpty && stats.gain > minInfoGain) {
        activeNodeMap.put(nodeIdx << 1, newNodeIdx)
        activeNodeMap.put((nodeIdx << 1) + 1, newNodeIdx + 1)
        newNodeIdx += 2
      }
    }
    activeNodeMap
  }

  /**
   * On driver: Grow tree based on chosen splits, and compute new set of active nodes.
   * @param oldPeriphery  Old periphery of active nodes.
   * @param bestSplitsAndGains  Best (split, gain) pairs, which can be zipped with the old
   *                            periphery.  These stats will be used to replace the stats in
   *                            any nodes which are split.
   * @param minInfoGain  Threshold for min info gain required to split a node.
   * @return  New active node periphery.
   *          If a node is split, then this method will update its fields.
   */
  private[impl] def computeActiveNodePeriphery(
                                                oldPeriphery: Array[LearningNode],
                                                bestSplitsAndGains: Array[(Option[Split], ImpurityStats)],
                                                minInfoGain: Double): Array[LearningNode] = {
    bestSplitsAndGains.zipWithIndex.flatMap {
      case ((split, stats), nodeIdx) =>
        val node = oldPeriphery(nodeIdx)
        if (split.nonEmpty && stats.gain > minInfoGain) {
          // TODO: remove node id
          node.leftChild = Some(LearningNode(node.id * 2, isLeaf = false,
            ImpurityStats(stats.leftImpurity, stats.leftImpurityCalculator)))
          node.rightChild = Some(LearningNode(node.id * 2 + 1, isLeaf = false,
            ImpurityStats(stats.rightImpurity, stats.rightImpurityCalculator)))
          node.split = split
          node.isLeaf = false
          node.stats = stats
          Iterator(node.leftChild.get, node.rightChild.get)
        } else {
          node.isLeaf = true
          Iterator()
        }
    }
  }

  /**
   * Aggregate bit vector (1 bit/instance) indicating whether each instance goes left/right.
   * - Send chosen splits to workers.
   * - Each worker creates part of the bit vector corresponding to the splits it created.
   * - Aggregate the partial bit vectors to create one vector (of length numRows).
   *   Correction: Aggregate only the pieces of that vector corresponding to instances at
   *   active nodes.
   * @param partitionInfos  RDD with feature data, plus current status metadata
   * @param bestSplits  Split for each active node, or None if that node will not be split
   * @return Array of bit vectors, ordered by offset ranges
   */
  private[impl] def aggregateBitVector(
      partitionInfos: RDD[PartitionInfo],
      bestSplits: Array[Option[Split]]): RoaringBitmap = {

    val bestSplitsBc: Broadcast[Array[Option[Split]]] =
      partitionInfos.sparkContext.broadcast(bestSplits)
    val workerBitSubvectors = partitionInfos.map {
      case PartitionInfo(columns: Array[FeatureVector], fullImpurityAggs: Array[ImpurityAggregatorSingle]) =>
        val localBestSplits: Array[Option[Split]] = bestSplitsBc.value
        // localFeatureIndex[feature index] = index into PartitionInfo.columns
        val localFeatureIndex: Map[Int, Int] = columns.map(_.featureIndex).zipWithIndex.toMap
        val bitvectorForNodes = localBestSplits.zipWithIndex.flatMap {
          case (Some(split: Split), nodeIndexInLevel) =>
            if (localFeatureIndex.contains(split.featureIndex)) {
              // This partition has the column (feature) used for this split.
              val colIndex = localFeatureIndex(split.featureIndex)
              val col = columns(colIndex)
              val bv = {
                if (col.sparse) {
                  bitVectorFromSplitSparse(col, nodeIndexInLevel, split)
                } else {
                  bitVectorFromSplit(col, nodeIndexInLevel, split)
                }
              }
              Iterator(bv)
            } else {
              Iterator()
            }
          case (None, nodeIndexInLevel) =>
            // Do not create a bitVector when there is no split.
            // PartitionInfo.update will detect that there is no
            // split, by how many instances go left/right.
            Iterator()
        }
        val totalBitVector = bitvectorForNodes.fold(new RoaringBitmap()) { (acc, bitv) =>
          acc.or(bitv)
          acc
        }
        totalBitVector
    }
    val aggBitVector: RoaringBitmap = workerBitSubvectors.reduce { (acc, bitv) =>
      acc.or(bitv)
      acc
    }
    bestSplitsBc.unpersist()
    aggBitVector
  }

  /**
   * For a given feature, for a given node, apply a split and return a bit vector indicating the
   * outcome of the split for each instance at that node.
   *
   * @param col  Column for feature
   * @param split  Split to apply to instances at this node.
   * @return  Bits indicating splits for instances at this node.
   *          These bits are sorted by the row indices, in order to guarantee an ordering
   *          understood by all workers.
   *          Thus, the bit indices used are based on 2-level sorting: first by node, and
   *          second by sorted row indices within the node's rows.
   *          bit[index in sorted array of row indices] = false for left, true for right
   */
  private[impl] def bitVectorFromSplit(
                                        col: FeatureVector,
                                        nodeIndex: Int,
                                        split: Split): RoaringBitmap = {
    val bitv = new RoaringBitmap()
    var i = 0
    while (i < col.nodeIndices.length) {
      if (col.nodeIndices(i) == nodeIndex) {
        val value = col.values(i)
        val idx = col.indices(i)
        if (!split.shouldGoLeft(value)) {
          bitv.add(idx)
        }
      }
      i += 1
    }
    bitv
  }

  /**
   * Same as above, but for sparse columns (columns that have been compressed via run-length encoding)
   */
  private[impl] def bitVectorFromSplitSparse(
                                        col: FeatureVector,
                                        nodeIndex: Int,
                                        split: Split): RoaringBitmap = {
    val bitv = new RoaringBitmap()
    var i = 0
    var valIndex = 0
    var valCountIndex = 0
    var currVal = Double.NegativeInfinity
    var currCount = 0
    while (i < col.nodeIndices.length) {
      if (valCountIndex == currCount) {
        val valAndCount = col.compressedVals(valIndex)
        currVal = valAndCount._1
        currCount = valAndCount._2
        valCountIndex = 0
        valIndex += 1
      }
      if (col.nodeIndices(i) == nodeIndex) {
        val idx = col.indices(i)
        if (!split.shouldGoLeft(currVal)) {
          bitv.add(idx)
        }
      }
      valCountIndex += 1
      i += 1
    }
    bitv
  }

  /**
   * Intermediate data stored on each partition during learning.
   *
   * Node indexing for nodeOffsets, activeNodes:
   * Nodes are indexed left-to-right along the periphery of the tree, with 0-based indices.
   * The periphery is the set of leaf nodes (active and inactive).
   *
   * @param columns  Subset of columns (features) stored in this partition.
   *                 Each column is sorted first by nodes (left-to-right along the tree periphery);
   *                 all columns share this first level of sorting.
   *                 Within each node's group, each column is sorted based on feature value;
   *                 this second level of sorting differs across columns.
   */
  private[impl] case class PartitionInfo(
                                          columns: Array[FeatureVector],
                                          fullImpurityAggs: Array[ImpurityAggregatorSingle])
    extends Serializable {

    /** For debugging */
    override def toString: String = {
      "PartitionInfo(" +
        "  columns: {\n" +
        columns.mkString(",\n") +
        "  }\n" +
        ")\n"
    }

    /**
     * Update columns and nodeOffsets for the next level of the tree.
     *
     * Update columns:
     *   For each column,
     *     For each (previously) active node,
     *       Sort corresponding range of instances based on bit vector.
     * Update nodeOffsets, activeNodes:
     *   Split offsets for nodes which split (which can be identified using the bit vector).
     *
     * @param instanceBitVector  Bit vector encoding splits for the next level of the tree.
     *                    These must follow a 2-level ordering, where the first level is by node
     *                    and the second level is by row index.
     *                    bitVector(i) = false iff instance i goes to the left child.
     *                    For instances at inactive (leaf) nodes, the value can be arbitrary.
     * @return Updated partition info
     */
    def update(instanceBitVector: BitSet, activeNodeMap: JavaHashMap[Int, Int],
               labels: Array[Byte], metadata: AltDTMetadata): PartitionInfo = {
      val newFullImpurityAggs = Array.fill[ImpurityAggregatorSingle](activeNodeMap.size)(metadata.createImpurityAggregator())
      columns.zipWithIndex.foreach { case (col, index) =>
        (index, col.isCategorical) match {
          case (0, true) => firstCategorical(col, instanceBitVector, newFullImpurityAggs, activeNodeMap, metadata, labels)
          case (_, true) => restCategorical(col, instanceBitVector, activeNodeMap, metadata, labels)
          case (0, false) => first(col, instanceBitVector, newFullImpurityAggs, activeNodeMap, metadata, labels)
          case (_, false) => rest(col, instanceBitVector, activeNodeMap)
        }
      }
      val newColumns = columns.map(col => new FeatureVector(col.featureIndex, col.featureArity,
        col.values, col.indices, col.nodeIndices, col.aggStats, col.compressedVals, col.sparse))
      PartitionInfo(newColumns, newFullImpurityAggs)
    }


    /**
     * Sort the very first column in the [[PartitionInfo.columns]]. While
     * (by modifying @param newFullImpurityAggs).
     * @param col The very first column in [[PartitionInfo.columns]]
     * @param metadata Used to create new [[ImpurityAggregatorSingle]] for a new child
     *                 node in the tree
     * @param labels   Labels are read as we sort column to populate stats for each
     *                 new ImpurityAggregatorSingle
     */
    private def first(
                       col: FeatureVector,
                       instanceBitVector: BitSet,
                       newFullImpurityAggs: Array[ImpurityAggregatorSingle],
                       activeNodeMap: JavaHashMap[Int, Int],
                       metadata: AltDTMetadata,
                       labels: Array[Byte]) = {
      var idx = 0
      while (idx < col.indices.length) {
        val indexForVal = col.indices(idx)
        var key = col.nodeIndices(idx) << 1
        if (instanceBitVector.get(indexForVal)) {
          key += 1
        }
        // -1 means that this node was not split
        val nodeIndex = if (activeNodeMap.containsKey(key)) activeNodeMap.get(key) else -1
        if (nodeIndex >= 0) {
          val label = labels(indexForVal)
          newFullImpurityAggs(nodeIndex).update(label)
        }
        col.nodeIndices(idx) = nodeIndex
        idx += 1
      }
    }

    /**
     * Sort the very first column in the [[PartitionInfo.columns]]. While
     * (by modifying @param newFullImpurityAggs).
     * @param col The very first column in [[PartitionInfo.columns]]
     * @param metadata Used to create new [[ImpurityAggregatorSingle]] for a new child
     *                 node in the tree
     * @param labels   Labels are read as we sort column to populate stats for each
     *                 new ImpurityAggregatorSingle
     */
    private def firstCategorical(
                       col: FeatureVector,
                       instanceBitVector: BitSet,
                       newFullImpurityAggs: Array[ImpurityAggregatorSingle],
                       activeNodeMap: JavaHashMap[Int, Int],
                       metadata: AltDTMetadata,
                       labels: Array[Byte]) = {

      col.aggStats = Array.fill[Array[ImpurityAggregatorSingle]](activeNodeMap.size)(
        Array.tabulate[ImpurityAggregatorSingle](col.featureArity)(
          _ => metadata.createImpurityAggregator()
        )
      )

      var idx = 0
      var valIndex = 0
      var valCountIndex = 0
      var currVal = Double.NegativeInfinity
      var currCount = 0
      while (idx < col.indices.length) {
        if (valCountIndex == currCount) {
          val valAndCount = col.compressedVals(valIndex)
          currVal = valAndCount._1
          currCount = valAndCount._2
          valCountIndex = 0
          valIndex += 1
        }
        val indexForVal = col.indices(idx)
        var key = col.nodeIndices(idx) << 1
        if (instanceBitVector.get(indexForVal)) {
          key += 1
        }
        // -1 means that this node was not split
        val nodeIndex = if (activeNodeMap.containsKey(key)) activeNodeMap.get(key) else -1
        if (nodeIndex >= 0) {
          val label = labels(indexForVal)
          newFullImpurityAggs(nodeIndex).update(label)
          col.aggStats(nodeIndex)(currVal.toInt).update(label)
        }
        col.nodeIndices(idx) = nodeIndex
        valCountIndex += 1
        idx += 1
      }
    }

    /**
     * Sort the remaining columns in the [[PartitionInfo.columns]]. Since
     * we skip the computation for those here.
     * @param col The very first column in [[PartitionInfo.columns]]
     */
    private def restCategorical(
                                 col: FeatureVector,
                                 instanceBitVector: BitSet,
                                 activeNodeMap: JavaHashMap[Int, Int],
                                 metadata: AltDTMetadata,
                                 labels: Array[Byte]) = {

      col.aggStats = Array.fill[Array[ImpurityAggregatorSingle]](activeNodeMap.size)(
        Array.tabulate[ImpurityAggregatorSingle](col.featureArity)(
          _ => metadata.createImpurityAggregator()
        )
      )

      var idx = 0
      var valIndex = 0
      var valCountIndex = 0
      var currVal = Double.NegativeInfinity
      var currCount = 0
      while (idx < col.indices.length) {
        if (valCountIndex == currCount) {
          val valAndCount = col.compressedVals(valIndex)
          currVal = valAndCount._1
          currCount = valAndCount._2
          valCountIndex = 0
          valIndex += 1
        }
        val indexForVal = col.indices(idx)
        var key = col.nodeIndices(idx) << 1
        if (instanceBitVector.get(indexForVal)) {
          key += 1
        }
        // -1 means that this node was not split
        val nodeIndex = if (activeNodeMap.containsKey(key)) activeNodeMap.get(key) else -1
        if (nodeIndex >= 0) {
          val label = labels(indexForVal)
          col.aggStats(nodeIndex)(currVal.toInt).update(label)
        }
        col.nodeIndices(idx) = nodeIndex
        valCountIndex += 1
        idx += 1
      }
    }

    /**
     * Update columns and nodeOffsets for the next level of the tree.
     *
     * Update columns:
     *   For each column,
     *     For each (previously) active node,
     *       Sort corresponding range of instances based on bit vector.
     * Update nodeOffsets, activeNodes:
     *   Split offsets for nodes which split (which can be identified using the bit vector).
     *
     * @param instanceBitVector  Bit vector encoding splits for the next level of the tree.
     *                    These must follow a 2-level ordering, where the first level is by node
     *                    and the second level is by row index.
     *                    bitVector(i) = false iff instance i goes to the left child.
     *                    For instances at inactive (leaf) nodes, the value can be arbitrary.
     * @return Updated partition info
     */
    def update(instanceBitVector: BitSet, activeNodeMap: JavaHashMap[Int, Int],
               labels: Array[Double], metadata: AltDTMetadata): PartitionInfo = {
      val newFullImpurityAggs = Array.fill[ImpurityAggregatorSingle](activeNodeMap.size)(metadata.createImpurityAggregator())
      columns.zipWithIndex.foreach { case (col, index) =>
        (index, col.isCategorical) match {
          case (0, true) => firstCategorical(col, instanceBitVector, newFullImpurityAggs, activeNodeMap, metadata, labels)
          case (_, true) => restCategorical(col, instanceBitVector, activeNodeMap, metadata, labels)
          case (0, false) => first(col, instanceBitVector, newFullImpurityAggs, activeNodeMap, metadata, labels)
          case (_, false) => rest(col, instanceBitVector, activeNodeMap)
        }
      }
      val newColumns = columns.map(col => new FeatureVector(col.featureIndex, col.featureArity,
        col.values, col.indices, col.nodeIndices, col.aggStats, col.compressedVals, col.sparse))
      PartitionInfo(newColumns, newFullImpurityAggs)
    }

    /**
     * Sort the very first column in the [[PartitionInfo.columns]]. While
     * (by modifying @param newFullImpurityAggs).
     * @param col The very first column in [[PartitionInfo.columns]]
     * @param metadata Used to create new [[ImpurityAggregatorSingle]] for a new child
     *                 node in the tree
     * @param labels   Labels are read as we sort column to populate stats for each
     *                 new ImpurityAggregatorSingle
     */
    private def first(
                       col: FeatureVector,
                       instanceBitVector: BitSet,
                       newFullImpurityAggs: Array[ImpurityAggregatorSingle],
                       activeNodeMap: JavaHashMap[Int, Int],
                       metadata: AltDTMetadata,
                       labels: Array[Double]) = {
      var idx = 0
      while (idx < col.indices.length) {
        val indexForVal = col.indices(idx)
        var key = col.nodeIndices(idx) << 1
        if (instanceBitVector.get(indexForVal)) {
          key += 1
        }
        // -1 means that this node was not split
        val nodeIndex = if (activeNodeMap.containsKey(key)) activeNodeMap.get(key) else -1
        if (nodeIndex >= 0) {
          val label = labels(indexForVal)
          newFullImpurityAggs(nodeIndex).update(label)
        }
        col.nodeIndices(idx) = nodeIndex
        idx += 1
      }
    }

    /**
     * Sort the very first column in the [[PartitionInfo.columns]]. While
     * (by modifying @param newFullImpurityAggs).
     * @param col The very first column in [[PartitionInfo.columns]]
     * @param metadata Used to create new [[ImpurityAggregatorSingle]] for a new child
     *                 node in the tree
     * @param labels   Labels are read as we sort column to populate stats for each
     *                 new ImpurityAggregatorSingle
     */
    private def firstCategorical(
                                  col: FeatureVector,
                                  instanceBitVector: BitSet,
                                  newFullImpurityAggs: Array[ImpurityAggregatorSingle],
                                  activeNodeMap: JavaHashMap[Int, Int],
                                  metadata: AltDTMetadata,
                                  labels: Array[Double]) = {

      col.aggStats = Array.fill[Array[ImpurityAggregatorSingle]](activeNodeMap.size)(
        Array.tabulate[ImpurityAggregatorSingle](col.featureArity)(
          _ => metadata.createImpurityAggregator()
        )
      )

      var idx = 0
      var valIndex = 0
      var valCountIndex = 0
      var currVal = Double.NegativeInfinity
      var currCount = 0
      while (idx < col.indices.length) {
        if (valCountIndex == currCount) {
          val valAndCount = col.compressedVals(valIndex)
          currVal = valAndCount._1
          currCount = valAndCount._2
          valCountIndex = 0
          valIndex += 1
        }
        val indexForVal = col.indices(idx)
        var key = col.nodeIndices(idx) << 1
        if (instanceBitVector.get(indexForVal)) {
          key += 1
        }
        // -1 means that this node was not split
        val nodeIndex = if (activeNodeMap.containsKey(key)) activeNodeMap.get(key) else -1
        if (nodeIndex >= 0) {
          val label = labels(indexForVal)
          newFullImpurityAggs(nodeIndex).update(label)
          col.aggStats(nodeIndex)(currVal.toInt).update(label)
        }
        col.nodeIndices(idx) = nodeIndex
        valCountIndex += 1
        idx += 1
      }
    }

    /**
     * Sort the remaining columns in the [[PartitionInfo.columns]]. Since
     * we skip the computation for those here.
     * @param col The very first column in [[PartitionInfo.columns]]
     */
    private def restCategorical(
                                 col: FeatureVector,
                                 instanceBitVector: BitSet,
                                 activeNodeMap: JavaHashMap[Int, Int],
                                 metadata: AltDTMetadata,
                                 labels: Array[Double]) = {

      col.aggStats = Array.fill[Array[ImpurityAggregatorSingle]](activeNodeMap.size)(
        Array.tabulate[ImpurityAggregatorSingle](col.featureArity)(
          _ => metadata.createImpurityAggregator()
        )
      )

      var idx = 0
      var valIndex = 0
      var valCountIndex = 0
      var currVal = Double.NegativeInfinity
      var currCount = 0
      while (idx < col.indices.length) {
        if (valCountIndex == currCount) {
          val valAndCount = col.compressedVals(valIndex)
          currVal = valAndCount._1
          currCount = valAndCount._2
          valCountIndex = 0
          valIndex += 1
        }
        val indexForVal = col.indices(idx)
        var key = col.nodeIndices(idx) << 1
        if (instanceBitVector.get(indexForVal)) {
          key += 1
        }
        // -1 means that this node was not split
        val nodeIndex = if (activeNodeMap.containsKey(key)) activeNodeMap.get(key) else -1
        if (nodeIndex >= 0) {
          val label = labels(indexForVal)
          col.aggStats(nodeIndex)(currVal.toInt).update(label)
        }
        col.nodeIndices(idx) = nodeIndex
        valCountIndex += 1
        idx += 1
      }
    }

    /**
     * Sort the remaining columns in the [[PartitionInfo.columns]]. Since
     * we skip the computation for those here.
     * @param col The very first column in [[PartitionInfo.columns]]
     */
    private def rest(
                      col: FeatureVector,
                      instanceBitVector: BitSet,
                      activeNodeMap: JavaHashMap[Int, Int]) = {
      var idx = 0
      while (idx < col.indices.length) {
        val indexForVal = col.indices(idx)
        var key = col.nodeIndices(idx) << 1
        if (instanceBitVector.get(indexForVal)) {
          key += 1
        }
        col.nodeIndices(idx) = if (activeNodeMap.containsKey(key)) activeNodeMap.get(key)
                               else -1 // -1 means that this node was not split
        idx += 1
      }
    }
  }

   /**
    * Feature vector types are based on (feature type, representation).
    * The feature type can be continuous or categorical, ordered or unordered.
    *
    * Features are sorted by value, so we must store indices + values.
    * @param featureArity  For categorical features, this gives the number of categories.
    *                      For continuous features, this should be set to 0.
    */
  private[impl] class FeatureVector(
                                     val featureIndex: Int,
                                     val featureArity: Int,
                                     var values: Array[Double],
                                     val indices: Array[Int],
                                     val nodeIndices: Array[Int],
                                     // for categorical features only
                                     var aggStats: Array[Array[ImpurityAggregatorSingle]] = null,
                                     // attributes for sparse columns that are run-length encoded
                                     var compressedVals: Array[(Double, Int)] = null,
                                     var sparse: Boolean = false) extends Serializable {

     def isCategorical: Boolean = featureArity > 0

     /** For debugging */
     override def toString: String = {
       "  FeatureVector(\n" +
         s"    featureIndex: $featureIndex,\n" +
         s"    featureType: ${if (featureArity == 0) "Continuous" else "Categorical"},\n" +
         s"    sparse: $sparse,\n" +
         s"    featureArity: $featureArity,\n" +
         s"    values: ${if (sparse) compressedVals.mkString(", ") else values.mkString(", ")},\n" +
         s"    indices: ${indices.mkString(", ")},\n" +
         s"    nodeIndices: ${nodeIndices.mkString(", ")},\n" +
         "  )"
     }

     def deepCopy(): FeatureVector =
       new FeatureVector(featureIndex, featureArity, values.clone(), indices.clone(), nodeIndices.clone())

     override def equals(other: Any): Boolean = {
       other match {
         case o: FeatureVector =>
           if (sparse) {
             if (o.sparse) {
               featureIndex == o.featureIndex && featureArity == o.featureArity &&
                 compressedVals.sameElements(o.compressedVals) &&
                 indices.sameElements(o.indices) &&
                 nodeIndices.sameElements(o.nodeIndices)
             } else false
           } else {
             if (o.sparse) false
             else {
               featureIndex == o.featureIndex && featureArity == o.featureArity &&
                 values.sameElements(o.values) && indices.sameElements(o.indices) &&
                 nodeIndices.sameElements(o.nodeIndices)
             }
           }
         case _ => false
       }
     }
  }

  private[impl] object FeatureVector {

    /** Store column sorted by feature values. */
    def fromOriginal(
                      featureIndex: Int,
                      featureArity: Int,
                      values: Array[Double],
                      initAggStats: Array[Array[ImpurityAggregatorSingle]] = null): FeatureVector = {
      val indices = values.indices.toArray
      val nodeIndices = new Array[Int](indices.length)
      val fv = new FeatureVector(featureIndex, featureArity, values, indices, nodeIndices, initAggStats)
      val sorter = new Sorter(new FeatureVectorSortByValue())
      sorter.sort(fv, 0, values.length, Ordering[KeyWrapper])
      // if the featuer is categorical or there are only half as many distinct values,
      // then run-length encoding is worth it
      if (initAggStats != null || values.length / values.distinct.length.toDouble > 2.0) {
        fv.compressedVals = runLengthEncoding(values)
        fv.sparse = true
        fv.values = null
      }
      fv
    }

    def initAggStats(
                      values: Array[Double],
                      labels: Array[Double],
                      featureArity: Int,
                      metadata: AltDTMetadata): Array[ImpurityAggregatorSingle] = {
      val aggStats = Array.tabulate[ImpurityAggregatorSingle](featureArity)(
        _ => metadata.createImpurityAggregator())
      var i = 0
      while (i < labels.length) {
        val cat = values(i)
        val label = labels(i)
        aggStats(cat.toInt).update(label)
        i += 1
      }
      aggStats
    }

    def initAggStats(
                      values: Array[Double],
                      labels: Array[Byte],
                      featureArity: Int,
                      metadata: AltDTMetadata): Array[ImpurityAggregatorSingle] = {
      val aggStats = Array.tabulate[ImpurityAggregatorSingle](featureArity)(
        _ => metadata.createImpurityAggregator())
      var i = 0
      while (i < labels.length) {
        val cat = values(i)
        val label = labels(i)
        aggStats(cat.toInt).update(label)
        i += 1
      }
      aggStats
    }

    def runLengthEncoding(values: Array[Double]): Array[(Double, Int)] = {
      val rle = new ArrayBuffer[(Double, Int)]()
      var currVal = values(0)
      var i = 1
      var count = 1
      while (i < values.length) {
        val nextVal = values(i)
        if (nextVal != currVal) {
          rle += ((currVal, count))
          currVal = nextVal
          count = 1
        } else {
          count += 1
        }
        i += 1
      }
      // add the last one
      rle += ((currVal, count))
      rle.toArray
    }
  }

  /**
    * Sort FeatureVector by values column; @see [[FeatureVector.fromOriginal()]]
    */
  private class FeatureVectorSortByValue extends SortDataFormat[KeyWrapper, FeatureVector] {

    override def newKey(): KeyWrapper = new KeyWrapper()

    override def getKey(data: FeatureVector,
                        pos: Int,
                        reuse: KeyWrapper): KeyWrapper = {
      if (reuse == null) {
        new KeyWrapper().setKey(data.values(pos))
      } else {
        reuse.setKey(data.values(pos))
      }
    }

    override def getKey(data: FeatureVector,
                        pos: Int): KeyWrapper = {
      getKey(data, pos, null)
    }

    private def swapElements(data: Array[Double],
                                pos0: Int,
                                pos1: Int): Unit = {
      val tmp = data(pos0)
      data(pos0) = data(pos1)
      data(pos1) = tmp
    }

    private def swapElements(data: Array[Int],
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
      new FeatureVector(0, 0, new Array[Double](length), new Array[Int](length), null, null)
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
  private class KeyWrapper extends Ordered[KeyWrapper] {

    var key: Double = _

    override def compare(that: KeyWrapper): Int = {
      scala.math.Ordering.Double.compare(key, that.key)
    }

    def setKey(key: Double): this.type = {
      this.key = key
      this
    }
  }
}
