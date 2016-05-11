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

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.AltDT.{PartitionInfo, FeatureVector, AltDTMetadata}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.impurity.{EntropyCalculator, Entropy, Gini}
import org.apache.spark.mllib.tree.model.ImpurityStats
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.util.collection.BitSet
import java.util.{HashMap => JavaHashMap}

import scala.util.Random

/**
 * Test suite for [[AltDT]].
 */
class AltDTSuite extends SparkFunSuite with MLlibTestSparkContext  {

  /* * * * * * * * * * * Integration tests * * * * * * * * * * */

  test("run deep example") {
    val data = Range(0, 3).map(x => LabeledPoint(math.pow(x, 3), Vectors.dense(x)))
    val df = sqlContext.createDataFrame(data)
    val dt = new DecisionTreeRegressor()
      .setFeaturesCol("features") // indexedFeatures
      .setLabelCol("label")
      .setMaxDepth(10)
      .setAlgorithm("byCol")
    val model = dt.fit(df)
    assert(model.rootNode.isInstanceOf[InternalNode])
    val root = model.rootNode.asInstanceOf[InternalNode]
    assert(root.leftChild.isInstanceOf[InternalNode] && root.rightChild.isInstanceOf[LeafNode])
    val left = root.leftChild.asInstanceOf[InternalNode]
    assert(left.leftChild.isInstanceOf[LeafNode], left.rightChild.isInstanceOf[LeafNode])
  }

  test("run example") {
    val data = Range(0, 8).map(x => LabeledPoint(x, Vectors.dense(x)))
    val df = sqlContext.createDataFrame(data)
    val dt = new DecisionTreeRegressor()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setMaxDepth(10)
      .setAlgorithm("byCol")
    val model = dt.fit(df)
    assert(model.rootNode.isInstanceOf[InternalNode])
    val root = model.rootNode.asInstanceOf[InternalNode]
    assert(root.leftChild.isInstanceOf[InternalNode] && root.rightChild.isInstanceOf[InternalNode])
    val left = root.leftChild.asInstanceOf[InternalNode]
    val right = root.rightChild.asInstanceOf[InternalNode]
    val grandkids = Array(left.leftChild, left.rightChild, right.leftChild, right.rightChild)
    assert(grandkids.forall(_.isInstanceOf[InternalNode]))
  }

  test("example with imbalanced tree") {
    val data = Seq(
      (0.0, Vectors.dense(0.0, 0.0)),
      (0.0, Vectors.dense(0.0, 0.0)),
      (1.0, Vectors.dense(0.0, 1.0)),
      (0.0, Vectors.dense(0.0, 1.0)),
      (1.0, Vectors.dense(1.0, 0.0)),
      (1.0, Vectors.dense(1.0, 0.0)),
      (1.0, Vectors.dense(1.0, 1.0)),
      (1.0, Vectors.dense(1.0, 1.0))
    ).map { case (l, p) => LabeledPoint(l, p) }
    val df = sqlContext.createDataFrame(data)
    val dt = new DecisionTreeRegressor()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setMaxDepth(5)
      .setAlgorithm("byCol")
    val model = dt.fit(df)
    assert(model.depth === 2)
    assert(model.numNodes === 5)
  }

  test("example providing transposed dataset") {
    val data = Range(0, 8).map(x => LabeledPoint(x, Vectors.dense(x)))
    val transposedDataset = TreeUtil.rowToColumnStoreDense(sc.parallelize(data.map(_.features)))
    val df = sqlContext.createDataFrame(data)
    val dt = new DecisionTreeRegressor()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setMaxDepth(10)
      .setAlgorithm("byCol")
    val model = dt.fit(df, transposedDataset)
    assert(model.rootNode.isInstanceOf[InternalNode])
    val root = model.rootNode.asInstanceOf[InternalNode]
    assert(root.leftChild.isInstanceOf[InternalNode] && root.rightChild.isInstanceOf[InternalNode])
    val left = root.leftChild.asInstanceOf[InternalNode]
    val right = root.rightChild.asInstanceOf[InternalNode]
    val grandkids = Array(left.leftChild, left.rightChild, right.leftChild, right.rightChild)
    assert(grandkids.forall(_.isInstanceOf[InternalNode]))
  }

  /* * * * * * * * * * * Helper classes * * * * * * * * * * */

  test("FeatureVector") {
    val metadata = new AltDTMetadata(numClasses = 2, maxBins = 4, minInfoGain = 0.0, Entropy, Map(1 -> 3))
    val emptyAgg = metadata.createImpurityAggregator()

    val v = new FeatureVector(1, 0, Array(0.1, 0.3, 0.7), Array(1, 2, 0), Array(emptyAgg), Array(0, 0, 0))

    val vCopy = v.deepCopy()
    vCopy.values(0) = 1000
    assert(v.values(0) !== vCopy.values(0))

    val original = Array(0.7, 0.1, 0.3)
    val v2 = FeatureVector.fromOriginal(1, 0, original, emptyAgg)
    assert(v === v2)
  }

  test("FeatureVectorSortByValue") {
    val values = Array(0.1, 0.2, 0.4, 0.6, 0.7, 0.9, 1.5, 1.55)
    val col = Random.shuffle(values.toIterator).toArray
    val unsortedIndices = col.indices
    val sortedIndices = unsortedIndices.sortBy(x => col(x)).toArray
    val featureIndex = 3
    val featureArity = 0
    val fvSorted =
      FeatureVector.fromOriginal(featureIndex, featureArity, col, null)
    assert(fvSorted.featureIndex === featureIndex)
    assert(fvSorted.featureArity === featureArity)
    assert(fvSorted.values.deep === values.deep)
    assert(fvSorted.indices.deep === sortedIndices.deep)
  }

  test("PartitionInfo") {
    val numRows = 4
    val labels = Array(0, 0, 1, 1).map(_.toByte)
    val metadata = new AltDTMetadata(numClasses = 2, maxBins = 4, minInfoGain = 0.0, Entropy, Map(1 -> 3))
    val fullImpurityAgg = metadata.createImpurityAggregator()
    labels.foreach(label => fullImpurityAgg.update(label))
    val col1 = FeatureVector.fromOriginal(0, 0, Array(0.8, 0.2, 0.1, 0.6), fullImpurityAgg)
    val col2 = FeatureVector.fromOriginal(1, 3, Array(0, 1, 0, 2), fullImpurityAgg)

    assert(col1.values.length === numRows)
    assert(col2.values.length === numRows)

    val info = PartitionInfo(Array(col1, col2))

    // Create bitVector for splitting the 4 rows: L, R, L, R
    // New groups are {0, 2}, {1, 3}
    val bitVector = new BitSet(numRows)
    bitVector.set(1)
    bitVector.set(3)
    val activeNodeMap = new JavaHashMap[Int, Int]()
    activeNodeMap.put(0, 0)
    activeNodeMap.put(1, 1)

    // for these tests, use the activeNodes for nodeSplitBitVector
    val newInfo = info.update(bitVector, activeNodeMap, labels, metadata)

    val leftImpurityAgg = metadata.createImpurityAggregator()
    leftImpurityAgg.update(labels(0))
    leftImpurityAgg.update(labels(2))
    val rightImpurityAgg = metadata.createImpurityAggregator()
    rightImpurityAgg.update(labels(1))
    rightImpurityAgg.update(labels(3))
    val expectedFullImpurityAggs1 = Array(leftImpurityAgg, rightImpurityAgg)

    assert(newInfo.columns.length === 2)
    val expectedCol1a = new FeatureVector(0, 0, Array(0.1, 0.2, 0.6, 0.8), Array(2, 1, 3, 0),
      expectedFullImpurityAggs1, Array(0, 1, 1, 0))

    val expectedCol1b = new FeatureVector(1, 3, Array(0, 0, 1, 2), Array(0, 2, 1, 3),
      expectedFullImpurityAggs1, Array(0, 0, 1, 1))

    assert(newInfo.columns(0) === expectedCol1a)
    assert(newInfo.columns(1) === expectedCol1b)

    // Create 2 bitVectors for splitting into: 0, 2, 1, 3
    val bitVector2 = new BitSet(numRows)
    bitVector2.set(2) // 2 goes to the right
    bitVector2.set(3) // 3 goes to the right
    val activeNodeMap2 = new JavaHashMap[Int, Int]()
    activeNodeMap2.put(0, 0)
    activeNodeMap2.put(1, 1)
    activeNodeMap2.put(2, 2)
    activeNodeMap2.put(3, 3)

    val newInfo2 = newInfo.update(bitVector2, activeNodeMap2, labels, metadata)

    val impurityAgg1 = metadata.createImpurityAggregator()
    impurityAgg1.update(labels(0))
    val impurityAgg2 = metadata.createImpurityAggregator()
    impurityAgg2.update(labels(2))
    val impurityAgg3 = metadata.createImpurityAggregator()
    impurityAgg3.update(labels(1))
    val impurityAgg4 = metadata.createImpurityAggregator()
    impurityAgg4.update(labels(3))

    val expectedFullImpurityAggs2 = Array(impurityAgg1, impurityAgg2, impurityAgg3, impurityAgg4)

    assert(newInfo2.columns.length === 2)
    val expectedCol2a = new FeatureVector(0, 0, Array(0.1, 0.2, 0.6, 0.8), Array(2, 1, 3, 0),
      expectedFullImpurityAggs2, Array(1, 2, 3, 0))
    val expectedCol2b = new FeatureVector(1, 3, Array(0, 0, 1, 2), Array(0, 2, 1, 3),
      expectedFullImpurityAggs2, Array(0, 1, 2, 3))
    assert(newInfo2.columns(0) === expectedCol2a)
    assert(newInfo2.columns(1) === expectedCol2b)
  }

//  /* * * * * * * * * * * Choosing Splits  * * * * * * * * * * */
//
//  test("computeBestSplits") {
//    // TODO
//  }
//
//  test("chooseSplit: choose correct type of split") {
//    val labels = Array(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
//    val labelsAsBytes = labels.map(_.toByte)
//    val fromOffset = 1
//    val toOffset = 4
//    val impurity = Entropy
//    val metadata = new AltDTMetadata(numClasses = 2, maxBins = 4, minInfoGain = 0.0, impurity, Map(1 -> 3))
//    val fullImpurityAgg = metadata.createImpurityAggregator()
//    labels.foreach(label => fullImpurityAgg.update(label))
//
//    val col1 = FeatureVector.fromOriginal(featureIndex = 0, featureArity = 0,
//      values = Array(0.8, 0.1, 0.1, 0.2, 0.3, 0.5, 0.6))
//    val (split1, _) = AltDTClassification.chooseSplit(col1, labelsAsBytes, fromOffset, toOffset, fullImpurityAgg, metadata)
//    assert(split1.nonEmpty && split1.get.isInstanceOf[ContinuousSplit])
//
//    val col2 = FeatureVector.fromOriginal(featureIndex = 1, featureArity = 3,
//      values = Array(0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0))
//    val (split2, _) = AltDTRegression.chooseSplit(col2, labels, fromOffset, toOffset, fullImpurityAgg, metadata)
//    assert(split2.nonEmpty && split2.get.isInstanceOf[CategoricalSplit])
//  }
//
//  test("chooseOrderedCategoricalSplit: basic case") {
//    val featureIndex = 0
//    val values = Array(0, 0, 1, 2, 2, 2, 2).map(_.toDouble)
//    val featureArity = values.max.toInt + 1
//
//    def testHelper(
//        labels: Array[Byte],
//        expectedLeftCategories: Array[Double],
//        expectedLeftStats: Array[Double],
//        expectedRightStats: Array[Double]): Unit = {
//      val expectedRightCategories = Range(0, featureArity)
//        .filter(c => !expectedLeftCategories.contains(c)).map(_.toDouble).toArray
//      val impurity = Entropy
//      val metadata = new AltDTMetadata(numClasses = 2, maxBins = 4, minInfoGain = 0.0,
//        impurity, Map.empty[Int, Int])
//      val (split, stats) =
//        AltDTClassification.chooseOrderedCategoricalSplit(featureIndex, values, values.indices.toArray,
//          labels, 0, values.length, metadata, featureArity)
//      split match {
//        case Some(s: CategoricalSplit) =>
//          assert(s.featureIndex === featureIndex)
//          assert(s.leftCategories === expectedLeftCategories)
//          assert(s.rightCategories === expectedRightCategories)
//        case _ =>
//          throw new AssertionError(
//            s"Expected CategoricalSplit but got ${split.getClass.getSimpleName}")
//      }
//      val fullImpurityStatsArray =
//        Array(labels.count(_ == 0.0).toDouble, labels.count(_ == 1.0).toDouble)
//      val fullImpurity = impurity.calculate(fullImpurityStatsArray, labels.length)
//      assert(stats.gain === fullImpurity)
//      assert(stats.impurity === fullImpurity)
//      assert(stats.impurityCalculator.stats === fullImpurityStatsArray)
//      assert(stats.leftImpurityCalculator.stats === expectedLeftStats)
//      assert(stats.rightImpurityCalculator.stats === expectedRightStats)
//      assert(stats.valid)
//    }
//
//    val labels1 = Array(0, 0, 1, 1, 1, 1, 1).map(_.toByte)
//    testHelper(labels1, Array(0.0), Array(2.0, 0.0), Array(0.0, 5.0))
//
//    val labels2 = Array(0, 0, 0, 1, 1, 1, 1).map(_.toByte)
//    testHelper(labels2, Array(0.0, 1.0), Array(3.0, 0.0), Array(0.0, 4.0))
//  }
//
//  test("chooseOrderedCategoricalSplit: return bad split if we should not split") {
//    val featureIndex = 0
//    val values = Array(0, 0, 1, 2, 2, 2, 2).map(_.toDouble)
//    val featureArity = values.max.toInt + 1
//
//    val labels = Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
//
//    val impurity = Entropy
//    val metadata = new AltDTMetadata(numClasses = 2, maxBins = 4, minInfoGain = 0.0, impurity,
//      Map(featureIndex -> featureArity))
//    val (split, stats) =
//      AltDTRegression.chooseOrderedCategoricalSplit(featureIndex, values, values.indices.toArray,
//        labels, 0, values.length, metadata, featureArity)
//    assert(split.isEmpty)
//    val fullImpurityStatsArray =
//      Array(labels.count(_ == 0.0).toDouble, labels.count(_ == 1.0).toDouble)
//    val fullImpurity = impurity.calculate(fullImpurityStatsArray, labels.length)
//    assert(stats.gain === 0.0)
//    assert(stats.impurity === fullImpurity)
//    assert(stats.impurityCalculator.stats === fullImpurityStatsArray)
//    assert(stats.valid)
//  }
//
//  test("chooseUnorderedCategoricalSplit: basic case") {
//    val featureIndex = 0
//    val featureArity = 4
//    val values = Array(3.0, 1.0, 0.0, 2.0, 2.0)
//    val labels = Array(0.0, 0.0, 1.0, 1.0, 2.0)
//    val impurity = Entropy
//    val metadata = new AltDTMetadata(numClasses = 3, maxBins = 16, minInfoGain = 0.0, impurity,
//      Map(featureIndex -> featureArity))
//    val allSplits = metadata.getUnorderedSplits(featureIndex)
//    val (split, _) = AltDTRegression.chooseUnorderedCategoricalSplit(featureIndex, values, values.indices.toArray,
//      labels, 0, values.length, metadata, featureArity, allSplits)
//    split match {
//      case Some(s: CategoricalSplit) =>
//        assert(s.featureIndex === featureIndex)
//        assert(s.leftCategories.toSet === Set(0.0, 2.0))
//        assert(s.rightCategories.toSet === Set(1.0, 3.0))
//        // TODO: test correctness of stats
//      case _ =>
//        throw new AssertionError(
//          s"Expected CategoricalSplit but got ${split.getClass.getSimpleName}")
//    }
//  }
//
//  test("chooseUnorderedCategoricalSplit: return bad split if we should not split") {
//    val featureIndex = 0
//    val featureArity = 4
//    val values = Array(3.0, 1.0, 0.0, 2.0, 2.0)
//    val labels = Array(1.0, 1.0, 1.0, 1.0, 1.0).map(_.toByte)
//    val impurity = Entropy
//    val metadata = new AltDTMetadata(numClasses = 2, maxBins = 4, minInfoGain = 0.0, impurity,
//      Map(featureIndex -> featureArity))
//    val (split, stats) =
//      AltDTClassification.chooseOrderedCategoricalSplit(featureIndex, values, values.indices.toArray,
//        labels, 0, values.length, metadata, featureArity)
//    assert(split.isEmpty)
//    val fullImpurityStatsArray =
//      Array(labels.count(_ == 0.0).toDouble, labels.count(_ == 1.0).toDouble)
//    val fullImpurity = impurity.calculate(fullImpurityStatsArray, labels.length)
//    assert(stats.gain === 0.0)
//    assert(stats.impurity === fullImpurity)
//    assert(stats.impurityCalculator.stats === fullImpurityStatsArray)
//    assert(stats.valid)
//  }

  test("chooseContinuousSplit: basic case") {
    val featureIndex = 0
    val values = Array(0.1, 0.2, 0.3, 0.4, 0.5)
    val labels = Array(0.0, 0.0, 1.0, 1.0, 1.0)
    val nodeIndices = Array(0, 0, 0, 0, 0)
    val impurity = Entropy
    val metadata = new AltDTMetadata(numClasses = 2, maxBins = 4, minInfoGain = 0.0, impurity, Map.empty[Int, Int])
    val fullImpurityAgg = metadata.createImpurityAggregator()
    labels.foreach(label => fullImpurityAgg.update(label))

    val splitsAndStats = AltDTRegression.chooseContinuousSplitsForActiveNodes(featureIndex, 1, values,
      values.indices.toArray, labels, nodeIndices, Array(fullImpurityAgg), metadata)
    val (splits, statsArr) = splitsAndStats.unzip
    assert(splits.length == 1)
    val split = splits(0)
    split match {
      case Some(s: ContinuousSplit) =>
        assert(s.featureIndex === featureIndex)
        assert(s.threshold === 0.2)
      case _ =>
        throw new AssertionError(
          s"Expected ContinuousSplit but got ${split.getClass.getSimpleName}")
    }
    val stats = statsArr(0)

    val fullImpurityStatsArray =
      Array(labels.count(_ == 0.0).toDouble, labels.count(_ == 1.0).toDouble)
    val fullImpurity = impurity.calculate(fullImpurityStatsArray, labels.length)
    assert(stats.gain === fullImpurity)
    assert(stats.impurity === fullImpurity)
    assert(stats.impurityCalculator.stats === fullImpurityStatsArray)
    assert(stats.leftImpurityCalculator.stats === Array(2.0, 0.0))
    assert(stats.rightImpurityCalculator.stats === Array(0.0, 3.0))
    assert(stats.valid)
  }

  test("chooseContinuousSplitSparse: basic case") {
    val featureIndex = 0
    val values = Array(0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.5, 0.5)
    val compressedValues = FeatureVector.runLengthEncoding(values)
    val labels = Array(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0).map(_.toByte)
    val nodeIndices = Array(0, 0, 0, 0, 0, 0, 0, 0)
    val impurity = Gini
    val metadata = new AltDTMetadata(numClasses = 2, maxBins = 4, minInfoGain = 0.0, impurity, Map.empty[Int, Int])
    val fullImpurityAgg = metadata.createImpurityAggregator()
    labels.foreach(label => fullImpurityAgg.update(label))

    val splitsAndStats = AltDTClassification.chooseContinuousSplitsSparseForActiveNodes(featureIndex, 1,
      compressedValues, labels.indices.toArray, labels, nodeIndices, Array(fullImpurityAgg), metadata)
    val (splits, statsArr) = splitsAndStats.unzip
    assert(splits.length == 1)
    val split = splits(0)
    split match {
      case Some(s: ContinuousSplit) =>
        assert(s.featureIndex === featureIndex)
        assert(s.threshold === 0.3)
      case _ =>
        throw new AssertionError(
          s"Expected ContinuousSplit but got ${split.getClass.getSimpleName}")
    }
    val stats = statsArr(0)

    val fullImpurityStatsArray =
      Array(labels.count(_ == 0.0).toDouble, labels.count(_ == 1.0).toDouble)
    val fullImpurity = impurity.calculate(fullImpurityStatsArray, labels.length)
    assert(stats.gain === fullImpurity)
    assert(stats.impurity === fullImpurity)
    assert(stats.impurityCalculator.stats === fullImpurityStatsArray)
    assert(stats.rightImpurityCalculator.stats === Array(0.0, 3.0))
    assert(stats.leftImpurityCalculator.stats === Array(5.0, 0.0))
    assert(stats.valid)
  }

  test("chooseContinuousSplit: return bad split if we should not split") {
    val featureIndex = 0
    val values = Array(0.1, 0.2, 0.3, 0.4, 0.5)
    val labels = Array(0.0, 0.0, 0.0, 0.0, 0.0).map(_.toByte)
    val nodeIndices = Array(0, 0, 0, 0, 0)
    val impurity = Entropy
    val metadata = new AltDTMetadata(numClasses = 2, maxBins = 4, minInfoGain = 0.0, impurity, Map.empty[Int, Int])
    val fullImpurityAgg = metadata.createImpurityAggregator()
    labels.foreach(label => fullImpurityAgg.update(label))

    val splitsAndStats = AltDTClassification.chooseContinuousSplitsForActiveNodes(featureIndex, 1, values,
      values.indices.toArray, labels, nodeIndices, Array(fullImpurityAgg), metadata)
    val (splits, statsArr) = splitsAndStats.unzip
    assert(splits.length == 1)

    val split = splits(0)
    // split should be None
    assert(split.isEmpty)

    // stats for parent node should be correct
    val stats = statsArr(0)
    val fullImpurityStatsArray =
      Array(labels.count(_ == 0.0).toDouble, labels.count(_ == 1.0).toDouble)
    val fullImpurity = impurity.calculate(fullImpurityStatsArray, labels.length)
    assert(stats.gain === 0.0)
    assert(stats.impurity === fullImpurity)
    assert(stats.impurityCalculator.stats === fullImpurityStatsArray)
  }

  /* * * * * * * * * * * Bit subvectors * * * * * * * * * * */

  test("bitSubvectorFromSplit: 1 node") {
    val col =
      FeatureVector.fromOriginal(0, 0, Array(0.1, 0.2, 0.4, 0.6, 0.7), null)
    val split = new ContinuousSplit(0, threshold = 0.5)
    val bitv = AltDT.bitVectorFromSplit(col, 0, split)
    assert(bitv.toArray.toSet === Set(3, 4))
  }

  test("bitSubvectorFromSplit: 2 nodes") {
    // Initially, 1 split: (0, 2, 4) | (1, 3)
    val col = new FeatureVector(0, 0, Array(0.1, 0.2, 0.4, 0.6, 0.7), Array(4, 2, 0, 1, 3),
      null, Array(0, 0, 0, 1, 1))

    def checkSplit(nodeIndex: Int, threshold: Double,
      expectedRight: Set[Int]): Unit = {
        val split = new ContinuousSplit(0, threshold)
        val bitv = AltDT.bitVectorFromSplit(col, nodeIndex, split)
        assert(bitv.toArray.toSet === expectedRight)
    }

    // Left child node
    checkSplit(0, 0.05, Set(0, 2, 4))
    checkSplit(0, 0.15, Set(0, 2))
    checkSplit(0, 0.2, Set(0))
    checkSplit(0, 0.5, Set())
    // Right child node
    checkSplit(1, 0.1, Set(1, 3))
    checkSplit(1, 0.65, Set(3))
    checkSplit(1, 0.8, Set())
  }

  test("collectBitVectors with 1 vector") {
    val labels = Array(0, 0, 0, 1, 1, 1, 1).map(_.toDouble)
    val metadata = new AltDTMetadata(numClasses = 2, maxBins = 4, minInfoGain = 0.0, Entropy, Map(1 -> 3))
    val fullImpurityAgg = metadata.createImpurityAggregator()
    labels.foreach(label => fullImpurityAgg.update(label))
    val col = FeatureVector.fromOriginal(0, 0, Array(0.1, 0.2, 0.4, 0.6, 0.7), fullImpurityAgg)

    val info = PartitionInfo(Array(col))
    val partitionInfos = sc.parallelize(Seq(info))
    val bestSplit = new ContinuousSplit(0, threshold = 0.5)
    val bitVector = AltDT.aggregateBitVector(partitionInfos, Array(Some(bestSplit)))
    assert(bitVector.toArray.toSet === Set(3, 4))
  }

  test("collectBitVectors with 1 vector, with tied threshold") {
    val labels = Array(0, 0, 0, 1, 1, 1, 1, 1).map(_.toDouble)
    val metadata = new AltDTMetadata(numClasses = 2, maxBins = 4, minInfoGain = 0.0, Entropy, Map(1 -> 3))
    val fullImpurityAgg = metadata.createImpurityAggregator()
    labels.foreach(label => fullImpurityAgg.update(label))
    val col = new FeatureVector(0, 0,
      Array(-4.0, -4.0, -2.0, -2.0, -1.0, -1.0, 1.0, 1.0),
      Array(3, 7, 2, 6, 1, 5, 0, 4), Array(fullImpurityAgg),
    new Array[Int](8))

    val info = PartitionInfo(Array(col))
    val partitionInfos = sc.parallelize(Seq(info))
    val bestSplit = new ContinuousSplit(0, threshold = -2.0)
    val bitVector = AltDT.aggregateBitVector(partitionInfos, Array(Some(bestSplit)))
    assert(bitVector.toArray.toSet === Set(0, 1, 4, 5))
  }

  /* * * * * * * * * * * Active nodes * * * * * * * * * * */

  test("computeActiveNodePeriphery") {
    // old periphery: 2 nodes
    val left = LearningNode.emptyNode(id = 1)
    val right = LearningNode.emptyNode(id = 2)
    val oldPeriphery: Array[LearningNode] = Array(left, right)
    // bestSplitsAndGains: Do not split left, but split right node.
    val lCalc = new EntropyCalculator(Array(8.0, 1.0))
    val lStats = new ImpurityStats(0.0, lCalc.calculate(),
      lCalc, lCalc, new EntropyCalculator(Array(0.0, 0.0)))

    val rSplit: Split = new ContinuousSplit(featureIndex = 1, threshold = 0.6)
    val rCalc = new EntropyCalculator(Array(5.0, 7.0))
    val rRightChildCalc = new EntropyCalculator(Array(1.0, 5.0))
    val rLeftChildCalc = new EntropyCalculator(Array(
      rCalc.stats(0) - rRightChildCalc.stats(0),
      rCalc.stats(1) - rRightChildCalc.stats(1)))
    val rGain = {
      val rightWeight = rRightChildCalc.stats.sum / rCalc.stats.sum
      val leftWeight = rLeftChildCalc.stats.sum / rCalc.stats.sum
      rCalc.calculate() -
        rightWeight * rRightChildCalc.calculate() - leftWeight * rLeftChildCalc.calculate()
    }
    val rStats =
      new ImpurityStats(rGain, rCalc.calculate(), rCalc, rLeftChildCalc, rRightChildCalc)

    val bestSplitsAndGains: Array[(Option[Split], ImpurityStats)] =
      Array((None, lStats), (Some(rSplit), rStats))

    // Test A: Split right node
    val newPeriphery1: Array[LearningNode] =
      AltDT.computeActiveNodePeriphery(oldPeriphery, bestSplitsAndGains, minInfoGain = 0.0)
    // Expect 2 active nodes
    assert(newPeriphery1.length === 2)
    // Confirm right node was updated
    assert(right.split.get === rSplit)
    assert(!right.isLeaf)
    assert(right.stats.exactlyEquals(rStats))
    assert(right.leftChild.nonEmpty && right.leftChild.get === newPeriphery1(0))
    assert(right.rightChild.nonEmpty && right.rightChild.get === newPeriphery1(1))
    // Confirm new active nodes have stats but no children
    assert(newPeriphery1(0).leftChild.isEmpty && newPeriphery1(0).rightChild.isEmpty &&
      newPeriphery1(0).split.isEmpty &&
      newPeriphery1(0).stats.impurityCalculator.exactlyEquals(rLeftChildCalc))
    assert(newPeriphery1(1).leftChild.isEmpty && newPeriphery1(1).rightChild.isEmpty &&
      newPeriphery1(1).split.isEmpty &&
      newPeriphery1(1).stats.impurityCalculator.exactlyEquals(rRightChildCalc))

    // Test B: Increase minInfoGain, so split nothing
    val newPeriphery2: Array[LearningNode] =
      AltDT.computeActiveNodePeriphery(oldPeriphery, bestSplitsAndGains, minInfoGain = 1000.0)
    assert(newPeriphery2.isEmpty)
  }
}
