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

package org.apache.spark.mllib.regression

import org.scalatest.FunSuite

import org.apache.spark.mllib.impl.TreeTests
import TreeTests.checkEqual
import org.apache.spark.mllib.tree.{DecisionTree => OldDecisionTree,
  DecisionTreeSuite => OldDecisionTreeSuite}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.rdd.RDD

import DecisionTreeRegressorSuite.compareAPIs

class DecisionTreeRegressorSuite extends FunSuite with MLlibTestSparkContext {

  private var categoricalDataPointsRDD: RDD[LabeledPoint] = _

  override def beforeAll() {
    super.beforeAll()
    categoricalDataPointsRDD =
      sc.parallelize(OldDecisionTreeSuite.generateCategoricalDataPoints())
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests calling train()
  /////////////////////////////////////////////////////////////////////////////

  test("Regression stump with 3-ary (ordered) categorical features") {
    val dt = new DecisionTreeRegressor()
      .setImpurity("variance")
      .setMaxDepth(2)
      .setMaxBins(100)
    val categoricalFeatures = Map(0 -> 3, 1-> 3)
    compareAPIs(categoricalDataPointsRDD, dt, categoricalFeatures)
  }

  test("Regression stump with binary (ordered) categorical features") {
    val dt = new DecisionTreeRegressor()
      .setImpurity("variance")
      .setMaxDepth(2)
      .setMaxBins(100)
    val categoricalFeatures = Map(0 -> 2, 1-> 2)
    compareAPIs(categoricalDataPointsRDD, dt, categoricalFeatures)
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests of model save/load
  /////////////////////////////////////////////////////////////////////////////

  // TODO: test("model save/load")
}

private[mllib] object DecisionTreeRegressorSuite extends FunSuite {

  /**
   * Train 2 decision trees on the given dataset, one using the old API and one using the new API.
   * Convert the old tree to the new format, compare them, and fail if they are not exactly equal.
   */
  def compareAPIs(
      data: RDD[LabeledPoint],
      dt: DecisionTreeRegressor,
      categoricalFeatures: Map[Int, Int]): Unit = {
    val oldStrategy = dt.getOldStrategy(categoricalFeatures)
    val oldTree = OldDecisionTree.train(data, oldStrategy)
    val newTree = dt.run(data, categoricalFeatures)
    val oldTreeAsNew = DecisionTreeRegressionModel.fromOld(oldTree)
    checkEqual(oldTreeAsNew, newTree)
  }
}
