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

package org.apache.spark.examples.mllib

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.GaussianMixtureModelEM
import org.apache.spark.mllib.linalg.Vectors

/**
 * An example Gaussian Mixture Model EM app. Run with
 * {{{
 * ./bin/run-example org.apache.spark.examples.mllib.DenseGmmEM <input> <k> <covergenceTol>
 * }}}
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object DenseGmmEM {
  def main(args: Array[String]): Unit = {
    if (args.length != 3) {
      println("usage: DenseGmmEM <input file> <k> <convergenceTol>")
    } else {
      run(args(0), args(1).toInt, args(2).toDouble)
    }
  }

  def run(inputFile: String, k: Int, convergenceTol: Double) {
    val conf = new SparkConf().setAppName("Spark EM Sample")
    val ctx  = new SparkContext(conf)
    
    val data = ctx.textFile(inputFile).map{ line =>
      Vectors.dense(line.trim.split(' ').map(_.toDouble))
    }.cache
      
    val clusters = new GaussianMixtureModelEM()
      .setK(k)
      .setConvergenceTol(convergenceTol)
      .run(data)
    
    for (i <- 0 until clusters.k) {
      println("weight=%f\nmu=%s\nsigma=\n%s\n" format 
        (clusters.weight(i), clusters.mu(i), clusters.sigma(i)))
    }
    
    println("Cluster labels:")
    val clusterLabels = clusters.predict(data)
    for (x <- clusterLabels.collect) {
      print(" " + x)
    }
    println
  }
}
