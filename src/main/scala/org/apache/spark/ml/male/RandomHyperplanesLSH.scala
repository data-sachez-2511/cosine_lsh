package org.apache.spark.ml.male

import breeze.linalg.{norm, sum}
import breeze.numerics.{acos, signum}
import org.apache.spark.ml.feature.{LSH, LSHModel}
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.{DoubleParam, ParamMap, ParamValidators, Params}
import org.apache.spark.ml.param.shared.HasSeed
import org.apache.spark.ml.util.{Identifiable, MLWriter, SchemaUtils}
import org.apache.spark.sql.types.StructType

import scala.util.Random

class RandomHyperplanesLSHModel private[ml](
   override val uid: String,
   private[ml] val randUnitVectors: Array[Vector])
   extends LSHModel[RandomHyperplanesLSHModel] with HasSeed{
  override protected[ml] def hashFunction(elems: Vector): Array[Vector] = {
    val hashValues = randUnitVectors.map(
      randUnitVector => signum(BLAS.dot(randUnitVector, elems))
    )
    hashValues.map(Vectors.dense(_))
  }

  override def setInputCol(value: String): this.type = super.set(inputCol, value)

  override def setOutputCol(value: String): this.type = super.set(outputCol, value)

  override protected[ml] def keyDistance(x: Vector, y: Vector): Double = {
    1 - BLAS.dot(x,y) / norm(x.asBreeze, 2) / norm(y.asBreeze, 2)
  }

  override protected[ml] def hashDistance(x: Seq[Vector], y: Seq[Vector]): Double = {
    x.zip(y).map(vectorPair => sum(vectorPair._1.asBreeze * vectorPair._2.asBreeze)/randUnitVectors.length).max
  }

  override def copy(extra: ParamMap): RandomHyperplanesLSHModel = {
    val copied = new RandomHyperplanesLSHModel(uid, randUnitVectors).setParent(parent)
    copyValues(copied, extra)
  }

  override def write: MLWriter = ???
}

class RandomHyperplanesLSH(override val uid: String)
  extends LSH[RandomHyperplanesLSHModel] with HasSeed {

  def this() = {
    this(Identifiable.randomUID("rh-lsh"))
  }

  override def setNumHashTables(value: Int): this.type = super.setNumHashTables(value)

  def setSeed(value: Long): this.type = set(seed, value)

  override def createRawLSHModel(inputDim: Int): RandomHyperplanesLSHModel = {
    val rand = new Random($(seed))
    val randUnitVectors: Array[Vector] = {
      Array.fill($(numHashTables)) {
        val randArray = Array.fill(inputDim)(rand.nextInt().abs % 2)
        Vectors.dense(randArray.map(x => 2 * x - 1).map(_.toDouble))
      }
    }
    new RandomHyperplanesLSHModel(uid, randUnitVectors)
  }

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(inputCol), new VectorUDT)
    validateAndTransformSchema(schema)
  }
}
