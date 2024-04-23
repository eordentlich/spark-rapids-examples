#
# Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from spark_rapids_ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.classification import RandomForestClassifier as SparkRandomForestClassifier, RandomForestClassificationModel as SparkRandomForestClassificationModel
from pyspark.ml.feature import StandardScaler

from .consts import *
from com.nvidia.spark.examples.utility.utils import *
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

estimator_cls = RandomForestClassifier
def main(args, ml_args):
    spark = (SparkSession
             .builder
             .appName(args.mainClass)
             .getOrCreate())

    train_data, eval_data, trans_data = valid_input_data(spark, args, '', schema)

    if args.mode in ['all', 'train']:
        if train_data is None:
            print('-' * 80)
            print('Usage: training data path required when mode is all or train')
            exit(1)

        train_data, features = transform_data(train_data, label, args.use_gpu)
        ml_args['features_col'] = features
        ml_args['label_col'] = label

        estimator, model_cls = ( 
            (RandomForestClassifier(verbose=7), RandomForestClassificationModel) if args.use_gpu 
                else (SparkRandomForestClassifier(), SparkRandomForestClassificationModel)
        )
        print(f'Estimator class: {estimator.__class__}')

        classifier = ( 
                estimator.setFeaturesCol(features)
                         .setLabelCol(label)
        )

        numTrees = args.numTrees
        maxDepth = args.maxDepth
        maxBins = args.maxBins
        if max(len(numTrees), len(maxDepth), len(maxBins)) > 1:
            # cross validation
            from pyspark.ml.evaluation import MulticlassClassificationEvaluator
            from spark_rapids_ml.tuning import CrossValidator
            from pyspark.ml.tuning import CrossValidator as SparkCrossValidator, ParamGridBuilder

            eval = MulticlassClassificationEvaluator(metricName='logLoss', labelCol=label)
            grid = ( 
                ParamGridBuilder().addGrid(classifier.numTrees, numTrees)
                                    .addGrid(classifier.maxDepth, maxDepth)
                                    .addGrid(classifier.maxBins, maxBins)
                                    .build()
            )
            cv_cls = CrossValidator if args.use_gpu else SparkCrossValidator

            cv = cv_cls(estimator=classifier, estimatorParamMaps=grid, evaluator=eval, parallelism=1, seed=1)

            model = with_benchmark('Training CV', lambda: cv.fit(train_data))
            print(f"average metrics: {model.avgMetrics}")
        else:
            classifier = ( 
                classifier.setRegParam(numTrees[0])
                            .setElasticNetParam(maxDepth[0])
                            .setMaxIter(maxBins[0])
            )
        
            if eval_data:
                # TODO
                pass


            model = with_benchmark('Training', lambda: classifier.fit(train_data))
            if not args.use_gpu:
                print(f"iterations: {model.summary.totalIterations}")
                print(f"objective history: {model.summary.objectiveHistory}")

        if args.modelPath:
            writer = model.write().overwrite() if args.overwrite else model
            writer.save(args.modelPath)
    else:
        model = model_cls.load(args.modelPath)

    if args.mode in ['all', 'transform']:
        trans_data, _ = transform_data(trans_data, label, args.use_gpu)

        def transform():
            prob_col = model.getProbabilityCol()
            prediction_col = model.getPredictionCol()
            raw_col = model.getRawPredictionCol()
            result = model.transform(trans_data).select(label, prob_col, prediction_col, raw_col)
            result.select(F.sum(prediction_col)).collect()
            return result

        if not trans_data:
            print('-' * 80)
            print('Usage: trans data path required when mode is all or transform')
            exit(1)

        result = with_benchmark('Transformation', transform)
        show_sample(args, result, label)
        with_benchmark('Evaluation', lambda: check_classification_accuracy(result, label))

    spark.stop()
