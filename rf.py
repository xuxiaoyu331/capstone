#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:53:17 2020

@author: irene
"""

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

conf = SparkConf().setAppName('App').setMaster('local[2]')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

df = sqlContext.read.format('com.databricks.spark.csv') \
                   .options(header='true', inferschema='true') \
                   .load('/Users/irene/Desktop/project/data/corona_clean.csv')


regexTokenizer = RegexTokenizer(gaps = False, pattern = '\w+', inputCol = 'text', outputCol = 'text_token')
df_token = regexTokenizer.transform(df)

swr = StopWordsRemover(inputCol = 'text_token', outputCol = 'text_sw_removed')
df_swr = swr.transform(df_token)

word2vec = Word2Vec(vectorSize = 100, minCount = 3, inputCol = 'text_sw_removed', outputCol = 'features')
w2v_model = word2vec.fit(df_swr)
df_result = w2v_model.transform(df_swr)

si = StringIndexer(inputCol="sentiment", outputCol="label")
df_si = si.transform(df_result)

train_df, test_df = df_result.randomSplit([0.8, 0.2])

##### random forest#####
rf_classifier = RandomForestClassifier(labelCol="label", featuresCol="features")
rf_model = rf_classifier.fit(train_df)
rf_prediction = rf_model.transform(test_df)

rf_model_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = rf_model_evaluator.evaluate(rf_predictions)

print("Accuracy = %g" % (accuracy))

sc.stop()



