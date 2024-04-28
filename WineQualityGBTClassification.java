import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.functions; 

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.io.IOException;


public class WineQualityGBTClassification {

    public static void main(String[] args) throws IOException {
        SparkSession spark = SparkSession.builder().appName("GBT Training").getOrCreate();

        // Define schema for CSV data
        List<StructField> fields = Arrays.asList(
                DataTypes.createStructField("fixed_acidity", DataTypes.DoubleType, true),
                DataTypes.createStructField("volatile_acidity", DataTypes.DoubleType, true),
                DataTypes.createStructField("citric_acid", DataTypes.DoubleType, true),
                DataTypes.createStructField("residual_sugar", DataTypes.DoubleType, true),
                DataTypes.createStructField("chlorides", DataTypes.DoubleType, true),
                DataTypes.createStructField("free_sulfur_dioxide", DataTypes.DoubleType, true),
                DataTypes.createStructField("total_sulfur_dioxide", DataTypes.DoubleType, true),
                DataTypes.createStructField("density", DataTypes.DoubleType, true),
                DataTypes.createStructField("pH", DataTypes.DoubleType, true),
                DataTypes.createStructField("sulphates", DataTypes.DoubleType, true),
                DataTypes.createStructField("alcohol", DataTypes.DoubleType, true),
                DataTypes.createStructField("quality", DataTypes.DoubleType, true)
        );
        StructType wineSchema = DataTypes.createStructType(fields);

         // Load and prepare data
        Dataset<Row> wineData = spark.read()
                .format("csv")
                .schema(wineSchema)
                .option("header", true)
                .option("delimiter", ";")
                .option("quote", "\"")
                .option("ignoreLeadingWhiteSpace", true)
                .option("ignoreTrailingWhiteSpace", true)
                .load("file:///home/ec2-user/TrainingDataset.csv");

        // Remove quotes from column names 
        wineData = wineData.toDF(Stream.of(wineData.columns())
                .map(col -> col.replaceAll("\"", ""))
                .collect(Collectors.toList())
                .toArray(String[]::new));

        // Convert quality column to binary
        wineData = wineData.withColumn("quality", 
                functions.when(wineData.col("quality").gt(7), 1.0).otherwise(0.0));

        // Create feature assembler
        String[] featureCols = wineData.columns();
        featureCols = Arrays.copyOf(featureCols, featureCols.length - 1);
        VectorAssembler vectorizer = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");
        wineData = vectorizer.transform(wineData);

        // Split data into training and testing
        Dataset<Row>[] splits = wineData.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> training = splits[0];
        Dataset<Row> testing = splits[1];

        // Create and train Gradient-Boosted Trees (GBT) classifier
        GBTClassifier gbtClassifier = new GBTClassifier()
                .setLabelCol("quality")
                .setFeaturesCol("features")
                .setMaxIter(100);
        GBTClassificationModel trainedModel = gbtClassifier.fit(training);

        // Make predictions and evaluate
        Dataset<Row> predictions = trainedModel.transform(testing);
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double f1 = evaluator.evaluate(predictions);
        System.out.println("F1 Score: " + f1);

        // Save the model
        trainedModel.write().overwrite().save("file:///home/ec2-user/modelweights");
    }
}
