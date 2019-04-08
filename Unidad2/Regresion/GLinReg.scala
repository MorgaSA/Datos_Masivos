import org.apache.spark.ml.regression.GeneralizedLinearRegression

// Load training dataetRegParam(
val dataset = spark.read.format("libsvm").load("/home/alex/spark/data/mllib/sample_linear_regression_data.txt")

val glr = new GeneralizedLinearRegression().setFamily("gaussian").setLink("identity").setMaxIter(10).setRegParam(0.3)

// Fit the model
val model = glr.fit(dataset)

// Print the coefficients and intercept for generalized linear regression model
println(s"Coefficients: ${model.coefficients}")
println(s"Intercept: ${model.intercept}")

// Sumarize the model over the training set and print out some metrics
val smmary = model.summary
printn(s"Coefficient Standard Errors: ${summary.coefficientStandardErrors.mkString(",")}")
printn(s"T Values: ${summary.tValues.mkString(",")}")
printn(s"P Values: ${summary.pValues.mkString(",")}")
printn(s"Dispersion: ${summary.dispersion}")

printn(s"Null Deviance: ${summary.nullDeviance}")
printn(s"Residual Degree Of Freedom Null: ${summary.residualDegreeOfFreedomNull}")
printn(s"Deviance: ${summary.deviance}")
printn(s"Residual Degree Of Freedom: ${summary.residualDegreeOfFreedom}")
printn(s"AIC: ${summary.aic}")
printn("Deviance Residuals: ")
summay.residuals().show()