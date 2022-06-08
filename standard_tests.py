import json
import os
import re
import traceback
import warnings
from pathlib import Path

import great_expectations as ge
import modelop.monitors.drift as drift
import modelop.monitors.performance as performance
import modelop.monitors.stability as stability
import modelop.schema.infer as infer
import numpy
import pandas as pd
from ydata_quality import DataQuality

INPUT_SCHEMA = {}
ADDITIONAL_ASSETS = []
DEPLOYABLE_MODEL = {}
GREAT_EXPECTATIONS_RESULT_FILE = None


# modelop.init
def init(init_param):
	global INPUT_SCHEMA
	global ADDITIONAL_ASSETS
	global DEPLOYABLE_MODEL

	job = json.loads(init_param["rawJson"])
	INPUT_SCHEMA = infer.extract_input_schema(init_param)
	ADDITIONAL_ASSETS = job.get('additionalAssets', [])
	DEPLOYABLE_MODEL = job.get('referenceModel', {})


# modelop.metrics
def metrics(baseline, comparator) -> dict:
	global INPUT_SCHEMA
	global DEPLOYABLE_MODEL

	warnings.filterwarnings(action="ignore", category=numpy.VisibleDeprecationWarning)
	stability_result = calculate_stability(baseline, comparator, INPUT_SCHEMA)
	performance_result = calculate_regression_performance(comparator, INPUT_SCHEMA)
	drift_result = calculate_drift(baseline, comparator)
	ge_result = run_great_expectations()
	dq_result = run_data_quality(baseline, INPUT_SCHEMA, ADDITIONAL_ASSETS)

	result = {}
	result.update(stability_result)
	result.update(performance_result)
	result.update(drift_result)
	result.update(ge_result)
	result.update(dq_result)
	result.update({'modelUseCategory': DEPLOYABLE_MODEL.get('storedModel', {}).get('modelMetaData', {}).get('modelUseCategory', ''),
				   'modelOrganization': DEPLOYABLE_MODEL.get('storedModel', {}).get('modelMetaData', {}).get('modelOrganization', ''),
				   'modelRisk': DEPLOYABLE_MODEL.get('storedModel', {}).get('modelMetaData', {}).get('modelRisk', ''),
				   'modelMethodology': DEPLOYABLE_MODEL.get('storedModel', {}).get('modelMetaData', {}).get('modelMethodology', '')})
	yield result


def calculate_regression_performance(baseline, schema) -> dict:
	try:
		monitoring_parameters = infer.set_monitoring_parameters(schema, check_schema=True)
		performance_monitor = performance.ModelEvaluator(dataframe=baseline,
														 score_column=monitoring_parameters["score_column"],
														 label_column=monitoring_parameters["label_column"])
		performance_result =  performance_monitor.evaluate_performance(pre_defined_metrics='regression_metrics')
		return {"performance": [ performance_result ],
				"mae": performance_result.get("values", {}).get("mae", None),
				"r2_score": performance_result.get("values", {}).get("r2_score"),
				"rmse": performance_result.get("values", {}).get("rmse")}
	except Exception as ex:
		print("Error occurred cacluating performance metrics")
		print(ex)
		print(traceback.format_exc())
		return {}

def calculate_drift(baseline, sample) -> dict:
	try:
		drift_detector = drift.DriftDetector(df_baseline=baseline, df_sample=sample)
		drift_result = drift_detector.calculate_drift(pre_defined_test='Kolmogorov-Smirnov')
		return {"data_drift": [ drift_result ]}
	except Exception as ex:
		print("Error occurred calculating drift")
		print(ex)
		print(traceback.format_exc())
		return {}


def calculate_stability(baseline, comparator, schema) -> dict:
	try:
		monitoring_parameters = infer.set_monitoring_parameters(schema, check_schema=True)
		stability_monitor = stability.StabilityMonitor(
			df_baseline=baseline,
			df_sample=comparator,
			predictors=monitoring_parameters["predictors"],
			feature_dataclass=monitoring_parameters["feature_dataclass"],
			special_values=monitoring_parameters["special_values"],
			score_column=monitoring_parameters["score_column"],
			label_column=monitoring_parameters["label_column"],
			weight_column=monitoring_parameters["weight_column"])
		# Set default n_groups for each predictor and score
		n_groups = {}
		for feature in monitoring_parameters["numerical_columns"] + [
			monitoring_parameters["score_column"]
		]:
			# If a feature has more than 1 unique value, set n_groups to 2; else set to 1
			feature_has_distinct_values = int((min(baseline[feature]) != max(baseline[feature])))
			n_groups[feature] = 1 + feature_has_distinct_values

		# Compute stability metrics
		stability_metrics = stability_monitor.compute_stability_indices(n_groups=n_groups, group_cuts={})
		return {"stability": [ stability_metrics ]}
	except Exception as ex:
		print("Error occurred calculating stability metrics")
		print(ex)
		print(traceback.format_exc())
		return {}


def run_great_expectations() -> dict:
	global GREAT_EXPECTATIONS_RESULT_FILE

	try:
		# run great expectations if present
		context = ge.get_context()
		checkpoints = context.list_checkpoints()
		GREAT_EXPECTATIONS_RESULT_FILE = None
		results = []
		for checkpoint in checkpoints:
			checkpoint_result = context.run_checkpoint(checkpoint_name=checkpoint)
			result_id = list(checkpoint_result.run_results.keys())[0]
			GREAT_EXPECTATIONS_RESULT_FILE = str(result_id).replace("ValidationResultIdentifier::", context.root_directory + "/uncommitted/validations/").replace(os.getcwd() + os.path.sep, './') + '.json'
			results.append(checkpoint_result.to_json_dict())
		return {"great_expectations": results}
	except ge.exceptions.exceptions.ConfigNotFoundError:
		print("No Great Expectations Tests Found")
		return {}


def run_data_quality(baseline, schema, additional_assets) -> dict:
	try:
		monitoring_parameters = infer.set_monitoring_parameters(schema, check_schema=True)
		protected_fields = monitoring_parameters.get('protected_classes', [])
		label = monitoring_parameters.get("label_column", None)
		if GREAT_EXPECTATIONS_RESULT_FILE is None:
			great_expectations_results = next(
				(asset for asset in additional_assets if asset.get('assetRole', {}) == 'GREAT_EXPECTATIONS_TEST'), {}).get(
				"filename", None)
		else:
			great_expectations_results = GREAT_EXPECTATIONS_RESULT_FILE
		# run the tests and outputs a summary of the quality tests
		try:
			# create a DataQuality object from the main class that holds all quality modules
			dq = DataQuality(df=baseline, results_json_path=great_expectations_results,
							 sensitive_features=protected_fields, label=label, plot=False)
			# Remove the data correlations generation as it does not scale
			if 'data-relations' in dq._engines_new:
				del dq._engines_new['data-relations']
			dq.evaluate()
		except Exception:
			print("Warning, failed to parse great expectations file in ydata-quality, skipping analysis")
			# If it throws, try again without the great expectations parsing as that seems a little buggy
			dq = DataQuality(df=baseline, sensitive_features=protected_fields, label=label, plot=False)
			if 'data-relations' in dq._engines_new:
				del dq._engines_new['data-relations']
			dq.evaluate()
		warnings = dq.get_warnings()

		dq_result = {"P1_Issues": sum(1 for value in warnings if (value.priority.value == 1)),
					 "P2_Issues": sum(1 for value in warnings if (value.priority.value == 2)),
					 "P3_Issues": sum(1 for value in warnings if (value.priority.value == 3)),
					 "warnings": []}
		for value in warnings:
			category_name = value.category.replace(' ', '_').replace('&', '_And_') + "_Issues"
			if category_name in dq_result:
				dq_result[category_name] += 1
			else:
				dq_result[category_name] = 1
			dq_result["warnings"].append(re.sub('\\s\\s+', ' ',
												value.priority.name + ': ' + value.category + ': ' + value.test + ': ' + value.description + ': ' + str(
													value.data)).replace("\n", " "))
		return dq_result
	except Exception as ex:
		print("Error occurred while examining data quality")
		print(ex)
		print(traceback.format_exc())
		return {}

def main():
	raw_json = Path('example_datasets/example_job.json').read_text()
	init_param = {'rawJson': raw_json}
	init(init_param)

	baseline_df = pd.read_csv('example_datasets/green_tripdata_2019-01-training.csv')
	comparator_df = pd.read_csv('example_datasets/green_tripdata_2019-01-sample.csv')

	result = metrics(baseline_df, comparator_df)
	print(json.dumps(next(result), indent=3, sort_keys=True))


if __name__ == '__main__':
	main()
