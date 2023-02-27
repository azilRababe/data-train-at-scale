.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y taxifare || :
	@pip install -e .

##################### TESTS #####################
test_gcp_setup:
	@pytest \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_key_env \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_key_path \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_code_get_project \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_code_get_wagon_project

default:
	cat tests/train_at_scale/test_output.txt

test_kitt:
	@echo "🧪 computing and saving your progress at 'tests/train_at_scale/test_output.txt'..."
	@pytest tests/train_at_scale -c "./tests/pytest_kitt.ini" 2>&1 > tests/train_at_scale/test_output.txt
	@echo "📌 Please git add --all; git commit; ggpush to track progress on kitt"

test_preprocess_and_train:
	@pytest \
	tests/train_at_scale/test_clean.py \
	tests/train_at_scale/test_processor_pipeline.py \
	tests/train_at_scale/test_model.py \
	tests/train_at_scale/test_main_local.py::TestMainLocal::test_route_preprocess_and_train \
	tests/train_at_scale/test_main_local.py::TestMainLocal::test_route_pred

test_preprocess_by_chunk:
	@pytest \
	tests/train_at_scale/test_main_local.py::TestMainLocal::test_route_preprocess

test_train_by_chunk:
	@pytest \
	tests/train_at_scale/test_main_local.py::TestMainLocal::test_route_train

################### DATA SOURCES ACTIONS ################

# Data sources: targets for monthly data imports
ML_DIR=~/.lewagon/mlops
HTTPS_DIR=https://storage.googleapis.com/datascience-mlops/taxi-fare-ny/
GS_DIR=gs://datascience-mlops/taxi-fare-ny

show_sources_all:
	-ls -laR ~/.lewagon/mlops/data
	-bq ls ${BQ_DATASET}
	-bq show ${BQ_DATASET}.processed_1k
	-bq show ${BQ_DATASET}.processed_200k
	-bq show ${BQ_DATASET}.processed_all
	-gsutil ls gs://${BUCKET_NAME}

reset_local_files:
	rm -rf ${ML_DIR}
	mkdir -p ~/.lewagon/mlops/data/
	mkdir ~/.lewagon/mlops/data/raw
	mkdir ~/.lewagon/mlops/data/processed
	mkdir ~/.lewagon/mlops/training_outputs
	mkdir ~/.lewagon/mlops/training_outputs/metrics
	mkdir ~/.lewagon/mlops/training_outputs/models
	mkdir ~/.lewagon/mlops/training_outputs/params

reset_local_files_with_csv_solutions: reset_local_files
	-curl ${HTTPS_DIR}solutions/data_query_fixture_2009-01-01_2015-01-01_1k.csv > ${ML_DIR}/data/raw/query_2009-01-01_2015-01-01_1k.csv
	-curl ${HTTPS_DIR}solutions/data_query_fixture_2009-01-01_2015-01-01_200k.csv > ${ML_DIR}/data/raw/query_2009-01-01_2015-01-01_200k.csv
	-curl ${HTTPS_DIR}solutions/data_query_fixture_2009-01-01_2015-01-01_all.csv > ${ML_DIR}/data/raw/query_2009-01-01_2015-01-01_all.csv
	-curl ${HTTPS_DIR}solutions/data_processed_fixture_2009-01-01_2015-01-01_1k.csv > ${ML_DIR}/data/processed/processed_2009-01-01_2015-01-01_1k.csv
	-curl ${HTTPS_DIR}solutions/data_processed_fixture_2009-01-01_2015-01-01_200k.csv > ${ML_DIR}/data/processed/processed_2009-01-01_2015-01-01_200k.csv
	-curl ${HTTPS_DIR}solutions/data_processed_fixture_2009-01-01_2015-01-01_all.csv > ${ML_DIR}/data/processed/processed_2009-01-01_2015-01-01_all.csv
