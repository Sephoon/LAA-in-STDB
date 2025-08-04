import csv
import os
import pickle
import time

from Database import Database

from attack.Attack_estimator import DensityEstimator, Attack_estimator
from utils import Range_statistic, Keywords_statistic, Overall_statistics

File_path = "data/"
results_path = "results/attack_estimator/"
num_run = 5

# Ensure the results directory exists
os.makedirs(results_path, exist_ok=True)
# Define the results filename
results_filename = os.path.join(results_path, "attack_estimator_summary_no_corr.csv")

# Datasets to be tested
Dataset = [
    [File_path + "enron_12x31_3w.pkl", "O2M", 12, 31],
    [File_path + "lucene_12x31_3w.pkl", "O2M", 12, 31],
    [File_path + "wiki_50x50_2w.pkl", "M2M", 50, 50],
    [File_path + "yfcc_50x50_4w.pkl", "O2M", 50, 50]
    # [File_path + "lucene_12x31_6w.pkl", "O2M", 12, 31]
]
# Known data percentages to be tested
Percentage = [0.01,0.03,0.05, 0.08 ,0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1]

# Create the CSV file and write the header before starting the experiments
header = [
    "Dataset",
    "Known_Ratio",
    "Query_Distribution",
    "Point_Recovery_Rate/Precision",
    "Document_Recovery_Rate/Precision",
    "Overall_Recovery_Rate/Accuracy",
    "Time_Elapsed(s)",
    "Run_Number"
]
Query_Distribution = ["uniform","gaussian","beta"]
with open(results_filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)

print(f"Results will be saved to: {results_filename}")

# --- Main Loop Starts ---
for dataset_info in Dataset:
    # Extract a clean name for the dataset from its file path
    dataset_name = os.path.basename(dataset_info[0]).replace('.pkl', '')
    print(f"\n{'=' * 25} STARTING NEW TEST {'=' * 25}")


    with open(dataset_info[0], "rb") as f:
        raw_records = pickle.load(f)

    DB = Database(raw_records, mode=dataset_info[1])
    N0, N1 = dataset_info[2], dataset_info[3]
    for distribution in Query_Distribution:
        RM,token_response_pairs = DB.simulate_client_queries(N0, N1,0.5,distribution)
        DE = DensityEstimator(token_response_pairs, "chao_lee")
        for percentage in Percentage:
            for run_idx in range(num_run):
                print(f"\n--- Testing Known Ratio: {percentage:.2%} in {distribution} distribution ---")
                print(f"Run: {run_idx + 1}/{num_run}, Dataset: {dataset_name}")


                idx, known_records = DB.get_known_part(percentage)
                M_, known_global_ids = DB.generate_doc_occ_matrix(known_records)
                known_volumes = [DB.records_volume[i - 1] for i in idx]

                # Start timer
                start_time = time.time()
                AT = Attack_estimator(known_records, DB.en_records, DE, 20, 0.25, N0, N1, 20,
                                      5, DB.mode, DB.matrix_M, M_, known_volumes, DB.records_volume,
                                      DB.ordered_enc_doc_ids, known_global_ids, 50, 30)
                points_map, doc_map = AT.attack()
                # Stop timer and calculate elapsed time
                end_time = time.time()
                # Call statistic functions and capture their results
                point_precision, point_recall = Range_statistic(points_map, DB.points_mapping[0], known_records)
                doc_precision, doc_recall = Keywords_statistic(doc_map, DB.documents_mapping[0], known_global_ids)
                overall_recovery, overall_accuracy = Overall_statistics(known_records, DB.points_mapping[0],
                                                                        DB.documents_mapping[0], points_map, doc_map)

                elapsed_time = end_time - start_time

                # Prepare the row of data to be written to the file
                result_row = [
                    dataset_name,
                    f"{percentage:.2%}",
                    distribution,
                    f"{point_recall:.4f}/{point_precision:.4f}",
                    f"{doc_recall:.4f}/{doc_precision:.4f}",
                    f"{overall_recovery:.4f}/{overall_accuracy:.4f}",
                    f"{elapsed_time:.2f}",
                    f"{run_idx + 1}"  # Just the run number is cleaner
                ]

                # Append the result row to the CSV file
                with open(results_filename, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(result_row)

                print(f"\n[SUCCESS] Results for this run have been saved. Time elapsed: {elapsed_time:.2f} seconds.")

print(f"\n{'=' * 25} ALL TESTS COMPLETED {'=' * 25}")
print(f"All results have been successfully saved to: {results_filename}")