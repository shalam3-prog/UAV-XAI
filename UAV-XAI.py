# --------------------------------------------------------------------------------------------------------------
#
# Filename: UAV-XAI.py
# Author: Shahid Alam (shalam3@gmail.com)
# Dated: August, 7, 2026
# Explain the results of UAV-XAI using different XAI techniques
#
# --------------------------------------------------------------------------------------------------------------

import os
import glob
import time
import shap
import pandas as pd
import numpy as np
import ipaddress
import matplotlib
matplotlib.use('Agg') # Reinforce headless backend locally
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

import sys
import pickle
import psutil
from sklearn.metrics import matthews_corrcoef
from scipy.stats import spearmanr
from lime import lime_tabular

import argparse

__RANDOM_STRATIFIED_SPLIT__ = True  # Set to True if you want to stratify the split by class labels
__TEMPORAL_SPLIT__ = False          # Set to True if you want to split based on temporal order (e.g., timestamps)
__ISOT_DRONE_2024__ = True          # Set to True if using the ISOT Drone Dataset 2024, False for UAVIDS-2025

parser = argparse.ArgumentParser()
parser.add_argument("-stratify", action="store_true")
parser.add_argument("-temporal", action="store_true")
parser.add_argument("-isot-2024", action="store_true")
args = parser.parse_args()
if args.stratify != None:
    __RANDOM_STRATIFIED_SPLIT__ = args.stratify
if args.temporal != None:
    __TEMPORAL_SPLIT__ = args.temporal
if args.isot_2024 != None:
    __ISOT_DRONE_2024__ = args.isot_2024

if __RANDOM_STRATIFIED_SPLIT__ == __TEMPORAL_SPLIT__:
    print("ERROR: Please select either Stratified Split or Temporal Split, not both.")
    sys.exit(1)

print(f"Stratified Split: {__RANDOM_STRATIFIED_SPLIT__}")
print(f"Temporal Split:   {__TEMPORAL_SPLIT__}")
print(f"ISOT Drone 2024:  {__ISOT_DRONE_2024__}")

# =====================================================================
# STEP 1: UNCAPPED CONSOLIDATION & MEMORY OPTIMIZATION
# =====================================================================
#dataset_root = "./ISOT-Drone-Dataset"  # Root directory containing subfolders for each drone network class
if __ISOT_DRONE_2024__ == True:
    master_output_file = "./isot_drone_master_dataset_small.csv"
else:
    master_output_file = "./UAVIDS-2025.csv"
if not os.path.exists(master_output_file):
    print(f"Master dataset file '{master_output_file}' not found. Please ensure the dataset is available.")
    sys.exit(1)
all_dataframes = []

print("="*60)
print("PHASE 1: CONSOLIDATING 100% OF UNLIMITED DATA (RAM OPTIMIZED)")
print("="*60)

def ip_to_int(ip_str: str) -> tuple[int, int]:
    """Converts any IPv4 or IPv6 string into an integer and its IP version.

    Returns (-1) if the IP is invalid.
    """
    try:
        # ip_address automatically detects if it is v4 or v6
        ip_obj = ipaddress.ip_address(str(ip_str).strip())
        return int(ip_obj)
    except (ValueError, TypeError):
        return -1  # Missing/Malformed data fallback

def protocol_to_int(protocol: str) -> int:
    """Converts a protocol name or string number into its standard IANA integer.

    Returns -1 if the protocol is invalid or unrecognized.
    """
    # 1. Standard IANA mapping for common protocols
    iana_map = {"ICMP": 1, "IGMP": 2, "TCP": 6, "UDP": 17, "IPV6": 41}

    # 2. Handle missing or empty inputs safely
    if protocol is None:
        return -1

    # 3. Clean up the string (remove spaces and make uppercase)
    cleaned = str(protocol).strip().upper()

    # 4. Check if it's a known textual protocol name
    if cleaned in iana_map:
        return iana_map[cleaned]

    # 5. Check if it's already a numeric string (e.g., "6" -> 6)
    if cleaned.isdigit():
        return int(cleaned)

    # 6. Return -1 for unknown protocols or text
    return -1

def mac_to_float(mac_address):
	try:
		if isinstance(mac_address, int):
			return 0
		# Convert to integer
		mac_int = int(mac_address.replace(":", ""), 16)
		# Normalize: A MAC address is 48-bit, so max value is 2^48 - 1
		# This keeps the value between 0 and 1
		return mac_int / (2**48 - 1)
	except ValueError:
		# Returns 0 or None if the mac_address is invalid
		return 0

master_df = pd.read_csv(master_output_file)
# For the dataset UAVIDS-2025
if __ISOT_DRONE_2024__ == False:
	master_df = master_df.rename(columns={"label": "Class"})
	if 'SrcAddr' in master_df.columns:
		master_df['SrcAddr'] = master_df['SrcAddr'].apply(ip_to_int)
	if 'DstAddr' in master_df.columns:
		master_df['DstAddr'] = master_df['DstAddr'].apply(ip_to_int)
	if 'Protocol' in master_df.columns:
		master_df['Protocol'] = master_df['Protocol'].apply(protocol_to_int)
#		master_df = master_df.drop(columns=["SrcAddr", "DstAddr", "FlowID", "SrcPort", "DstPort"], errors='ignore')

print(f"Working Dataset Shape: {master_df.shape}")

# =====================================================================
# STEP 2: CLASS DISTRIBUTION LOGGING & VISUALIZATION
# =====================================================================
print("\n" + "="*60)
print("PHASE 2: ANALYZING & GRAPHING FULL DATA DISTRIBUTION")
print("="*60)
class_counts = master_df['Class'].value_counts()
print("Full-Scale Sample Counts Per Class:")
for idx, (classname, count) in enumerate(class_counts.items(), 1):
	print(f" {idx:02d}. {classname:<20} : {count:,} samples")

plt.figure(figsize=(11, 6))
sns.set_theme(style="whitegrid")
ax = sns.barplot(x=class_counts.values, y=class_counts.index, hue=class_counts.index, palette="viridis", legend=False)

# Add numeric tags on the horizontal bars
for p in ax.patches:
	width = p.get_width()
	if width > 0:
		ax.text(width + (max(class_counts.values) * 0.01), p.get_y() + p.get_height() / 2, f'{int(width):,}', ha='left', va='center', fontsize=10, fontweight='semibold')

plt.title('Full Scale Sample Count Per Category', fontsize=13, fontweight='bold')
plt.xlabel('Total Log Records', fontsize=11, fontweight='bold')
plt.ylabel('Network Traffic Class', fontsize=11, fontweight='bold')
plt.tight_layout()
pdf_filename = f"dataset_distribution.pdf"
plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
plt.close()
print("  --> Distribution chart saved as 'dataset_distribution.pdf'")

# =====================================================================
# STEP 3: TRAIN-TEST SPLIT & STANDARDIZATION
# =====================================================================
from sklearn.model_selection import GroupShuffleSplit

X = master_df.drop(columns=['Class'])
y = master_df['Class']
if __RANDOM_STRATIFIED_SPLIT__ == True:
	# =====================================================================
	# METHOD 1: STRATIFIED-LEVEL SPLIT & STANDARDIZATION
	# =====================================================================
	print("\nPerforming stratified train-test split (80% train, 20% test)...")
	
	# 1. Perform the initial split to extract the correct stratified indices
	# Assumes X and y are already extracted from master_df
	X_train_temp, X_test_temp, _, _ = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y, shuffle=True)
	
	# Define the indices using the split results
	train_indices = X_train_temp.index
	test_indices = X_test_temp.index

	# Rebuild the final DataFrames from the accumulated indices
	train_df = master_df.loc[train_indices].copy()
	test_df = master_df.loc[test_indices].copy()

	# Extract inputs and labels
	X_train = train_df.drop(columns=['Class'])
	y_train = train_df['Class']
	X_test = test_df.drop(columns=['Class'])
	y_test = test_df['Class']

	# KEEP THE COLUMNS FOR XAI INTEGRITY BUT BLIND THE MODEL BY ZEROING VALUES
	if __ISOT_DRONE_2024__ == True:
		X_train['ts'] = 0.0; X_test['ts'] = 0.0
		print("Successfully blinded 'ts' feature matrix to prevent cheating.")

	# Standardize remaining features
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	print("\n=== SAFE CLASS-PRESERVING PORT SPLIT COMPLETE ===")
	print(f"Train samples: {X_train_scaled.shape} | Test samples: {X_test_scaled.shape}")
	print(f"\n✅ Train Class Breakdown (No vanishing classes):\n{y_train.value_counts()}")
	print(f"\n✅ Test Class Breakdown:\n{y_test.value_counts()}")
elif __TEMPORAL_SPLIT__ == True:
	# =====================================================================
	# METHOD 2: DATASET-AGNOSTIC ISOLATED CLASS SPLITTING
	# =====================================================================
	has_ts = 'ts' in master_df.columns

	# 1. Group consecutive records into blocks (by 'ts' if available, else by row index)
	if has_ts:
		WINDOW_SIZE = 10
		master_df['Time_Group'] = (master_df['ts'] / WINDOW_SIZE).astype(int)
	else:
		CHUNK_SIZE = 50  # Adjust based on dataset size (e.g., 50 or 100 consecutive flows)
		master_df['Time_Group'] = (master_df.index // CHUNK_SIZE).astype(int)

	# 2. Separate Benign rows from Attack rows completely
	# Supports both string labels ('Benign') and integer-encoded labels (0)
	if pd.api.types.is_numeric_dtype(master_df['Class']):
		benign_mask = master_df['Class'] == 0
	else:
		benign_mask = master_df['Class'].str.strip().str.lower() == 'benign'

	df_benign = master_df[benign_mask].copy()
	df_attacks = master_df[~benign_mask].copy()

	# 3. Perform a Pure Group Shuffle Split on the BENIGN data
	gss_b = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
	b_train_idx, b_test_idx = next(gss_b.split(df_benign, groups=df_benign['Time_Group']))
	X_b_train, X_b_test = df_benign.iloc[b_train_idx], df_benign.iloc[b_test_idx]

	# 4. Perform a Pure Group Shuffle Split on the ATTACK data
	gss_a = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
	a_train_idx, a_test_idx = next(gss_a.split(df_attacks, groups=df_attacks['Time_Group']))
	X_a_train, X_a_test = df_attacks.iloc[a_train_idx], df_attacks.iloc[a_test_idx]

	# 5. Concatenate them back into master Train and Test pools
	train_final = pd.concat([X_b_train, X_a_train]).sample(frac=1, random_state=42).reset_index(drop=True)
	test_final = pd.concat([X_b_test, X_a_test]).sample(frac=1, random_state=42).reset_index(drop=True)

	# 6. Extract raw inputs and labels
	X_train = train_final.drop(columns=['Class', 'Time_Group'])
	y_train = train_final['Class']

	X_test = test_final.drop(columns=['Class', 'Time_Group'])
	y_test = test_final['Class']

	# 7. Blind 'ts' ONLY if the dataset actually contains it
	if has_ts:
		X_train['ts'] = 0.0
		X_test['ts'] = 0.0
		print("Successfully blinded 'ts' columns by zeroing values out. Array shape preserved for XAI.")
	else:
		print("No 'ts' column found; used row-index chunking for temporal grouping.")

	# 8. Standardize remaining features
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	print("\n=== TIME-ISOLATED CLASS SEPARATION SPLIT COMPLETE ===")
	print(f"Train samples: {X_train_scaled.shape[0]} | Test samples: {X_test_scaled.shape[0]}")
	print(f"✅ Train Class Breakdown:\n{y_train.value_counts()}")
	print(f"✅ Test Class Breakdown (Benign is now preserved):\n{y_test.value_counts()}")
else:
    print("No valid split method selected. Please set either __RANDOM_STRATIFIED_SPLIT__ or __TEMPORAL_SPLIT__ to True.")
    sys.exit(1)

# =====================================================================
# STEP 4: RANDOM FOREST TRAINING & BENCHMARKING
# =====================================================================
print("\n" + "="*60)
print("PHASE 3: MODEL TRAINING AND METRICS EVALUATION")
print("="*60)

log_output = []
def log_and_print(text):
	print(text)
	log_output.append(text)

log_and_print("Training Full-Scale Random Forest Classifier...")
# n_jobs=-1 utilizes all available CPU cores for maximum speed
rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)

start = time.time()
rf_model.fit(X_train_scaled, y_train)
rf_train_time = time.time() - start

start = time.time()
rf_preds = rf_model.predict(X_test_scaled)
rf_infer_time = time.time() - start

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

print("\n" + "="*60)
print("PHASE 3: TRAINING ONE-VS-REST RANDOM FOREST FOR PER-CLASS XAI")
print("="*60)

# 1. Initialize the Base Random Forest
base_rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)

# 2. Wrap it inside One-Vs-Rest to break the multi-class problem into 10 distinct models
ovr_model = OneVsRestClassifier(base_rf)

start_time = time.time()
ovr_model.fit(X_train_scaled, y_train)
ovr_train_time = time.time() - start_time
print(f"One-vs-Rest Random Forest trained in: {ovr_train_time:.2f} seconds")

# Predict and get class names array
rf_preds = ovr_model.predict(X_test_scaled)
class_names = sorted(list(set(y_test)))

# =====================================================================
# STEP 5: COMPREHENSIVE PERFORMANCE EXPORT (WITH MCC, SIZE & MEMORY)
# =====================================================================
class_names = sorted(list(set(y_test)))

log_and_print(f"\n=== METRICS SUMMARY: RANDOM FOREST ===")
log_and_print(f"Training Time: {rf_train_time:.4f} seconds")
log_and_print(f"On-Board Inference Time: {rf_infer_time:.4f} seconds")

# 1. Compute Model Disk Size Footprint via Pickle Serializer Byte Counting
model_bytes = len(pickle.dumps(rf_model))
model_size_mb = model_bytes / (1024 * 1024)
log_and_print(f"Model Disk Footprint  : {model_size_mb:.2f} MB")

# 2. Capture Active Virtual Memory Consumption of the Python Process
process = psutil.Process()
current_ram_mb = process.memory_info().rss / (1024 * 1024)
log_and_print(f"Active RAM Consumption : {current_ram_mb:.2f} MB")

# 3. 10x10 Confusion Matrix Text DataFrame
cm = confusion_matrix(y_test, rf_preds, labels=class_names)
cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in class_names], columns=[f"Pred_{c}" for c in class_names])
log_and_print("\n10x10 Confusion Matrix:\n" + cm_df.to_string())

# 4. Extract Per-Class Performance Indicators (Including Per-Class Binary MCC)
precision, recall, f1, support = precision_recall_fscore_support(y_test, rf_preds, labels=class_names, zero_division=0)
per_class_accuracy = [cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0 for i in range(len(class_names))]

# Dynamically calculate the binary Matthews Correlation Coefficient for each of the 10 classes
per_class_mcc = []
for c in class_names:
    y_test_binary = (y_test == c).astype(int)
    rf_preds_binary = (pd.Series(rf_preds) == c).astype(int)
    class_mcc = matthews_corrcoef(y_test_binary, rf_preds_binary)
    per_class_mcc.append(class_mcc)

metrics_df = pd.DataFrame({
    'Class Category': class_names, 
    'Accuracy': [f"{a*100:.2f}%" for a in per_class_accuracy],
    'Precision': [f"{p*100:.2f}%" for p in precision], 
    'Recall': [f"{r*100:.2f}%" for r in recall],
    'F1-Score': [f"{f*100:.2f}%" for f in f1], 
    'MCC Score': [f"{m:.4f}" for m in per_class_mcc], # High-precision 4-decimal tracking for academic review
    'Support': support
})
log_and_print("\nPer-Class Performance Matrix:\n" + metrics_df.to_string(index=False))

# 5. Macro, System-Wide Averages, and Multi-Class Global MCC
global_multiclass_mcc = matthews_corrcoef(y_test, rf_preds)

log_and_print(f"\nGlobal Macro Performance Indicators:")
log_and_print(f"  -> System Accuracy : {accuracy_score(y_test, rf_preds)*100:.2f}%")
log_and_print(f"  -> Average Precision: {np.mean(precision)*100:.2f}%")
log_and_print(f"  -> Average Recall   : {np.mean(recall)*100:.2f}%")
log_and_print(f"  -> Average F1-Score : {np.mean(f1)*100:.2f}%")
log_and_print(f"  -> Global Multi-Class MCC: {global_multiclass_mcc:.4f}")

# 6. Save High-Resolution Visual Heatmap
print("\nGenerating visual confusion matrix chart...")
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_normalized = np.nan_to_num(cm_normalized)

plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, square=True)
plt.title('Normalized Confusion Matrix - Random Forest', fontsize=13, fontweight='bold', pad=15)
plt.ylabel('True Attack Category Label', fontsize=11, fontweight='bold')
plt.xlabel('Predicted Network Classification', fontsize=11, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("random_forest_confusion_matrix.pdf", format='pdf', bbox_inches='tight')
plt.close()
print("  --> Heatmap graph saved as 'random_forest_confusion_matrix.pdf'")

# =====================================================================
# PHASE 4: EXTRACTING PER-CLASS FEATURE IMPORTANCE (XAI)
# =====================================================================
print("\n" + "="*60)
print("PHASE 4: EXPLAINABLE AI (XAI) PER-CLASS SUB-ANALYSIS")
print("="*60)

per_class_xai_log = []

# Iterate through each of the 10 binary sub-estimators
for idx, class_label in enumerate(class_names):
    # Extract the specific trained Random Forest for this individual class
    sub_estimator = ovr_model.estimators_[idx]
    sub_importances = sub_estimator.feature_importances_
    
    # SAFE CHECK: Attempt to get feature names the model was trained on,
    # otherwise fallback to a generic numerical index list matching the length.
    if hasattr(sub_estimator, 'feature_names_in_'):
        current_features = sub_estimator.feature_names_in_
    elif len(X.columns) == len(sub_importances):
        current_features = X.columns
    else:
        current_features = [f"Feature_{i}" for i in range(len(sub_importances))]
    
    # Structure into a temporary dataframe
    class_df = pd.DataFrame({
        'Feature': current_features,
        'Importance': sub_importances
    }).sort_values(by='Importance', ascending=False).head(5) # Get top 5 features per class
    
    # Format and store the output text
    header = f"\n--- Top 5 Features Identifying Class: [{class_label}] ---"
    print(header)
    print(class_df.to_string(index=False))
    
    per_class_xai_log.append(header)
    per_class_xai_log.append(class_df.to_string(index=False))

# Append this per-class data to your main 'drone_security_results.txt' output logic
log_and_print("\nXAI Per Class Analysis:\n" + "\n".join(per_class_xai_log))

print("\n" + "="*60)
print("PHASE 4: LIGHTWEIGHT EXPLAINABLE AI (XAI) EXTRACTION")
print("="*60)

importances = rf_model.feature_importances_

# Extract names used during model training, or fallback safely to prevent a crash
if hasattr(rf_model, 'feature_names_in_'):
    trained_features = rf_model.feature_names_in_
elif len(X.columns) == len(importances):
    trained_features = X.columns
else:
    trained_features = [f"Feature_{i}" for i in range(len(importances))]

# Safely build the DataFrame using matched arrays
xai_df = pd.DataFrame({'Network Feature': trained_features, 'Gini Score Importance': importances})
xai_df = xai_df.sort_values(by='Gini Score Importance', ascending=False).head(10)

log_and_print("\nTop 10 Most Influential Network Features Decoded by XAI:\n" + xai_df.to_string(index=False))

# Write textual data arrays down to the shared logs file
with open("drone_security_results.txt", "w", encoding="utf-8") as f:
	f.write("\n".join(log_output))
print("\n[Complete] Data processing, training, visual plots, and text metrics logs saved successfully.")

def export_per_class_xai_chart(ovr_model, feature_names, class_names, filename="per_class_xai_comparison.pdf"):
	print("\nProcessing multi-class XAI matrix chart...")
	
	# 1. Gather all feature importances across all sub-estimators
	importance_data = {}
	for idx, class_label in enumerate(class_names):
		sub_estimator = ovr_model.estimators_[idx]
		importance_data[class_label] = sub_estimator.feature_importances_
		
	# --- SAFE CORRECTION FOR FEATURE NAMES LENGTH ---
	# Get the expected length from the actual trained importances array
	expected_len = len(next(iter(importance_data.values()))) if importance_data else 0
	
	# If the passed feature_names array size mismatches (e.g., 22 vs 24), recover from model memory
	if len(feature_names) != expected_len:
		first_estimator = ovr_model.estimators_[0]
		if hasattr(first_estimator, 'feature_names_in_'):
			feature_names = first_estimator.feature_names_in_
		else:
			# Absolute baseline fallback to prevent crash if names aren't in metadata
			feature_names = [f"Feature_{i}" for i in range(expected_len)]

	# Create master matrix dataframe safely now that lengths match perfectly
	matrix_df = pd.DataFrame(importance_data, index=feature_names)
	
	# 2. Identify the top 5 globally important features to keep the plot highly scannable
	# We rank features by their maximum impact score in any single class
	matrix_df['Max_Impact'] = matrix_df.max(axis=1)
	top_features = matrix_df.sort_values(by='Max_Impact', ascending=False).head(5).index.tolist()
	
	# Filter matrix to only showcase the top 5 features across all classes
	plot_df = matrix_df.loc[top_features].drop(columns=['Max_Impact'])
	
	# Transform data from Wide to Long format for Seaborn compatibility
	plot_df = plot_df.reset_index().melt(id_vars='index', var_name='Drone Traffic Class', value_name='Gini Importance Score')
	plot_df.rename(columns={'index': 'Network Feature'}, inplace=True)
	
	# 3. Initialize matplotlib canvas
	plt.figure(figsize=(14, 7))
	sns.set_theme(style="whitegrid")
	
	# 4. Generate the Grouped Multi-Class Bar Chart
	# Features go on X-axis, Importance on Y-axis, grouped by the 10 Class colors
	ax = sns.barplot(
		data=plot_df,
		x='Network Feature',
		y='Gini Importance Score',
		hue='Drone Traffic Class',
		palette='tab10' # Distinct colors for the 10 classes
	)
	
	# 5. Format layout styling
	plt.title('Explainable AI (XAI): Top Network Features Breakdown Across All Classes', fontsize=14, fontweight='bold', pad=20)
	plt.xlabel('Extracted Network Traffic Feature Name', fontsize=12, fontweight='bold', labelpad=12)
	plt.ylabel('Feature Contribution (Gini Importance Score)', fontsize=12, fontweight='bold')
	
	# Position the legend so it doesn't block the actual bars
	plt.legend(title='Traffic Profile Category', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=10)
	
	plt.xticks(rotation=15, ha='right', fontsize=10)
	plt.tight_layout()
	
	# 6. Save crisp image at 300 DPI
	plt.savefig(filename, format='pdf', bbox_inches='tight')
	plt.close()
	print(f"  --> Multi-class XAI chart saved as '{filename}'")

# Execute visualization right after Phase 4 completing your script
export_per_class_xai_chart(ovr_model, X.columns, class_names)

def compare_true_per_class_shap_vs_gini(ovr_model, X_train_scaled, y_train, feature_names, class_names):
    print("\n" + "="*60)
    print("PHASE 5: RIGOROUS APPLES-TO-APPLES PER-CLASS XAI COMPARISON")
    print("="*60)
    print("Extracting per-class metrics directly from identical binary sub-estimators...")
    
    # --- SELF-HEALING FEATURE NAMES GUARD ---
    # Detect the true number of features inside the scaled matrix (e.g., 24)
    actual_num_features = X_train_scaled.shape[1]
    
    # If the passed names list (22) doesn't match the data array (24), pull from model memory
    if len(feature_names) != actual_num_features:
        first_estimator = ovr_model.estimators_[0]
        if hasattr(first_estimator, 'feature_names_in_'):
            feature_names = first_estimator.feature_names_in_
        else:
            # Fallback names if model metadata is missing
            feature_names = [f"Feature_{i}" for i in range(actual_num_features)]

    # 1. Establish a memory-safe, balanced dataset slice for the SHAP Explainer
    temp_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    temp_df['Class_Label'] = y_train.values
    
    sample_size_per_class = 200
    shap_sample = temp_df.groupby('Class_Label', group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_size_per_class), random_state=42),
        include_groups=False
    )
    X_shap_subset = shap_sample.values
    
    comparison_logs = []
    
    # 2. Iterate sequentially through each of the 10 separate binary classifiers
    for idx, class_label in enumerate(class_names):
        print(f"\nProcessing Sub-Model [{idx+1}/10] for Class: {class_label}...")
        
        # Extract the specific trained Random Forest model for THIS class alone
        binary_rf_sub_model = ovr_model.estimators_[idx]
        
        # Extract Gini values from the binary sub-model
        binary_gini_importances = binary_rf_sub_model.feature_importances_
        
        # Build the TreeSHAP Explainer on top of the exact same binary sub-model
        explainer = shap.TreeExplainer(binary_rf_sub_model)
        binary_shap_values = explainer.shap_values(X_shap_subset)
        
        if isinstance(binary_shap_values, list):
            final_shap_matrix = binary_shap_values[1] if len(binary_shap_values) > 1 else binary_shap_values[0]
        elif len(binary_shap_values.shape) == 3:
            final_shap_matrix = binary_shap_values[:, :, 1]
        else:
            final_shap_matrix = binary_shap_values

        # Calculate the localized magnitude impact
        mean_abs_shap = np.mean(np.abs(final_shap_matrix), axis=0)
        
        # Force flatten arrays into true 1D structures
        mean_abs_shap = np.squeeze(np.asarray(mean_abs_shap))
        binary_gini_importances = np.squeeze(np.asarray(binary_gini_importances))
        
        # 3. Structure the unified, apples-to-apples comparison matrix
        comp_df = pd.DataFrame({
            'Network Feature': feature_names,
            'Binary Mean Abs SHAP': mean_abs_shap,
            'Binary Gini Importance': binary_gini_importances
        })
        
        comp_df = comp_df.sort_values(by='Binary Mean Abs SHAP', ascending=False).head(5)
        
        header = f"\n[Class Profile: {class_label}] - Pure Per-Class Comparison (SHAP vs Gini from Same Sub-Model):"
        print(header)
        print(comp_df.to_string(index=False))
        
        comparison_logs.append(header)
        comparison_logs.append(comp_df.to_string(index=False))
        
        # 4. Export side-by-side bar charts safely using headless backend
        plt.figure(figsize=(10, 4))
        melted_df = comp_df.melt(
            id_vars='Network Feature', 
            value_vars=['Binary Mean Abs SHAP', 'Binary Gini Importance'], 
            var_name='XAI Metric Type', 
            value_name='Metric Value'
        )
        
        sns.barplot(data=melted_df, x='Metric Value', y='Network Feature', hue='XAI Metric Type', palette='muted')
        plt.title(f'True Per-Class XAI Metrics Alignment: {class_label}', fontsize=12, fontweight='bold')
        plt.xlabel('Normalized Importance Weight')
        plt.ylabel('Network Feature Field')
        plt.tight_layout()
        
        # Save as a clean PDF figure
        plt.savefig(f"true_xai_comparison_{class_label.lower().replace(' ', '_')}.pdf", format='pdf', bbox_inches='tight')
        
        # THREAD SAFETY: Clear and close the active plot completely to free memory and GUI links
        plt.clf()
        plt.close('all')

    # 5. Append clean output text to your central log file
    with open("drone_security_results.txt", "a", encoding="utf-8") as f:
        f.write("\n\n" + "\n".join(comparison_logs))
    print(f"\n[Success] All side-by-side per-class verification charts saved safely.")

# Execute Phase 5 safely using the patched function
compare_true_per_class_shap_vs_gini(ovr_model, X_train_scaled, y_train, X.columns, class_names)

def evaluate_quantitative_xai_convergence(ovr_model, X_train_scaled, X_test_scaled, y_train, y_test, feature_names, class_names, K=5):
    print("\n" + "="*60)
    print("PHASE 5: QUANTITATIVE XAI CONVERGENCE ENGINE")
    print("="*60)
    print(f"Evaluating Spearman Rank Correlation and Top-{K} Set Overlap...")
    
    # 1. Establish memory-safe background slice for TreeSHAP
	# --- SELF-HEALING FEATURE NAMES GUARD ---
	# Detect the true number of features inside the scaled matrix (24)
    actual_num_features = X_train_scaled.shape[1]
	
    # If the passed names list (22) doesn't match the matrix (24), pull from model memory
    if len(feature_names) != actual_num_features:
        # Access the underlying estimator's feature names or fallback to a first sub-model
        first_estimator = ovr_model.estimators_[0] if hasattr(ovr_model, 'estimators_') else ovr_model
        if hasattr(first_estimator, 'feature_names_in_'):
            feature_names = first_estimator.feature_names_in_
        else:
            # Safety fallback names if model metadata is missing
            feature_names = [f"Feature_{i}" for i in range(actual_num_features)]

    # Line 721 (Now runs perfectly because shapes match exactly):
    temp_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    temp_df['Class_Label'] = y_train.values

    sample_size_per_class = 200
    shap_sample = temp_df.groupby('Class_Label', group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_size_per_class), random_state=42),
        include_groups=False
    )
    X_shap_subset = shap_sample.values
    
    # 2. Initialize a local LIME explainer aligned with the training space
    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    
    xai_metrics_accumulator = []
    text_summary_logs = []
    
    # 3. Process each class sequentially
    for idx, class_label in enumerate(class_names):
        print(f"Analyzing metric convergence for class [{idx+1}/10]: {class_label}...")
        
        # Extract identical binary sub-estimator
        binary_rf_sub_model = ovr_model.estimators_[idx]
        binary_gini_importances = binary_rf_sub_model.feature_importances_
        
        # Extract matching binary TreeSHAP values
        explainer = shap.TreeExplainer(binary_rf_sub_model)
        binary_shap_values = explainer.shap_values(X_shap_subset)
        
        if isinstance(binary_shap_values, list):
            final_shap_matrix = binary_shap_values if len(binary_shap_values) > 1 else binary_shap_values
        elif len(binary_shap_values.shape) == 3:
            final_shap_matrix = binary_shap_values[:, :, 1]
        else:
            final_shap_matrix = binary_shap_values
            
        mean_abs_shap = np.mean(np.abs(final_shap_matrix), axis=0)
        
        # Squeeze arrays to absolute 1D flat structures
        mean_abs_shap = np.squeeze(np.asarray(mean_abs_shap))
        binary_gini_importances = np.squeeze(np.asarray(binary_gini_importances))
        
        # 4. Extract aligned localized LIME explanations for this specific class
        # Find first instance in test partition belonging to this class
        y_test_reset = y_test.reset_index(drop=True)
        matching_test_indices = y_test_reset[y_test_reset == class_label].index.tolist()
        
        # If an explicit packet instance exists, extract its local LIME array
        if matching_test_indices:
            target_packet_idx = matching_test_indices[0]
            specific_packet = X_test_scaled[target_packet_idx]
            
            # Predict using the identical binary model array to ensure evaluation symmetry
            lime_exp = lime_explainer.explain_instance(
                data_row=specific_packet,
                predict_fn=binary_rf_sub_model.predict_proba, 
                num_features=len(feature_names) # Map entire feature space vector
            )
            
            # Reconstruct full-length flat LIME weight array matching feature names ordering
            lime_map = {feature_names[f_idx]: weight for f_idx, weight in lime_exp.local_exp[1]}
            lime_weights_flat = np.array([abs(lime_map.get(col, 0.0)) for col in feature_names])
        else:
            lime_weights_flat = np.zeros(len(feature_names))
            
        # 5. Core Metric Calculations
        # A. Spearman Rank Correlation (Gini vs SHAP)
        rho_gini_shap, _ = spearmanr(binary_gini_importances, mean_abs_shap)
        if np.isnan(rho_gini_shap): rho_gini_shap = 0.0
        
        # B. Set Overlap Calculations (Top-K)
        top_k_gini = set(pd.Series(binary_gini_importances, index=feature_names).sort_values(ascending=False).head(K).index)
        top_k_shap = set(pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False).head(K).index)
        top_k_lime = set(pd.Series(lime_weights_flat, index=feature_names).sort_values(ascending=False).head(K).index)
        
        overlap_gini_shap = len(top_k_gini.intersection(top_k_shap)) / K
        overlap_shap_lime = len(top_k_shap.intersection(top_k_lime)) / K
        
        # Accumulate scores for global multi-class averaging
        xai_metrics_accumulator.append({
            'Class': class_label,
            'Spearman_Gini_SHAP': rho_gini_shap,
            'Overlap_Gini_SHAP': overlap_gini_shap,
            'Overlap_SHAP_LIME': overlap_shap_lime
        })
        
    # 6. Generate Performance DataFrames and Academic Output Strings
    metrics_df = pd.DataFrame(xai_metrics_accumulator)
    
    avg_rho = metrics_df['Spearman_Gini_SHAP'].mean()
    avg_overlap_gs = metrics_df['Overlap_Gini_SHAP'].mean() * 100
    avg_overlap_sl = metrics_df['Overlap_SHAP_LIME'].mean() * 100
    
    header = f"\n=======================================================\nQUANTITATIVE XAI CONVERGENCE RESULTS (Top-{K})\n======================================================="
    text_summary_logs.append(header)
    text_summary_logs.append(metrics_df.to_string(index=False))
    
    # Print straight to terminal execution screen
    print(header)
    print(metrics_df.to_string(index=False))
    
    # 7. Permanently save these statistical findings down to central results log
    with open("drone_security_results.txt", "a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(text_summary_logs))
        
    print("\n[Success] Quantitative XAI matrices and abstract narrative logged successfully.")

# Create the missing variable using the model's true training memory layout
if hasattr(ovr_model.estimators_[0], 'feature_names_in_'):
    feature_names = ovr_model.estimators_[0].feature_names_in_
else:
    feature_names = X.columns  # Fallback to current columns if metadata is missing

# Now call the function securely using the defined variable
evaluate_quantitative_xai_convergence(ovr_model, X_train_scaled, X_test_scaled, y_train, y_test, feature_names, class_names, K=5)

def batch_export_stratified_lime_pdfs(rf_model, X_train_scaled, X_test_scaled, y_test, feature_names, class_names, samples_per_status=2, seed=42):
    print("\n" + "="*60)
    print("PHASE 6: STRATIFIED RANDOM BATCH LIME EXPLANATION EXPORTER")
    print("="*60)
    print(f"Sampling {samples_per_status} Correct and {samples_per_status} Misclassified packets per class (Seed: {seed})...")
    
    # --- SELF-HEALING FEATURE NAMES GUARD ---
    # Detect the true number of features inside the scaled matrix (e.g., 24)
    actual_num_features = X_train_scaled.shape[1]
    
    # If the passed names list (22) doesn't match the data array (24), pull from model memory
    if len(feature_names) != actual_num_features:
        if hasattr(rf_model, 'feature_names_in_'):
            feature_names = rf_model.feature_names_in_
        else:
            # Fallback names if model metadata is missing
            feature_names = [f"Feature_{i}" for i in range(actual_num_features)]

    # 1. Initialize tabular explainer linked tightly to training baseline data space
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    
    # 2. Re-map test targets and generate clean global predictions array
    y_test_reset = y_test.reset_index(drop=True)
    all_probs = rf_model.predict_proba(X_test_scaled)
    all_preds = np.array([class_names[np.argmax(p)] for p in all_probs])
    all_confidences = np.max(all_probs, axis=1)
    
    np.random.seed(seed)  # Fix structural random generator state for academic reproducibility
    
    # 3. Iterate sequentially through each distinct network profile
    for target_class in class_names:
        print(f"\nProcessing Stratified LIME Extractions for Profile Type: [{target_class}]")
        
        # Sift structural true identities matching this category
        class_mask = (y_test_reset == target_class)
        
        # Partition 1: Isolate Correctly Classified instances (True Positives)
        correct_indices = y_test_reset[class_mask & (all_preds == target_class)].index.tolist()
        # Partition 2: Isolate Misclassified instances (False Negatives or False Positives matching boundary error)
        incorrect_indices = y_test_reset[class_mask & (all_preds != target_class)].index.tolist()
        
        # Randomly sample fixed targets per partition state using reproducibility constraints
        selected_correct = np.random.choice(correct_indices, min(len(correct_indices), samples_per_status), replace=False) if correct_indices else []
        selected_incorrect = np.random.choice(incorrect_indices, min(len(incorrect_indices), samples_per_status), replace=False) if incorrect_indices else []
        
        # Define evaluation pipeline array pairs: (test_index, evaluation_status_string)
        evaluation_queue = (
            [(idx, "Correct_Classification") for idx in selected_correct] + 
            [(idx, "Misclassification") for idx in selected_incorrect]
        )
        
        # 4. Generate local explanations and save standalone vector figures
        for test_row_index, status in evaluation_queue:
            specific_packet = X_test_scaled[test_row_index]
            true_class_name = str(y_test_reset.iloc[test_row_index])
            predicted_class_name = str(all_preds[test_row_index])
            confidence = all_confidences[test_row_index]
            
            print(f"  -> Exporting PDF for Instance ID {test_row_index} | Status: {status}...")
            
            # Map local boundary conditions weights matching feature attributes length
            exp = explainer.explain_instance(
                data_row=specific_packet,
                predict_fn=rf_model.predict_proba,
                num_features=5
            )
            
            # Extract internal structural matplotlib canvas reference object
            fig = exp.as_pyplot_figure()
            
            # Force formal, structured academic titles outlining exact forensic states
            plt.title(
                f"LIME Local Explanation Vector Profile ({status.replace('_', ' ')})\n"
                f"Instance ID: #{test_row_index} | True Profile: [{true_class_name}] | Predicted: [{predicted_class_name}] ({confidence*100:.1f}%)",
                fontsize=11, fontweight='bold', pad=15
            )
            plt.xlabel('Local Feature Contribution Weight Flag', fontsize=10, fontweight='bold')
            plt.ylabel('Evaluated Boundary Threshold Criteria', fontsize=10, fontweight='bold')
            
            # Structure filename cleanly containing all identifying parameters for LaTeX import
            clean_class_str = str(target_class).lower().replace(" ", "_")
            clean_status_str = status.lower()
            pdf_filename = f"lime_case_study_{clean_class_str}_{clean_status_str}_id{test_row_index}.pdf"
            
            plt.tight_layout()
            plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
            plt.clf()
            plt.close('all')
            
    print("\n[Success] Stratified multi-instance case study PDF charts exported to local directory.")

# Add this invocation block directly to the absolute bottom of your main script pipeline:
batch_export_stratified_lime_pdfs(rf_model, X_train_scaled, X_test_scaled, y_test, X.columns, class_names, samples_per_status=2, seed=42)

import time
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

def run_empirical_feature_ablation_study(rf_model, X_train_scaled, X_test_scaled, y_train, y_test, feature_names):
    print("\n" + "="*60)
    print("PHASE 7: EMPIRICAL FEATURE ABLATION SUITE")
    print("="*60)
    print("Evaluating model degradation across XAI-guided feature subsets...")
    
    # 1. Identify and rank the globally dominant features based on the base Random Forest model
    global_importances = rf_model.feature_importances_
    ranked_features_indices = np.argsort(global_importances)[::-1] # Sorted high-to-low index positions
    
    top_5_indices = ranked_features_indices[:5]
    top_10_indices = ranked_features_indices[:10]
    without_top_10_indices = ranked_features_indices[10:]
    
    # Map back to readable string names for logging
    top_5_names = [feature_names[i] for i in top_5_indices]
    top_10_names = [feature_names[i] for i in top_10_indices]
    
    print(f"  -> XAI Identified Top-5 Features  : {top_5_names}")
    print(f"  -> XAI Identified Top-10 Features : {top_10_names}")
    
    # 2. Define the structural configurations for the ablation experiment
    ablation_scenarios = {
        "Full Feature Set (Baseline)": (slice(None), X_train_scaled.shape[1]),
        "Top-5 Dominant Features Only": (top_5_indices, 5),
        "Top-10 Dominant Features Only": (top_10_indices, 10),
        "Ablated (Without Top-10 Features)": (without_top_10_indices, X_train_scaled.shape[1] - 10)
    }
    
    ablation_results_accumulator = []
    
    # 3. Sequentially train and evaluate cloned models on sliced data matrices
    for scenario_name, (feature_subset_indices, count) in ablation_scenarios.items():
        print(f"Processing Scenario: [{scenario_name}] evaluating {count} feature columns...")
        
        # Slice columns from pre-scaled train and test sets
        if isinstance(feature_subset_indices, slice):
            X_train_sliced = X_train_scaled
            X_test_sliced = X_test_scaled
        else:
            X_train_sliced = X_train_scaled[:, feature_subset_indices]
            X_test_sliced = X_test_scaled[:, feature_subset_indices]
            
        # Clone structural hyperparameters of your base Random Forest model to ensure mathematical isolation
        ablation_clf = clone(rf_model)
        
        # Benchmark training latency
        start_time = time.time()
        ablation_clf.fit(X_train_sliced, y_train)
        elapsed_train_time = time.time() - start_time
        
        # Predict across testing partition
        preds = ablation_clf.predict(X_test_sliced)
        
        # Quantify metrics
        acc = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average='macro', zero_division=0)
        mcc_global = matthews_corrcoef(y_test, preds)
        
        ablation_results_accumulator.append({
            "Feature Configuration Scenario": scenario_name,
            "Feature Count": count,
            "Accuracy (%)": f"{acc*100:.2f}%",
            "F1-Macro (%)": f"{f1_macro*100:.2f}%",
            "Global MCC": f"{mcc_global:.4f}",
            "Training Overhead (s)": f"{elapsed_train_time:.3f}s"
        })
        
    # 4. Format outputs into a highly readable summary table
    summary_df = pd.DataFrame(ablation_results_accumulator)
    
    header = f"\n====================================================================\nEXPERIMENTAL FEATURE ABLATION PERFORMANCE MATRIX\n===================================================================="
    print(header)
    print(summary_df.to_string(index=False))
    
    # 6. Append metrics directly to your paper's results log file
    with open("drone_security_results.txt", "a", encoding="utf-8") as f:
        f.write("\n\n" + header + "\n" + summary_df.to_string(index=False))
        
    print("\n[Complete] Feature ablation matrix successfully executed and logged.")

run_empirical_feature_ablation_study(rf_model, X_train_scaled, X_test_scaled, y_train, y_test, X.columns)

sys.exit(0)  # Graceful termination of the script after all phases complete
