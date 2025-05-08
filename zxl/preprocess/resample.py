import SimpleITK as sitk
import json
import os
import glob

def resample_image_sitk(image_sitk, target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear, original_spacing=None, is_label=False):
    if original_spacing is None:
        original_spacing = image_sitk.GetSpacing()
    original_size = image_sitk.GetSize()

    # Calculate new size based on target spacing
    # new_size = [old_size * old_spacing / new_spacing]
    new_size = [
        int(round(orig_size * orig_spacing_dim / target_spacing_dim))
        for orig_size, orig_spacing_dim, target_spacing_dim in zip(original_size, original_spacing, target_spacing)
    ]
    # Ensure size is at least 1 in each dimension
    new_size = [max(0.1, s) for s in new_size]

    #设置resample
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(list(target_spacing)) # SimpleITK expects a list or tuple
    resample.SetSize(new_size)  # 目标图像大小 [x,y,z]
    resample.SetOutputDirection(image_sitk.GetDirection())
    resample.SetOutputOrigin(image_sitk.GetOrigin())
    resample.SetTransform(sitk.Transform()) # Identity transform，可设置旋转，缩放，平移
    # Set default pixel value using the minimum value of the input image's pixel type
    # For labels (UInt8), this will typically be 0 (background).
    # For CTs (Float32), this will be its minimum representable value.
    if is_label:
        resample.SetDefaultPixelValue(0)  # 标签默认设为背景0
    else:
        resample.SetDefaultPixelValue(-1000)  
    resample.SetInterpolator(interpolator)

    resampled_image = resample.Execute(image_sitk)
    return resampled_image

def preprocess_dataset(input_base_dir, output_base_dir, target_spacing_tuple):
    """
    Processes the entire dataset: reads CTs, labels, and JSON metadata,
    resamples them, and saves the output.
    """
    ct_input_dir = os.path.join(input_base_dir, "ct")
    label_input_dir = os.path.join(input_base_dir, "label")
    json_input_dir = os.path.join(input_base_dir, "json")

    ct_output_dir = os.path.join(output_base_dir, "ct_resampled")
    label_output_dir = os.path.join(output_base_dir, "label_resampled")

    os.makedirs(ct_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)

    json_files = glob.glob(os.path.join(json_input_dir, "*.json"))

    if not json_files:
        print(f"No JSON files found in {json_input_dir}. Please check the path.")
        return

    print(f"Found {len(json_files)} JSON files to process.")

    for json_filepath in json_files:
        json_filename_base = os.path.splitext(os.path.basename(json_filepath))[0]
        print(f"\nProcessing case associated with: {os.path.basename(json_filepath)}")

        try:
            with open(json_filepath, 'r') as f:
                metadata = json.load(f)

            # Determine case_id: prefer "case_id" from JSON, fallback to JSON filename
            case_id_from_json = metadata.get("case_id")
            if case_id_from_json is not None:
                case_id_str = str(case_id_from_json)
            else:
                case_id_str = json_filename_base
                print(f"  Warning: 'case_id' not found in {os.path.basename(json_filepath)}. Using filename part '{case_id_str}' as ID.")

            # --- Construct input file paths ---
            # This assumes that CT and Label NIFTI files are named like "{case_id}.nii.gz".
            # If your naming convention is different (e.g., based on source_files in JSON),
            # you MUST adjust the ct_filename and label_filename logic here.
            ct_filename = f"{case_id_str}.nii.gz"
            label_filename = f"{case_id_str}.nii.gz"

            ct_filepath = os.path.join(ct_input_dir, ct_filename)
            label_filepath = os.path.join(label_input_dir, label_filename)

            if not os.path.exists(ct_filepath):
                print(f"  Error: CT file not found: {ct_filepath}")
                print(f"  Please ensure CT files are named '{case_id_str}.nii.gz' or adjust filename logic in the script.")
                continue
            if not os.path.exists(label_filepath):
                print(f"  Error: Label file not found: {label_filepath}")
                print(f"  Please ensure Label files are named '{case_id_str}.nii.gz' or adjust filename logic in the script.")
                continue

            original_spacing_from_json = metadata.get("geometry", {}).get("spacing")
            if not original_spacing_from_json or len(original_spacing_from_json) != 3:
                print(f"  Warning: Valid 'spacing' not found in {os.path.basename(json_filepath)} for case {case_id_str}. Will attempt to use image's own spacing.")
                original_spacing_for_ct = None # Will be read from image
            else:
                # SimpleITK spacing is (x,y,z), JSON provides it in this order.
                original_spacing_for_ct = tuple(original_spacing_from_json)

            # 1. Read CT image
            ct_image_sitk = sitk.ReadImage(ct_filepath, sitk.sitkFloat32) # CT images are continuous
            if original_spacing_for_ct:
                ct_image_sitk.SetSpacing(original_spacing_for_ct)
            else: # If not in JSON, use the spacing from the NIFTI header
                original_spacing_for_ct = ct_image_sitk.GetSpacing()
            print(f"  Original CT spacing for case {case_id_str}: {original_spacing_for_ct}")


            # 2. Resample CT image
            print(f"  Resampling CT for case {case_id_str} to spacing {target_spacing_tuple}...")
            resampled_ct = resample_image_sitk(ct_image_sitk, target_spacing_tuple, sitk.sitkLinear, original_spacing_for_ct, False)
            output_ct_filepath = os.path.join(ct_output_dir, ct_filename)
            sitk.WriteImage(resampled_ct, output_ct_filepath)
            print(f"    Resampled CT saved to: {output_ct_filepath}")

            # 3. Read Label image
            label_image_sitk = sitk.ReadImage(label_filepath, sitk.sitkUInt8) # Labels are discrete (0-11, fits in UInt8)
            # Assume label image has the same original geometry as the CT for this case
            label_image_sitk.SetSpacing(original_spacing_for_ct)
            label_image_sitk.SetDirection(ct_image_sitk.GetDirection()) # Ensure alignment
            label_image_sitk.SetOrigin(ct_image_sitk.GetOrigin())       # Ensure alignment


            # 4. Resample Label image
            print(f"  Resampling Label for case {case_id_str} to spacing {target_spacing_tuple}...")
            resampled_label = resample_image_sitk(label_image_sitk, target_spacing_tuple, sitk.sitkNearestNeighbor, original_spacing_for_ct, True)
            output_label_filepath = os.path.join(label_output_dir, label_filename)
            sitk.WriteImage(resampled_label, output_label_filepath)
            print(f"    Resampled Label saved to: {output_label_filepath}")

        except Exception as e:
            print(f"  Error processing case from {os.path.basename(json_filepath)} (ID: {case_id_str if 'case_id_str' in locals() else json_filename_base}): {e}")

    print("\nDataset preprocessing complete.")

if __name__ == '__main__':
    INPUT_BASE_DIRECTORY = "CLAPATPS"  # Path to the original dataset root
    OUTPUT_BASE_DIRECTORY = "CLAPATPS_resampled_1x1x1" # Path where resampled data will be saved
    TARGET_SPACING = (1.0, 1.0, 1.0) # Example: (1mm, 1mm, 1.0mm)

    # --- Run Preprocessing ---
    print(f"Starting preprocessing...")
    print(f"Input directory: {INPUT_BASE_DIRECTORY}")
    print(f"Output directory: {OUTPUT_BASE_DIRECTORY}")
    print(f"Target spacing: {TARGET_SPACING}")

    if not os.path.isdir(INPUT_BASE_DIRECTORY):
        print(f"\nError: Input directory '{INPUT_BASE_DIRECTORY}' not found.")
        print("Please update 'INPUT_BASE_DIRECTORY' to the correct path of your dataset.")
        print("Expected structure under your input directory:")
        print(f"  {os.path.join(INPUT_BASE_DIRECTORY, 'ct/*.nii.gz')}")
        print(f"  {os.path.join(INPUT_BASE_DIRECTORY, 'label/*.nii.gz')}")
        print(f"  {os.path.join(INPUT_BASE_DIRECTORY, 'json/*.json')}")
    else:
        preprocess_dataset(INPUT_BASE_DIRECTORY, OUTPUT_BASE_DIRECTORY, TARGET_SPACING)
