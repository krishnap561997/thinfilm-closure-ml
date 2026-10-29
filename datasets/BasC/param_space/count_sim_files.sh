#!/usr/bin/env bash

simulation_root="/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/datasets/BasC/param_space"

output_csv="$simulation_root/simulation_file_counts.csv"
empty_csv="$simulation_root/simulation_incomplete.csv"

echo "simulation_directory,num_h5_files" > "$output_csv"
echo "simulation_directory,num_h5_files,output_directory_exists" > "$empty_csv"

for sim_dir in "$simulation_root"/Re*_Ca*_theta*; do
    [[ -d "$sim_dir" ]] || continue

    sim_name=$(basename "$sim_dir")
    output_dir="$sim_dir/output"

    if [[ -d "$output_dir" ]]; then
        num_files=$(
            find "$output_dir" \
                -maxdepth 1 \
                -type f \
                -name "*.h5" \
                -printf "." |
            wc -c
        )

        output_exists="yes"
    else
        num_files=0
        output_exists="no"
    fi

    echo "$sim_name,$num_files" >> "$output_csv"

    if (( num_files < 201 )); then
        echo "$sim_name,$num_files,$output_exists" >> "$empty_csv"
    fi
done


echo
echo "HDF5 file counts:"
column -s, -t "$output_csv"

echo
echo "Simulations with no HDF5 data:"
column -s, -t "$empty_csv"

echo
echo "Files written:"
echo "  $output_csv"
echo "  $empty_csv"
