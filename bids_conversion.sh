#!/bin/bash

project_path="/Users/mariazimmermann/desktop/PJM"
bids_dir="/Users/mariazimmermann/dropbox/MRI_data_deaf/triplet_nifti"
dicomdir="/Users/mariazimmermann/Desktop/dicomdir"
tmp="/Users/mariazimmermann/Desktop/tmp"
dcm2niidir=/usr/local/bin/dcm2niix
## Convert dicoms to nifti and adjust them to the BIDS structure
mkdir -p $tmp # creates temporal working directory that we'll delete after conversion

cd $dicomdir
echo $dicomdir
for sub in */; do
    sub=${sub::${#sub}-1}
    

    ## Create bids file structure, if not created yet
    mkdir -p ${bids_dir}/${sub}/anat
    #mkdir -p ${bids_dir}/${sub}/fmap
    mkdir -p ${bids_dir}/${sub}/func
    #bids_dir=${bids_dir}/${sub}
    # Convert files and save them to the tmp folder
    dcm2niix -o $tmp -z y $dicomdir/$sub 

    # Adjust naming of files to BIDS requirements and put them in the right folders
    ## CAUTION! Don't reverse the order of elif-s unless you know what you're doing! (has to do with scan naming and regular expressions specified in the elifs)
    cd $tmp
    for file in *.nii.gz; do
    file=${file::${#file}-7} #cuts the extension so that we could use the name for .nii.gz and .json files
        if [[ $file == *run* ]]; then
            if [[ $file == *run-01* ]]; then
            task="diffeo"
            fi
            if [[ $file == *run-02* ]]; then
            task="2sec"
            fi
            if [[ $file == *run-03* ]]; then
            task="12sec"
            fi
            if [[ $file == *run-04* ]]; then
            task="intact1a"
            fi
            if [[ $file == *run-05* ]]; then
            task="intact1b"
            fi
            if [[ $file == *run-06* ]]; then
            task="intact2"
            fi
            if [[ $file == *story_run-05* ]]; then
            task="intact2"
            fi
            if [[ $file == *faces_run-01* ]]; then
            task="faces_run-01"
            fi
            if [[ $file == *faces_run-02* ]]; then
            task="faces_run-02"
            fi
            #run=${file: -3}
            #task=$(echo $file | cut -d "_" -f 2)
            
            
            ##run=${file: -3}
            #run=$(echo $file | cut -d "_" -f 3)
            
            mv ${file}.nii.gz $bids_dir/${sub}/func/sub-${sub}_task-${task}_bold.nii.gz
            
            mv ${file}.json $bids_dir/${sub}/func/sub-${sub}_task-${task}_bold.json

            # append task name to the json file
            jq --arg t $task '.TaskName=$t' $bids_dir/${sub}/func/sub-${sub}_task-${task}_bold.json > $tmp/tmp.json
            mv $tmp/tmp.json $bids_dir/${sub}/func/sub-${sub}_task-${task}_bold.json
        
        elif [[ $file == *t1_mpr* ]]; then
            mv ${file}.nii.gz $bids_dir/${sub}/anat/sub-${sub}_T1w.nii.gz
            mv ${file}.json $bids_dir/${sub}/anat/sub-${sub}_T1w.json
        
        
        
        
        echo "File not recognized:" $file
        fi
    done

    # Append IntendedFor info to the fieldmap .json files
    

done

