## imports
import os.path as op
import dipy.reconst.csdeconv as csd
from dipy.data.fetcher import fetch_hcp
from dipy.core.gradients import gradient_table
import nibabel  as nib
import time
from dipy.reconst.csdeconv import (auto_response_ssst,
                                   mask_for_response_ssst,
                                   response_from_mask_ssst)
from dipy.align import resample
import numpy as np
import multiprocessing
import psutil
import csv
import os
import numpy as np


cpu_count = multiprocessing.cpu_count()
memory_size = psutil.virtual_memory().total


## fetch subject data
subject = 100307
dataset_path = fetch_hcp(subject)[1]
subject_dir = op.join(
    dataset_path,
    "derivatives",
    "hcp_pipeline",
    f"sub-{subject}",
    )
print(subject_dir)
subject_files = [op.join(subject_dir, "dwi",
        f"sub-{subject}_dwi.{ext}") for ext in ["nii.gz", "bval", "bvec"]]

## load data
dwi_img = nib.load(subject_files[0])
data = dwi_img.get_fdata()
seg_img = nib.load(op.join(
    subject_dir, "anat", f'sub-{subject}_aparc+aseg_seg.nii.gz'))
seg_data = seg_img.get_fdata()
brain_mask = seg_data > 0
dwi_volume = nib.Nifti1Image(data[..., 0], dwi_img.affine)
brain_mask_xform = resample(brain_mask, dwi_volume,
                            moving_affine=seg_img.affine)
brain_mask_data = brain_mask_xform.get_fdata().astype(int)
gtab = gradient_table(subject_files[1], subject_files[2])
response_mask = mask_for_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
response, _ = response_from_mask_ssst(gtab, data, response_mask)

## define model
csdm = csd.ConstrainedSphericalDeconvModel(gtab, response=response)

runTimeData = []

## run csdm with the given engine and vox_per_chunk
# appends the given time, engine, and vox_per_chunk to the data dataframe
# returns the time it took to run
def run_csdm(engine, num_chunks,save = True):

    global runTimeData, cpu_count, memory_size

    ## calc approx vox_per_chunk from num_chunks
    non_zero_count = np.count_nonzero(brain_mask_data)
    total_chunks = num_chunks
    chunk_size = non_zero_count // total_chunks
    vox_per_chunk = int(chunk_size)

    # get process pid
    process = psutil.Process(os.getpid())

    start = time.time()
    csdm.fit(data, mask=brain_mask_data, engine=engine, vox_per_chunk=vox_per_chunk)
    end = time.time()

    runTime = end-start

    if (save):
        runTimeData.append({'engine':engine,'vox_per_chunk':vox_per_chunk,'num_chunks':num_chunks,'time':runTime, 'cpu_count':cpu_count, 'memory_size':memory_size, 'num_vox':non_zero_count})
    else:
        print("save turned off, runTime not saved")

    print("engine: ", engine, " vox_per_chunk: ", vox_per_chunk, " num_chunks: ",num_chunks, " time: ", runTime)

    return runTime

def save_data(filename):
    global runTimeData

    # specify the names for CSV column headers
    fieldnames = runTimeData[0].keys() if runTimeData else Error("No data to save")

    # writing to csv file
    with open(filename, 'a', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # writing headers (field names) if the file doesn't exist or it is empty
        if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
            csvwriter.writeheader()

        # writing the data rows
        csvwriter.writerows(runTimeData)

    runTimeData.clear()

## run csdm with the given engine and vox_per_chunk
engines = ['ray', 'dask']

print("Running CSDM with different engines and vox_per_chunk")
non_zero_count = np.count_nonzero(brain_mask_data)
print("number of voxels to compute: ", non_zero_count)

for engine in engines:
    for x in range(0,13):
        num_chunks = 2^x
        for i in range(5):
            run_csdm(engine, num_chunks)
            save_data('autoMeasurementData.csv')