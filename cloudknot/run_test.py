#!/usr/bin/env python3
import boto3
import cloudpickle
import os
import pickle
from argparse import ArgumentParser
from functools import wraps


def pickle_to_s3(server_side_encryption=None, array_job=True):
    def real_decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            s3 = boto3.client("s3")
            bucket = os.environ.get("CLOUDKNOT_JOBS_S3_BUCKET")

            if array_job:
                array_index = os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX")
            else:
                array_index = "0"

            jobid = os.environ.get("AWS_BATCH_JOB_ID")

            if array_job:
                jobid = jobid.split(":")[0]

            key = "/".join(
                [
                    "cloudknot.jobs",
                    os.environ.get("CLOUDKNOT_S3_JOBDEF_KEY"),
                    jobid,
                    array_index,
                    "{0:03d}".format(int(os.environ.get("AWS_BATCH_JOB_ATTEMPT"))),
                    "output.pickle",
                ]
            )

            result = f(*args, **kwargs)

            # Only pickle output and write to S3 if it is not None
            if result is not None:
                pickled_result = cloudpickle.dumps(result, protocol=3)
                if server_side_encryption is None:
                    s3.put_object(Bucket=bucket, Body=pickled_result, Key=key)
                else:
                    s3.put_object(
                        Bucket=bucket,
                        Body=pickled_result,
                        Key=key,
                        ServerSideEncryption=server_side_encryption,
                    )

        return wrapper

    return real_decorator


def run_test(aws_access_key_id, aws_secret_access_key, subject=100307,
             num_chunks=None, factor=1, model='csd',
             engine='ray', return_fit=False):
    """
    Runs the test with the given parameters.

    Parameters:
    aws_access_key_id (str): The AWS access key ID for accessing HCP data
                             Keys can be obtained
                             `here <https://db.humanconnectome.org/>`_.
    aws_secret_access_key (str): The AWS secret access key for HCP
    subject (int, optional): The subject ID to use in the test. Defaults to
                             100307.
    num_chunks (int, optional): The number of chunks to use in the test. Must
                                be less than the number of voxels in the image.
                                The number of computed chunks might be higher
                                than the specified num_chunks due to intiger
                                divison. Defaults to None.
    factor (int, optional): The factor to use in the test. Defaults to 1.
    model (str, optional): The model to use in the test. Defaults to 'csd'.
    engine (str, optional): The engine to use in the test. Defaults to 'ray'.
    return_fit (bool, optional): Whether to return the fit object. Defaults to
                                 False.

    Returns:
    dict: A dictionary containing the results of the test.
    """

    import os.path as op
    from dipy.data.fetcher import fetch_hcp
    from dipy.core.gradients import gradient_table
    import nibabel as nib
    import time
    from dipy.align import resample
    import numpy as np
    import multiprocessing
    import psutil
    from nilearn import image
    import threading

    def downsample(img, shape, factor=2):
        shape = tuple(s // factor for s in shape)
        img_resampled = image.resample_img(img, target_affine=img.affine,
                                           target_shape=shape,
                                           interpolation='continuous')
        return img_resampled

    class MemoryMonitor:
        def __init__(self, interval):
            self.interval = interval
            self.memory_usage = []
            self.stop_monitor = False

        def monitor_memory(self):
            while not self.stop_monitor:
                mem_info = psutil.virtual_memory()
                used_memory_GB = mem_info.used / (1024 ** 3)
                self.memory_usage.append(used_memory_GB)
                time.sleep(self.interval)

        def get_memory_usage(self):
            return self.memory_usage,
            sum(self.memory_usage) / len(self.memory_usage)

    dataset_path = fetch_hcp(subject, profile_name=False,
                             aws_access_key_id=aws_access_key_id,
                             aws_secret_access_key=aws_secret_access_key)[1]
    subject_dir = op.join(dataset_path, "derivatives", "hcp_pipeline",
                          f"sub-{subject}",)
    subject_files = [op.join(subject_dir, "dwi", f"sub-{subject}_dwi.{ext}")
                     for ext in ["nii.gz", "bval", "bvec"]]

    dwi_img = nib.load(subject_files[0])
    shape = dwi_img.shape[:-1]
    dwi_img = downsample(img=dwi_img, shape=shape)
    data = dwi_img.get_fdata()

    seg_img = nib.load(op.join(
        subject_dir, "anat", f'sub-{subject}_aparc+aseg_seg.nii.gz'))
    shape = seg_img.shape
    seg_img = downsample(img=seg_img, shape=shape)
    seg_data = seg_img.get_fdata()

    brain_mask = seg_data > 0
    dwi_volume = nib.Nifti1Image(data[..., 0], dwi_img.affine)
    print(f'dwi_volume shape: {dwi_volume.shape}')
    brain_mask_xform = resample(brain_mask, dwi_volume,
                                moving_affine=seg_img.affine)
    brain_mask_data = brain_mask_xform.get_fdata().astype(int)
    gtab = gradient_table(subject_files[1], subject_files[2])

    if model == 'csd':
        import dipy.reconst.csdeconv as csd
        from dipy.reconst.csdeconv import (mask_for_response_ssst,
                                           response_from_mask_ssst)
        response_mask = mask_for_response_ssst(gtab, data, roi_radii=10,
                                               fa_thr=0.7)
        response, _ = response_from_mask_ssst(gtab, data, response_mask)
        model = csd(gtab, response=response)
    elif model == 'fwdti':
        import dipy.reconst.fwdti as fwdti
        model = fwdti.FreeWaterTensorModel(gtab)

    cpu_count = multiprocessing.cpu_count()
    memory_size = psutil.virtual_memory().total

    non_zero_count = np.count_nonzero(brain_mask_data)
    vox_per_chunk = non_zero_count // num_chunks

    monitor = MemoryMonitor(1)
    monitor_thread = threading.Thread(target=monitor.monitor_memory)
    monitor_thread.start()

    start = time.time()
    fit = model.fit(data, mask=brain_mask_data, engine=engine,
                    vox_per_chunk=vox_per_chunk)
    end = time.time()

    monitor.stop_monitor = True
    monitor_thread.join()

    memory_usage, avg_memory_usage = monitor.get_memory_usage()
    run_time = end - start
    model_name = model.__class__.__name__

    test_results = {'engine': engine, 'vox_per_chunk': vox_per_chunk,
                            'num_chunks': num_chunks, 'time': run_time,
                            'cpu_count': cpu_count,
                            'memory_size': memory_size,
                            'num_vox': non_zero_count,
                            'avg_mem': avg_memory_usage,
                            'mem_useage': memory_usage, 'model': model_name,
                            'downsample_factor': factor, 'subject': subject,
                            'data_shape': data.shape}

    if return_fit:
        return test_results, fit
    else:
        return test_results


if __name__ == "__main__":
    description = (
        "Download input from an S3 bucket and provide that input "
        "to our function. On return put output in an S3 bucket."
    )

    parser = ArgumentParser(description=description)

    parser.add_argument(
        "bucket",
        metavar="bucket",
        type=str,
        help="The S3 bucket for pulling input and pushing output.",
    )

    parser.add_argument(
        "--starmap",
        action="store_true",
        help="Assume input has already been grouped into a single tuple.",
    )

    parser.add_argument(
        "--arrayjob",
        action="store_true",
        help="If True, this is an array job and it should reference the "
        "AWS_BATCH_JOB_ARRAY_INDEX environment variable.",
    )

    parser.add_argument(
        "--sse",
        dest="sse",
        action="store",
        choices=["AES256", "aws:kms"],
        default=None,
        help="Server side encryption algorithm used when storing objects in S3.",
    )

    args = parser.parse_args()

    s3 = boto3.client("s3")
    bucket = args.bucket

    jobid = os.environ.get("AWS_BATCH_JOB_ID")

    if args.arrayjob:
        jobid = jobid.split(":")[0]

    key = "/".join(
        [
            "cloudknot.jobs",
            os.environ.get("CLOUDKNOT_S3_JOBDEF_KEY"),
            jobid,
            "input.pickle",
        ]
    )

    response = s3.get_object(Bucket=bucket, Key=key)
    input_ = pickle.loads(response.get("Body").read())

    if args.arrayjob:
        array_index = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX"))
        input_ = input_[array_index]

    if args.starmap:
        pickle_to_s3(args.sse, args.arrayjob)(run_test)(*input_)
    else:
        pickle_to_s3(args.sse, args.arrayjob)(run_test)(input_)
