## imports
import os.path as op
import dipy.reconst.csdeconv as csd
import dipy.reconst.fwdti as fwdti
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
import psutil
import time
import threading
from getData import getScaledData


## grap cpu count
cpu_count = multiprocessing.cpu_count()
memory_size = psutil.virtual_memory().total

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
        return self.memory_usage, sum(self.memory_usage) / len(self.memory_usage)

runTimeData = []

## run csdm with the given engine and vox_per_chunk
# appends the given time, engine, and vox_per_chunk to the data dataframe
# returns the time it took to run
def run_fit(model, engine, data, brain_mask_data, num_chunks, save = True):

    global runTimeData, cpu_count, memory_size

    ## track system memory on an interval of 1 second
    monitor = MemoryMonitor(1)

    ## calc approx vox_per_chunk from num_chunks
    non_zero_count = np.count_nonzero(brain_mask_data)
    chunk_size = non_zero_count // num_chunks
    vox_per_chunk = int(chunk_size)

    print("running with, engine: ", engine, " vox_per_chunk: ", vox_per_chunk, " num_chunks: ",num_chunks)

    ## start tracking memory useage
    monitor_thread = threading.Thread(target=monitor.monitor_memory)
    monitor_thread.start()

    start = time.time()
    model.fit(data, mask=brain_mask_data, engine=engine, vox_per_chunk=vox_per_chunk)
    end = time.time()

    #Stop tracking memeory
    monitor.stop_monitor = True
    monitor_thread.join()

    ## grab memory stats
    memory_usage, average_memory_usage = monitor.get_memory_usage()

    runTime = end-start

    model_name = model.__class__.__name__

    if (save):
        runTimeData.append({'engine':engine,'vox_per_chunk':vox_per_chunk,'num_chunks':num_chunks,
        'time':runTime, 'cpu_count':cpu_count, 'memory_size':memory_size, 'num_vox':non_zero_count,
        'avg_mem': average_memory_usage, 'mem_useage': memory_usage, 'model':model_name,
        'data_shape':data.shape})
    else:
        print("save turned off, runTime not saved")

    print("time: ", runTime)

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


for i in range(1,3):
    gtab,response,brain_mask_data,data = getScaledData(i)
    csdm = csd.ConstrainedSphericalDeconvModel(gtab, response=response)
    fwdtim = fwdti.FreeWaterTensorModel(gtab)
    models = [csdm, fwdtim]
    for model in models:
        for x in range(0,14):
            num_chunks = 2**x
            for i in range(5):
                run_fit(model,"ray",data,brain_mask_data, num_chunks)
                save_data('feb29.csv')