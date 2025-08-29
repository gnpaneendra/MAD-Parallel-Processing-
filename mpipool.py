# import modules
import numpy as np
import time
from multiprocessing import Pool, cpu_count

def worker(arr):
    return np.sum(arr)

def multi_process_sum(arr):
    num_processes = 24
    chunk_size = int(arr.shape[0] / num_processes)
    chunks = [arr[i:i + chunk_size] for i in range(0, arr.shape[0], chunk_size)]

    pool = mp.Pool(processes=num_processes)
    results = pool.map(f, chunks)

    return results

if __name__ == '__main__':

    start_time = time.perf_counter()
    result = multi_process_sum(arr)
    end_time = time.perf_counter()
    
    # calculating executing time
    total_time = end_time - start_time
    print(total_time)









from multiprocessing import Pool
# execute a task
def task(value):
	# add your work here...
	# ...
	# return a result, if needed
	return value

# protect the entry point
if __name__ == '__main__':
	# create the pool with the default number of workers
	with Pool() as pool:
		# issue one task for each call to the function
		for result in pool.map(task, range(100)):
			# handle the result
			print(f'>got {result}')
# report that all tasks are completed
	print('Done')







def f(n):
	for num in n:
		bite = num * num

arr = np.random.randint(256, size=(300000000))
ti = time.time()
f(arr)
print(time.time() - ti)





def mad_rfi_sliding(arr, threshold_multiplier=5, window_size=512, step_size=1):
	
	data = np.asarray(arr)
	clean = data.copy()
	rows, columns = data.shape
	
	for rn in range(rows):
		for wn in range(0, columns - window_size + 1, step_size):
			window = data[rn, wn:wn + window_size]
			median = np.median(window)
			mad = np.median(np.abs(window - median))
			upper_threshold = median + threshold_multiplier * mad
			lower_threshold = median - threshold_multiplier * mad
			rfi_mask = (window < lower_threshold) | (window > upper_threshold)

			clean[rn, wn:wn + window_size][rfi_mask] = median
	
	return clean

def process_chunk(chunk):
	return mad_rfi_sliding(chunk)

def multi_process_sum(arr):
	num_processes = max(1, cpu_count() - 1)

	chunk_size = int(np.ceil(arr.shape[0] / num_processes))
	chunks = [arr[i:i + chunk_size] for i in range(0, arr.shape[0], chunk_size)]
	
	with Pool(processes = num_processes) as p:
		clean_chunks = p.map(process_chunk, chunks)
	
	clean = np.vstack(clean_chunks)
	return clean

if __name__ == '__main__':

    start_time = time.time()
    
    ch1_v_rfi_mitigated = multi_process_sum(ch1_v_mean_sub)
    ch2_v_rfi_mitigated = multi_process_sum(ch2_v_mean_sub)
    
    end_time = time.time()
    print(end_time - start_time)
 
ch1_v_mean_sub and ch2_v_mean_sub are 2D arrays






























from multiprocessing import Pool, cpu_count
from datetime import datetime
from tqdm import tqdm

def mad_rfi_sliding(chunk, threshold_multiplier, window_size, step_size):
	
	rows, columns = chunk.shape
	mitigated = chunk.copy()
	
	for rn in tqdm(range(rows)):

		for wn in range(0, columns - window_size + 1, step_size):
			window = ch1_v_mean_sub[rn, wn:wn + window_size]
			median = np.median(window)
			mad = np.median(np.abs(window - median))
			upper_threshold = median + threshold_multiplier * mad
			lower_threshold = median - threshold_multiplier * mad
			rfi_mask = (window < lower_threshold) | (window > upper_threshold)

			mitigated[rn, wn:wn + window_size][rfi_mask] = median

	return mitigated

def multi_process(row):
	num_processes = max(1, cpu_count() - 1)

	chunk_size = int(np.ceil(row.shape[0] / num_processes))
	chunks = [row[cn:cn + chunk_size] for cn in range(0, row.shape[0], chunk_size)]
	
	with Pool(processes = num_processes) as p:
		clean_chunks = p.map(mad_rfi_sliding, chunks)
	
	clean = np.vstack(clean_chunks)
	return clean

threshold_multiplier = 5
window_size = 512
step_size = 1

if __name__ == '__main__':

    start_time = datetime.now()
    
    ch1_v_rfi_mitigated = multi_process(
    	ch1_v_mean_sub, threshold_multiplier, window_size, step_size)
    ch2_v_rfi_mitigated = multi_process(
    	ch2_v_mean_sub, threshold_multiplier, window_size, step_size)
    
    end_time = datetime.now()
    print(end_time - start_time)

















# MAD based RFI mitigation
def mad_rfi_sliding(chunk, threshold_multiplier, window_size, step_size):
	
	rows, columns = chunk.shape
	mitigated = chunk.copy()
	
	for rn in range(rows):

		for wn in range(0, columns - window_size + 1, step_size):
			window = ch1_v_mean_sub[rn, wn:wn + window_size]
			median = np.median(window)
			mad = np.median(np.abs(window - median))
			upper_threshold = median + threshold_multiplier * mad
			lower_threshold = median - threshold_multiplier * mad
			rfi_mask = (window < lower_threshold) | (window > upper_threshold)

			mitigated[rn, wn:wn + window_size][rfi_mask] = median

	return mitigated

# Parallel processing for MAD based RFI mitigation
def multi_process(row, threshold_multiplier, window_size, step_size):
	num_processes = max(1, cpu_count() - 1)

	chunk_size = int(np.ceil(row.shape[0] / num_processes))
	chunks = [row[cn:cn + chunk_size] for cn in range(0, row.shape[0], chunk_size)]
	
	with Pool(processes = num_processes) as p:
		clean_chunks = p.starmap(mad_rfi_sliding, [(chunk, threshold_multiplier, window_size, step_size) for chunk in chunks])
	
	clean = np.vstack(clean_chunks)
	return clean
	
threshold_multiplier = 5
window_size = 512
step_size = 1

if __name__ == '__main__':

    start_time = datetime.now()
    
    ch1_v_rfi_mitigated = multi_process(
    	ch1_v_mean_sub, threshold_multiplier, window_size, step_size)
    ch2_v_rfi_mitigated = multi_process(
    	ch2_v_mean_sub, threshold_multiplier, window_size, step_size)
    
    end_time = datetime.now()
    print(end_time - start_time)

