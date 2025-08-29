# Author: G N Paneendra

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import os
import sys
from multiprocessing import Pool, cpu_count

ti = datetime.now()

# Fucntions

# For mean subtraction
def meansub(antenna_v):
	antenna_mean = np.mean(antenna_v, axis=0)
	antenna_v_mean_sub = antenna_v - antenna_mean
	return antenna_v_mean_sub

# MAD based RFI mitigation
def mad_rfi_sliding(chunk, threshold_multiplier=5, window_size=512, step_size=1):
	
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
def multi_process(row):
	num_processes = max(1, cpu_count() - 1)

	chunk_size = int(np.ceil(row.shape[0] / num_processes))
	chunks = [row[cn:cn + chunk_size] for cn in range(0, row.shape[0], chunk_size)]
	
	with Pool(processes = num_processes) as p:
		clean_chunks = p.map(mad_rfi_sliding, chunks)
	
	clean = np.vstack(clean_chunks)
	return clean

# FFT
def fft(ch1_v, ch2_v, fft_length):
	ch1_spec = []
	ch2_spec = []
	cross_spec = []
	
	for anv in range(len(ch1_v[:,0])):			
		# Perform FFT
		fft_ch1 = np.fft.fft(ch1_v[anv,:])
		fft_ch2 = np.fft.fft(ch2_v[anv,:])
		N = len(ch1_v[anv,:])
		
		# Calculating the power
		ch1_mag = (np.abs(fft_ch1))**2 / N**2
		ch2_mag = (np.abs(fft_ch2))**2 / N**2
		  
		# Taking only the positive frequencies
		ch1_spec.append(ch1_mag[:fft_length //2])
		ch2_spec.append(ch2_mag[:fft_length //2])
		   
		# Correlation
		cross_power = fft_ch1 * np.conj(fft_ch2)
		   
		# Calculate correlation power
		cross_spectrum_magnitude = (np.abs(cross_power))**2 / N**2
				
		# Taking only the positive frequencies
		cross_spec.append(cross_spectrum_magnitude[:fft_length // 2])
			
	ch1_spec = np.array(ch1_spec)
	ch2_spec = np.array(ch2_spec)
	cross_spec = np.array(cross_spec)
		
	return ch1_spec, ch2_spec, cross_spec

# Bandpass isolation and satellite RFI masking
def bpirm(freq, spec1, spec2, both):
	# Pass band isolation
	freq_mask = (freq >= 179e6) & (freq <= 361e6)
	bp_freq = freq[freq_mask]
	bp_spec1 = spec1[:, freq_mask]
	bp_spec2 = spec2[:, freq_mask]
	bp_both = both[:, freq_mask]

	for spec in(bp_spec1, bp_spec2, bp_both):			
		
		rfi_mask_range = (bp_freq >= 320e6) & (bp_freq <= 340e6)
		rfim = spec[:, rfi_mask_range]
		
		rfi_replace = np.median(rfim, axis=1)

		rfi_bands = [(180e6, 181.5e6), (190e6, 196.2e6), (197.6e6, 198.2e6), (199.88e6, 200.1e6), (209.8e6, 210.25e6), (222e6, 225e6), (229.85e6, 230.05e6), (243e6, 271e6)]

		for low, high in rfi_bands:
			rfi_band_mask = (bp_freq >= low) & (bp_freq <= high)
			rfirows = spec[:,rfi_band_mask]
			replace_block = np.tile(rfi_replace, (rfirows.shape[1], 1))
			replace_block = np.array(replace_block)
			replace_block = replace_block.T
			spec[:,rfi_band_mask] = replace_block

	return bp_freq, bp_spec1, bp_spec2, bp_both

# To calculate running mean
def running_mean(data, window_size):
	window = np.ones(window_size) / window_size
	return np.convolve(data, window, mode='same')

#path = r'~/workspace/analysis/data/2EI/2024_02_12.csv'
path = str(sys.argv[1])

file_name = os.path.basename(os.path.normpath(path))
print(f"\nCurrent file: {file_name}")

print("\nReading the file...", end=" ")
df = pd.read_csv(path)
ch1 = df.iloc[:, 1:9953]
ch2 = df.iloc[:, 9953:df.shape[1]]
date_and_time = df.iloc[:, 0]

ch1_v = np.array(ch1)
ch2_v = np.array(ch2)
utc_time = np.array([datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in date_and_time])
print("Done")

date = utc_time[0].strftime('%Y-%m-%d')
time_1 = utc_time[0].strftime('%H:%M:%S')
time_n = utc_time[-1].strftime('%H:%M:%S')
print(f"\nObservation date: {date}")
print(f"Observation time(UT): {time_1} - {time_n} ")

print("\nPerforming mean subtraction...", end=" ")
# Mean subtraction
ch1_v_mean_sub = meansub(ch1_v)
ch2_v_mean_sub = meansub(ch2_v)
print("Done")

print("\nRFI Mitigation...", end=" ")
threshold_multiplier = 5
window_size = 512
step_size = 1

if __name__ == '__main__':

    start_time = datetime.now()
    
    ch1_v_rfi_mitigated = multi_process(ch1_v_mean_sub)
    ch2_v_rfi_mitigated = multi_process(ch2_v_mean_sub)
    
    end_time = datetime.now()
    print(end_time - start_time)
print("Done")

print("\nPerforming FFT... ", end=" ")
# Sampling parameter
sampling_rate = 1.25e9  # in Hz
fft_length = len(ch1_v_rfi_mitigated[0])

# Performing FFT
ch1_spec, ch2_spec, cross_spec = fft(ch1_v_rfi_mitigated, ch2_v_rfi_mitigated, fft_length)

# Frequency axis for the FFT (only positive frequencies)
frequencies = np.fft.fftfreq(fft_length, 1 / sampling_rate)[:fft_length // 2]
print("Done")

print("\nBand pass isolation... ", end=" ")
# Band pass isolation 
bp_frequencies, ch1_bp_spec, ch2_bp_spec, cross_bp_spec = bpirm(frequencies, ch1_spec, ch2_spec, cross_spec)
bpcn = np.arange(len(bp_frequencies))
print("Done")


ch1_total_power_1 = np.sum(ch1_bp_spec, axis = 1)
ch2_total_power_1 = np.sum(ch2_bp_spec, axis = 1)
crc_total_power_1 = np.sum(cross_bp_spec, axis = 1)


window_size = 5
ch1_total_power_2 = running_mean(ch1_total_power_1, window_size)
ch2_total_power_2 = running_mean(ch2_total_power_1, window_size)
crc_total_power_2 = running_mean(crc_total_power_1, window_size)

tf = datetime.now()
print("\n", tf - ti)

# PLOT 1

plt.figure(figsize=(17,10))
plt.suptitle(f"Channel 1, {date}, threshold multiplier = {threshold_multiplier}, window size = {window_size}, step size = {step_size}")

plt.subplot(2,1,1)
plt.plot(utc_time, ch1_total_power_1)
plt.xlabel('Time (UT)', fontsize=12)
plt.ylabel('Power', fontsize=12)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlim(utc_time[0], utc_time[-1])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(2,1,2)
plt.title(f"{window_size} point moving average")
plt.plot(utc_time, ch1_total_power_2)
plt.xlabel('Time (UT)', fontsize=12)
plt.ylabel('Power', fontsize=12)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlim(utc_time[0], utc_time[-1])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()

# PLOT 2

plt.figure(figsize=(17,10))
plt.suptitle(f'Channel 2, {date}, threshold multiplier = {threshold_multiplier}, window size = {window_size}, step size = {step_size}')

plt.subplot(2,1,1)
plt.plot(utc_time, ch2_total_power_1)
plt.xlabel('Time (UT)', fontsize=12)
plt.ylabel('Power', fontsize=12)
ch2_total_power_1 = np.sum(ch2_bp_spec, axis = 1)
crc_total_power_1 = np.sum(cross_bp_spec, axis = 1)

window_size = 5
ch1_total_power_2 = running_mean(ch1_total_power_1, window_size)
ch2_total_power_2 = running_mean(ch2_total_power_1, window_size)
crc_total_power_2 = running_mean(crc_total_power_1, window_size)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlim(utc_time[0], utc_time[-1])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(2,1,2)
plt.title(f"{window_size} point moving average")
plt.plot(utc_time, ch2_total_power_2)
plt.xlabel('Time (UT)', fontsize=12)
plt.ylabel('Power', fontsize=12)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlim(utc_time[0], utc_time[-1])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()

# PLOT 3

plt.figure(figsize=(17,10))
plt.suptitle(f"Cross-Correlated, {date}, threshold multiplier = {threshold_multiplier}, window size = {window_size}, step size = {step_size}")

plt.subplot(2,1,1)
plt.plot(utc_time, crc_total_power_1)
plt.xlabel('Time (UT)', fontsize=12)
plt.ylabel('Power', fontsize=12)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlim(utc_time[0], utc_time[-1])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(2,1,2)
plt.title(f"{window_size} point moving average")
plt.plot(utc_time, crc_total_power_2)
plt.xlabel('Time (UT)', fontsize=12)
plt.ylabel('Power', fontsize=12)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlim(utc_time[0], utc_time[-1])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()
