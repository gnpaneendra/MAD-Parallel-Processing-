# Author: G N Paneendra

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import os
import sys
from rich.progress import track



# For mean subtraction
def meansub(antenna_v):
	antenna_mean = np.mean(antenna_v, axis=0)
	antenna_v_mean_sub = antenna_v - antenna_mean
	return antenna_v_mean_sub

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

def mad_rfi_mitigation(spectrum, threshold_multiplier):
	median = np.median(spectrum)
	mad = np.median(np.abs(spectrum - median))
	upper_threshold = median + threshold_multiplier * mad
	lower_threshold = median - threshold_multiplier * mad
	rfi_mask = (spectrum < lower_threshold) | (spectrum > upper_threshold)
	return rfi_mask, median

def apply_mad_rfi_sliding(data, threshold_multiplier=5, window_size=512, step_size=1):
	data = np.asarray(data)
	clean = data.copy()
	mask = np.zeros(len(data), dtype=bool)

	for i in range(0, len(data) - window_size + 1, step_size):
		window = data[i:i + window_size]
		rfi_mask, rfi_replace = mad_rfi_mitigation(window, threshold_multiplier)

		if np.any(rfi_mask):
			mask[i:i + window_size] |= rfi_mask
			#data[i:i + window_size][rfi_mask] = rfi_replace
			clean[mask] = rfi_replace
			
	return clean #data


'''
path = r'~/workspace/analysis/data/2EI/2024_02_12.csv'
#path = str(sys.argv[1])

file_name = os.path.basename(os.path.normpath(path))
print(f"\nCurrent file: {file_name}")

print("\nReading the file...", end=" ")
df = pd.read_csv(path)
ch1_v = df.iloc[:, 1:9953]
ch2_v = df.iloc[:, 9953:df.shape[1]]
date_time = df.iloc[:, 0]

ch1_v = np.array(ch1_v)
ch2_v = np.array(ch2_v)
utc_time = np.array([datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in date_time])
print("Done")

date = utc_time[0].strftime('%Y-%m-%d')
time_1 = utc_time[0].strftime('%H:%M:%S')
time_n = utc_time[-1].strftime('%H:%M:%S')
print(f"\nObservation date: {date}")
print(f"Observation time(UT): {time_1} - {time_n} ")
'''

path = r'/home/paneendra/workspace/analysis/data/2EI/raw/2024_02_12'
#path = str(sys.argv[1])

print(f"\nCurrent directory: {path}")

folder_name = os.path.basename(os.path.normpath(path))
print(f"Current folder: {folder_name}")

nf = int(os.popen(f"ls {path} | wc -l").read().strip())
print(f"\nNo. of files: {nf}")
if not nf != 0:
	sys.exit("There are no files in this folder")

# Process each CSV file and sort the files based on the extracted time
files = glob.glob(os.path.join(path, '*.csv'))
files_sorted_by_time = sorted(files, key=extract_utc_time_from_filename)
print("\nFiles are sorted according to time")

datetime1 = extract_ist_time_from_filename(files_sorted_by_time[0]).strftime('%Y/%m/%d %H:%M:%S')
datetimel = extract_ist_time_from_filename(files_sorted_by_time[-1]).strftime('%Y/%m/%d %H:%M:%S')
time1 = extract_utc_time_from_filename(files_sorted_by_time[0]).strftime('%H:%M:%S')
timel = extract_utc_time_from_filename(files_sorted_by_time[-1]).strftime('%H:%M:%S')
date = extract_ist_time_from_filename(files_sorted_by_time[0]).strftime('%Y/%m/%d') 

print("\nDate & Time")
print(f"First file: {datetime1} (IST)")
print(f"Last file:  {datetimel} (IST)")

print(f"\nObservation date: {date}")
print(f"Observation time(UTC): {time1} - {timel}")

# Lists to accumulate time and voltage data of two antennas
antenna1_v = []
antenna2_v = []
utc_time = []

print("\nProcessing Two Element Interferometer Data")

print("\nSorting files according to time... ", end=" ")
# Process each CSV file and sort the files based on the extracted time
files = glob.glob(os.path.join(path, '*.csv'))
files_sorted_by_time = sorted(files, key=extract_utc_time_from_filename)
print("Done")

date = extract_utc_time_from_filename(files_sorted_by_time[0]).strftime('%Y/%m/%d')

print("\nReading the files...\n")
for filename in tqdm(files_sorted_by_time):
	df = pd.read_csv(filename)
	CH1 = df.iloc[:, 0]  # Antenna 1
	CH2 = df.iloc[:, 1]  # Antenna 2														   
	
	# Append the raw data (voltage)
	antenna1_v.append(CH1)
	antenna2_v.append(CH2)
	
	# Extract timestamp from filename
	utc_time.append(extract_utc_time_from_filename(filename))

antenna1_v = np.array(antenna1_v)
antenna2_v = np.array(antenna2_v)
utc_time = np.array(utc_time)
print("\nDone")

print("\nPerforming mean subtraction...", end=" ")
# Mean subtraction
ch1_v_mean_sub = meansub(ch1_v)
ch2_v_mean_sub = meansub(ch2_v)
print("Done")

t1 = datetime.now()
print("\nPerforming RFI mitigation...\n")

ch1_v_rfi_mitigated = []
for i in track(ch1_v_mean_sub, description="CH1 RFI Mitigation"):
	rfi_mitigated = apply_mad_rfi_sliding(i)
	ch1_v_rfi_mitigated.append(rfi_mitigated)

ch1_v_rfi_mitigated = np.array(ch1_v_rfi_mitigated)

ch2_v_rfi_mitigated = []
for i in track(ch2_v_mean_sub, description="CH2 RFI Mitigation"):
	rfi_mitigated = apply_mad_rfi_sliding(i)
	ch2_v_rfi_mitigated.append(rfi_mitigated)

ch2_v_rfi_mitigated = np.array(ch1_v_rfi_mitigated)
print("\nDone")
t2 = datetime.now()
print("Total time for RFI Mitigation", t2-t1)

print("\nPerforming FFT...", end=" ")
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
print("\nTotal power as ch1_total_power_1, ch2_total_power_1 and crc_total_power_}")

window_size = 50
ch1_total_power_2 = running_mean(ch1_total_power_1, window_size)
ch2_total_power_2 = running_mean(ch2_total_power_1, window_size)
crc_total_power_2 = running_mean(crc_total_power_1, window_size)

print(f"\nSmooted the total power by taking the running mean for {window_size} data points as ch1_total_power_2, ch2_total_power_2 and crc_total_power_2\n")



plt.figure(figsize=(15,10))
plt.suptitle(date)

plt.subplot(3,1,1)
plt.plot(utc_time, ch1_total_power_1)
plt.plot(utc_time, ch1_total_power_2, color='red', label=f'{window_size} Moving Average')
plt.xlabel('Time (UT)', fontsize=12)
plt.ylabel('Power', fontsize=12)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlim(utc_time[0], utc_time[-1])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()

plt.subplot(3,1,2)
plt.plot(utc_time, ch2_total_power_1)
plt.plot(utc_time, ch2_total_power_2, color='red', label=f'{window_size} Moving Average')
plt.xlabel('Time (UT)', fontsize=12)
plt.ylabel('Power', fontsize=12)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlim(utc_time[0], utc_time[-1])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()

plt.subplot(3,1,3)
plt.plot(utc_time, crc_total_power_1)
plt.plot(utc_time, crc_total_power_2, color='red', label=f'{window_size} Moving Average')
plt.xlabel('Time (UT)', fontsize=12)
plt.ylabel('Power', fontsize=12)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlim(utc_time[0], utc_time[-1])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()

plt.tight_layout()
plt.show()

'''
filename = f"{date}_total_power.png"
output_directory = os.path.expanduser('~/workspace/analysis/data/2EI/total_power/')
full_path = os.path.join(output_directory, filename)
plt.savefig(full_path, dpi=200)

df_total_power = pd.DataFrame(ch1_total_power_1, columns=['total_power_ch1'])
df_total_power.insert(0, 'utc_time', date_time)
df_total_power.insert(2, 'total_power_ch2', ch2_total_power_1)
df_total_power.insert(3, 'total_power_crc', crc_total_power_1)

filename = f"{date}_total_power.csv"
output_directory = os.path.expanduser('~/workspace/analysis/data/2EI/total_power/')
full_path = os.path.join(output_directory, filename)
df_total_power.to_csv(full_path, index=False)
'''
