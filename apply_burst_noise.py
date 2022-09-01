import sys
import numpy as np

def apply_burst_noise(raw_data, noise_strength_percent, burst_amounts, burst_length):
	new_data = []
	channels = raw_data.shape[1] #?
	ts = raw_data.shape[2]
	for sample in raw_data:
		new_sample = []
		for c in range(channels):
			c_data = sample[c]
			max_abs = max([abs(max(c_data)),abs(min(c_data))])
			max_abs = max_abs * noise_strength_percent
			noise = np.ones(burst_length) * max_abs
			noise_start_points = np.random.randint(low=0, high=ts-burst_length, size=burst_amounts)
			for start_point in noise_start_points:
				c_data[start_point:start_point+burst_length] += noise
			new_sample.append(c_data)
		new_data.append(new_sample)
	new_data = np.array(new_data)
	return new_data


def main(raw_data, noise_strength_percent, burst_amounts, burst_length):
	apply_burst_noise(raw_data, noise_strength_percent, burst_amounts, burst_length)

if __name__ == "__main__":
	main(*sys.argv[1:])