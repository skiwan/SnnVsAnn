import sys
import numpy as np

def apply_transient_noise(raw_data, noise_strength_percent, burst_amounts, burst_length):
	new_data = []
	channels = raw_data.shape[0] #?
	ts = raw_data.shape[1]
	for c in range(channels):
		c_data = raw_data[c]
		max_abs = max([abs(max(c_data)),abs(min(c_data))])
		max_abs = max_abs * noise_strength_percent
		noise = np.zeroes(burst_length)
		downfall = max_abs / burst_length
		for i in range(burst_length):
			noise[i] = max_abs - downfall * i

		noise_start_points = np.random.randint(low=0, high=ts-burst_length, size=burst_amounts)

		for start_point in noise_start_points:
			c_data[start_point:start_point+burst_length] += noise
		new_data.append(c_data)
	new_data = np.array(new_data)
	return new_data


def main(raw_data, noise_strength_percent, burst_amounts, burst_length):
	apply_transient_noise(raw_data, noise_strength_percent, burst_amounts, burst_length)

if __name__ == "__main__":
	main(*sys.argv[1:])