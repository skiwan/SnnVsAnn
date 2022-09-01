import sys
import numpy as np

def apply_guassian_noise(raw_data, noise_strength_percent):
	new_data = []
	channels = raw_data.shape[1] #?
	ts = raw_data.shape[2]
	for sample in raw_data:
		new_sample = []
		for c in range(channels):
			c_data = sample[c]
			max_abs = max([abs(max(c_data)),abs(min(c_data))])
			max_abs = max_abs * noise_strength_percent
			noise = np.random.normal(loc=0.0, scale=max_abs/3, size=ts)
			noise = [min(x, max_abs) for x in noise]
			noise = [max(x, -max_abs) for x in noise]
			n_c_data = c_data + noise
			new_sample.append(n_c_data)
		new_data.append(new_sample)
	new_data = np.array(new_data)
	return new_data


def main(raw_data, noise_strength_percent):
	apply_guassian_noise(raw_data, noise_strength_percent)

if __name__ == "__main__":
	main(*sys.argv[1:])