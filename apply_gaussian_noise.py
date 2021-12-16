import sys

def apply_guassian_noise(raw_data, noise_strength_percent):
	pass
	# take raw data
	# for each channel
		# figure out min max values
		# scale with noise strength percentage
		# create guassian distribution
		# apply over signal
	# save new data	|  return new data


def main(raw_data, noise_strength_percent):
	apply_guassian_noise(raw_data, noise_strength_percent)

if __name__ == "__main__":
	main(*sys.argv[1:])