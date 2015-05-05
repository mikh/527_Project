names = ['Single Core', 'OpenMP', 'Pthreads']

def load_all_values(filename):
	values = []
	lines = []
	with open(filename, 'r') as f:
		lines = f.readlines()
	for ii in range(0, len(lines), 2):
		value1 = float(lines[ii])
		value2 = float(lines[ii+1])
		if value2 == 0:
			value = value1
		else:
			value = (value1+value2)/2
		values.append(value)
	return values

def load_percent_values(filename):
	lines = []
	with open(filename, 'r') as f:
		lines = f.readlines()
	return lines


ff_v = load_all_values("ff_file.txt")
bp_v = load_all_values("bp_file.txt")

ff_base = ff_v[0]
bp_base = bp_v[0]

success = load_percent_values("success_file.txt")

with open("output_file.txt", "a") as f:
	f.write("\n\n\n")
	f.write("Speedup Rating:\n\n")
	for ii in range(0, len(names)):
		f.write("{0} = {1}%(FF) {2}%(BP)\n".format(names[ii], ff_base/ff_v[ii+1]*100.0, bp_base/bp_v[ii+1]*100.0))
	f.write("\n\n\n")
	f.write("Success Rating:\n\n")
	f.write("{0} = {1}".format("Base Case", success[0]))
	for ii in range(0, len(names)):
		f.write("{0} = {1}".format(names[ii], success[ii+1]))