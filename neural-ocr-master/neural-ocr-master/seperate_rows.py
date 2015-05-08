ROW_NUMBER = 3
FILE_NAME = "full_results_mini.txt"

lines = []

for i in range(0, ROW_NUMBER):
	lines.append([])

with open(FILE_NAME, 'r') as f:
	all_lines = f.readlines()
	for ii in range(0, len(all_lines)):
		lines[ii%ROW_NUMBER].append(all_lines[ii])

for i in range(0, ROW_NUMBER):
	with open('split_results/results_'+str(i)+'.txt', 'w') as f:
		for j in range(0,len(lines[i])):
			f.write(lines[i][j])