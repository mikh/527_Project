make clean
make all
rm output_file.txt
rm ff_file.txt
rm bp_file.txt
rm success_file.txt
echo `date` > output_file.txt
echo "" >> output_file.txt
echo "Base Case" >> output_file.txt

./neural_read_base_case -t 600 -T 100 -l 1 -b 60 -a 40 -r 55 -h 45

echo "" >> output_file.txt
echo "" >> output_file.txt

echo "Single Core Case" >> output_file.txt

./neural_read_single_core -t 600 -T 100 -l 1 -b 60 -a 40 -r 55 -h 45

echo "" >> output_file.txt
echo "" >> output_file.txt

echo "Multi Core OpenMP Case" >> output_file.txt

./neural_read_multi_core_openmp -t 600 -T 100 -l 1 -b 60 -a 40 -r 55 -h 45

echo "" >> output_file.txt
echo "" >> output_file.txt

echo "Multi Core Pthreads Case" >> output_file.txt

#./neural_read_multi_core_pthreads -t 600 -T 100 -l 1 -b 60 -a 40 -r 55 -h 45

python perform_comparison.py