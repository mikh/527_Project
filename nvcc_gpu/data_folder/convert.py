import shutil, os, Image, struct, array, binascii

SIZE = 28*28
SAMPLES = 1000
FILES = 10

image_bytes = []

print("Loading bytes")
for i in range(0,FILES):
	print("%d%% complete" % int(float(i)/float(FILES)*100.0))
	byte_data = []
	with open('data'+str(i), 'rb') as f:
		for j in range(0,SAMPLES):
			byte_data.append(f.read(SIZE))
	image_bytes.append(byte_data)
print("100% complete")

'''
print("Setting up output directory")
try:
	shutil.rmtree('data')

except:
	#whatever
	print("No data directory")
try:
	os.makedirs('data')
	for i in range(0, len(image_bytes)):
		os.makedirs('data\\'+str(i))
except:
	print("Could not make directories")


'''
print("Converting bytes to images")
for i in range(0, len(image_bytes)):
	print("%d%% complete" % int(float(i)/float(FILES)*100.0))
	for j in range(0,len(image_bytes[i])):
		img = Image.new('L', (28,28), "black")
		pixels = img.load()
		single_image = image_bytes[i][j]


		for x in range(0,28):
			for y in range(0,28):
				a = single_image[x*28+y]
				hex_a = binascii.hexlify(a)
				i_byte = 0xff - int(hex_a,16)
				#print(type(i_byte))
				#i_byte = (0xff - struct.unpack('H',single_image[x*28+y])[0])
				pixels[x,y] = i_byte
		img.save(("data\\%d\\data%d_%d.jpg" % (i, i , j)), "JPEG")	
		
