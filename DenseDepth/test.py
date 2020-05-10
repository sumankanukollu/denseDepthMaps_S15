import os,pdb
import glob
import argparse
import matplotlib

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images,saveDepthMapImages
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
parser.add_argument('--start', '-s',default=1, type=str, help='start of file name.')
parser.add_argument('--end', '-e',default=11, type=str, help='end-1 of file name.')
parser.add_argument('--outputpath','-o', default=11, type=str, help='end-1 of file name.')
parser.add_argument('--rstart', '-rs',default=1, type=int, help='start range.')
parser.add_argument('--rend', '-re',default=11, type=int, help='end range.')

args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Input images

#inputs = load_images( glob.glob(args.input) )
(inputs,imgName) = load_images(args.input,rstart=args.rstart,rend=args.rend)
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = predict(model, inputs)
#print(len(outputs))
#saveDepthMapImages(outputs)
#pdb.set_trace()

#matplotlib problem on ubuntu terminal fix
#matplotlib.use('TkAgg')   

'''
# Display results
viz = display_images(outputs.copy(), inputs.copy())
#plt.figure(figsize=(4,3))
plt.figure(figsize=(30,15))
plt.imshow(viz)
plt.savefig('test.png')
plt.show()
'''

# Display results
display_images(outputs.copy(), inputs.copy(),start = int(args.start), end = int(args.end),imgName=imgName)
# plt.figure(figsize=(1,1))
# plt.imshow(viz)
# plt.savefig('test.png')
# plt.show()
print("done")
