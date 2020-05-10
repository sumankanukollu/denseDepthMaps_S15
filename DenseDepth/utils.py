import numpy as np
from PIL import Image
import io,os,pdb
from zipfile import ZipFile
from PIL import Image
from matplotlib import cm
import gc,pdb
gc.enable()
gc.get_threshold()

def DepthNorm(x, maxDepth):
    return maxDepth / x

def predict(model, images, minDepth=10, maxDepth=1000, batch_size=20):
    # Support multiple RGBs, one RGB image, even grayscale
    #print('images shape', len(images.shape))
    #print('images shape 0',images.shape[0])
    #print('images shape 1', images.shape[1])
    #print('images shape 2', images.shape[2])
    if len(images.shape) < 3: images = np.stack((images,images,images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    # Compute predictions
    predictions = model.predict(images, batch_size=batch_size)
    # Put in expected range
    return np.clip(DepthNorm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth

def scale_up(scale, images):
    from skimage.transform import resize
    scaled = []
    
    for i in range(len(images)):
        img = images[i]
        output_shape = (scale * img.shape[0], scale * img.shape[1])
        scaled.append( resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True ) )

    return np.stack(scaled)

def load_images_new(image_bytes):
    loaded_images = []
    #for i,byte in enumerate(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    x = np.clip(np.asarray(image, dtype=float) / 255, 0, 1)
    loaded_images.append(x)
    return np.stack(loaded_images, axis=0)



def load_images(zipfilename):
    gc.enable()
    gc.get_threshold()
    loaded_images,imgName = [],[]
    zipObj = ZipFile(zipfilename, 'r')
    files = zipObj.namelist()
    for i in files[:13]:
        if i.startswith('bg_fg_1') or i.startswith('bg_fgFlip_1'):
            fileinp = zipObj.open(i, mode='r')
            img = Image.open(fileinp)
            ####
            img.resize((160,160), Image.ANTIALIAS)
            x = np.clip(np.asarray(img, dtype=float) / 255, 0, 1)
            loaded_images.append(x)
            imgName.append(os.path.basename(i))
    return (np.stack(loaded_images, axis=0),imgName)

def to_multichannel(i):
    if i.shape[2] == 3: return i
    i = i[:,:,0]
    return np.stack((i,i,i), axis=2)
    
def saveDepthMapImages(outputs,outputPath=None):
    start=1
    #print('You are in save depth imgs method')
    for op in outputs:
        pdb.set_trace()
        #im = Image.fromarray(np.uint8(cm.gist_earth(i)*255))
        if outputPath is None:
            if not os.path.exists(os.path.join(os.getcwd(),'depthMapOutput')):
                os.mkdir(os.path.join(os.getcwd(),'depthMapOutput'))
        #im = Image.fromarray(image.astype(np.uint8)).save('%i_op.jpg'%start)
        #im = Image.fromarray((op * 255).astype(np.uint8)).save('%i_op.jpg'%start)
        #im = Image.fromarray(op).save('%i_op.jpg'%start)
        #im = Image.fromarray((op).astype(np.float)).save('%i_op.jpg'%start)

        img = Image.fromarray(op, 'LA').save('%i_op.png'%start)
        img.show()


        # Creates PIL image
        img = Image.fromarray(np.uint8(mat * 255) , 'L')
        img.save('%i_op.jpg'%start, op)
        import matplotlib
        matplotlib.image.imsave('%i_op.png'%start, op)



        #im.save('%i_op.jpg'%start)
        start+=1
        
def display_images(outputs,outputPath=None, inputs=None, gt=None, is_colormap=True, is_rescale=True,start = 0, end = 10,imgName=None):
    import matplotlib.pyplot as plt
    import skimage
    from skimage.transform import resize
    # Create a folder
    if outputPath is None:
        #if not os.path.exists(os.path.join(os.getcwd(),'depthMapOutput')):
        if not os.path.exists(os.path.join(args.outputpath(),'depthMapOutput')):
            os.mkdir(os.path.join(args.outputpath(),'depthMapOutput'))
        print('Directory for depth mask outputs : {}'.format(os.path.join(args.outputpath(),'depthMapOutput')))
            
    plasma = plt.get_cmap('plasma')

    shape = (outputs[0].shape[0], outputs[0].shape[1], 3)
    #print(shape)
    
    for i in range(end-start):

        if is_colormap:
            rescaled = outputs[i][:,:,0]
            #print(rescaled.shape)
            #print(rescaled.shape)
            if is_rescale:
                rescaled = rescaled - np.min(rescaled)
                rescaled = rescaled / np.max(rescaled)
            #print(rescaled.shape)
            
            # img = Image.fromarray(plasma(rescaled)[:,:,:3],mode="RGB")
            # img.save(f"test{str(start)}.jpg")
            plt.figure(figsize=(2.24,2.24),dpi=100)
            plt.imshow(plasma(rescaled)[:,:,:3])
            plt.axis("off")
            #plt.savefig(f"test{str(start)}.jpg")
            #pdb.set_trace()
            name = os.path.basename(imgName[i])
            plt.savefig(os.path.join(os.getcwd(),'depthMapOutput',name))
            
        start+=1
        print(start)

    
def display_images_bkp(outputs, inputs=None, gt=None, is_colormap=True, is_rescale=True):
    import matplotlib.pyplot as plt
    import skimage
    from skimage.transform import resize

    plasma = plt.get_cmap('plasma')

    shape = (outputs[0].shape[0], outputs[0].shape[1], 3)
    
    all_images = []

    for i in range(outputs.shape[0]):
        imgs = []
        
        if isinstance(inputs, (list, tuple, np.ndarray)):
            x = to_multichannel(inputs[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True )
            imgs.append(x)

        if isinstance(gt, (list, tuple, np.ndarray)):
            x = to_multichannel(gt[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True )
            imgs.append(x)

        if is_colormap:
            rescaled = outputs[i][:,:,0]
            if is_rescale:
                rescaled = rescaled - np.min(rescaled)
                rescaled = rescaled / np.max(rescaled)
            imgs.append(plasma(rescaled)[:,:,:3])
        else:
            imgs.append(to_multichannel(outputs[i]))

        img_set = np.hstack(imgs)
        all_images.append(img_set)

    all_images = np.stack(all_images)
    
    return skimage.util.montage(all_images, multichannel=True, fill=(0,0,0))

def save_images(filename, outputs, inputs=None, gt=None, is_colormap=True, is_rescale=False):
    montage =  display_images(outputs, inputs, is_colormap, is_rescale)
    im = Image.fromarray(np.uint8(montage*255))
    im.save(filename)

def load_test_data(test_data_zip_file='nyu_test.zip'):
    print('Loading test data...', end='')
    import numpy as np
    from data import extract_zip
    data = extract_zip(test_data_zip_file)
    from io import BytesIO
    rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
    depth = np.load(BytesIO(data['eigen_test_depth.npy']))
    crop = np.load(BytesIO(data['eigen_test_crop.npy']))
    print('Test data loaded.\n')
    return {'rgb':rgb, 'depth':depth, 'crop':crop}

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()
    return a1, a2, a3, abs_rel, rmse, log_10

def evaluate(model, rgb, depth, crop, batch_size=6, verbose=False):
    N = len(rgb)

    bs = batch_size

    predictions = []
    testSetDepths = []
    
    for i in range(N//bs):    
        x = rgb[(i)*bs:(i+1)*bs,:,:,:]
        
        # Compute results
        true_y = depth[(i)*bs:(i+1)*bs,:,:]
        pred_y = scale_up(2, predict(model, x/255, minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0
        
        # Test time augmentation: mirror image estimate
        pred_y_flip = scale_up(2, predict(model, x[...,::-1,:]/255, minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0

        # Crop based on Eigen et al. crop
        true_y = true_y[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_y = pred_y[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_y_flip = pred_y_flip[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        
        # Compute errors per image in batch
        for j in range(len(true_y)):
            predictions.append(   (0.5 * pred_y[j]) + (0.5 * np.fliplr(pred_y_flip[j]))   )
            testSetDepths.append(   true_y[j]   )

    predictions = np.stack(predictions, axis=0)
    testSetDepths = np.stack(testSetDepths, axis=0)

    e = compute_errors(predictions, testSetDepths)

    if verbose:
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5]))

    return e
