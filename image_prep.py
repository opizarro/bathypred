import os
import cv2
import numpy as np
import glob

# images stored as png



def benthic_process(x):
    x = x.astype(np.float32) / 255.0
    return x

def max_min_rescale(x):
    x = x.astype(np.float32)
    x = ( x - min(x) ) / (max(x) - min(x))
    return x

def load_cache(fn):
    try:
        npzfile = np.load(fn)
        return npzfile
    except:
        print("cache not available, loading individual images")
        return 0

def pack_images(image_dir,cached_images):
    # images in set
    #n = len([name for name in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, name))])
    n = len([name for name in glob.glob(os.path.join(image_dir,"*LC16.png"))])
    print("number of images {}".format(n))

    # fraction to be 'test'
    testfrac = 0.0

    #
    ntest = int(n * testfrac)
    ntrain = n - ntest

    imsize = 64
    packed_image_list = []

    # check if images have been cached into array
    #npzfile = load_cache('/data/cacheBenthoz32.npz')
    npzfile = load_cache(cached_images)
    if not npzfile:

        xtrain = np.zeros((ntrain, imsize, imsize, 3))
        xtest = np.zeros((ntest, imsize, imsize, 3))

        i = 0
        #loop through images
        for filename in os.listdir(image_dir):
            if filename.endswith("LC16.png"):
                #read
                if i%100==0 :
                    print('\r', "processing image number {} name {} ".format(i,filename), end='')
                image = cv2.imread(os.path.join(image_dir,filename))
                #small = cv2.resize(image, (0,0), fx=rescale_im,fy=rescale_im)
                small = cv2.resize(image, (imsize,imsize))
                #print("mean small image {}".format(np.mean(small)))
                #cv2.imwrite("output/aae-benthic/" + filename, small)
                #print("counter i {}".format(i))
                if i < ntrain:
                    # save a train array
                    #print("i {} less than num train samps {}".format(i,ntrain))
                    xtrain[i,:,:,:] = benthic_process(small)
                elif i >= ntrain and i < n:
                    #print("i {} equal or greater than {} and less than {}".format(i,ntrain,n))
                    xtest[i-ntrain,:,:,:] = benthic_process(small)
                else:
                    #print("mean xtrain {}".format(np.mean(xtrain)))
                    break
                i += 1
                packed_image_list.append(filename)

        np.savez(cached_images,xtrain=xtrain, xtest=xtest, packed_image_list=packed_image_list)
    else:
        print("load cached images")
        xtrain = npzfile['xtrain']
        xtest = npzfile['xtest']
        packed_image_list = npzfile['packed_image_list']
    return xtrain, xtest, packed_image_list

def pack_bpatches(bpatch_dir,cached_bpatches,packed_image_list,bathy_patch_size):
    # images in set
    n = len(packed_image_list)
    # fraction to be 'test'
    testfrac = 0.0

    #
    ntest = int(n * testfrac)
    ntrain = n - ntest

    bpsize = bathy_patch_size
    patch_suffix = "bp" + str(bpsize) +".npy"

    # check if images have been cached into array
    #npzfile = load_cache('/data/cacheBenthoz32.npz')


    xtrain = np.zeros((ntrain, bpsize, bpsize, 1))
    xtest = np.zeros((ntest, bpsize, bpsize, 1))


    i = 0
    #loop through images
    for filename in packed_image_list:
        if filename.endswith("LC16.png"):
        #read
            if i%100 == 0:
                print("processing bpatch for image number {} name {} ".format(i,filename))

            patch_name = filename.replace("LC16.png",patch_suffix)
            bpatch = np.load(os.path.join(bpatch_dir,patch_name))

            if i < ntrain:
                # save a train array
                #print("i {} less than num train samps {}".format(i,ntrain))
                xtrain[i,:,:,0] = bpatch
            elif i >= ntrain and i < n:
                #print("i {} equal or greater than {} and less than {}".format(i,ntrain,n))
                xtest[i-ntrain,:,:,0] = bpatch
            else:                    #print("mean xtrain {}".format(np.mean(xtrain)))
                break
            i += 1

    bmax = np.amax(np.concatenate((xtrain,xtest)))
    bmin = np.amin(np.concatenate((xtrain,xtest)))
    xtrain  =  (xtrain - bmin )/ (bmax-bmin)
    xtest = (xtest - bmin) / (bmax-bmin)
    np.savez(cached_bpatches,xtrain=xtrain, xtest=xtest)

    return xtrain, xtest
