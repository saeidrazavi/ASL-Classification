# ASL Classification

## ASL 
The American Sign Language (ASL) dataset consists of 2515 images from different numbers and alphabets. The images are in RGB format and are not the same size. The dataset is classified into 36 classes (10 numbers and 26 alphabets).In this repo, you will train a neural network for classifying sign language images. below, you can see one sample of each class : 



## Transforms 
we use two helpful transforms. first is `grayscale` tranform which helps us to reduce image's channel and make training proccess faster. 
second transform is `resize` which again help us to reduce size of images and make training faster. we implement both transforms from scratch.

- Grayscale
```python
class GrayScale(object):

    def __init__(self):
        pass
    def __call__(self,img):
        
        if(img.shape[0]==3):
            r,g,b=img[0,:,:],img[1,:,:],img[2,:,:]
            grayscale=1/3*(r+g+b)
        #--------------------    
        else : 
          grayscale=img    
        
        return grayscale
```

- Resize

```python
class Resize(object):
    
    def __init__(self, size=227):
        
       self.size=size
    
  
    def __call__(self, x):
    
        resized=cv2.resize(np.array(x), (self.size, self.size))
        resized=torch.tensor(np.float32(resized))

        return resized
```        
