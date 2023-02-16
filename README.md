# ASL Classification

## ASL 
The American Sign Language (ASL) dataset consists of 2515 images from different numbers and alphabets. The images are in RGB format and are not the same size. The dataset is classified into 36 classes (10 numbers and 26 alphabets).In this repo, you will train a neural network for classifying sign language images. below, you can see one sample of each class : 


![1](https://user-images.githubusercontent.com/67091916/219327275-3dd726a8-bedf-4e34-8af8-e734afaffc67.png)


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

you can see result of these to tranformation on one arbitrary image(note that we combine these two transforms using `transforms.Compose`)

![2](https://user-images.githubusercontent.com/67091916/219327284-c1a7313a-5660-427d-9580-ff2e4985c96c.png)

## Create Dataset 

Code for processing data samples can get messy and hard to maintain; we ideally want our dataset code to be decoupled from our model training code for better readability and modularity.
In this section, you will implement a custom dataset which gets address of files and loads them as needed.


```python
class ASLDataset(Dataset):
    def __init__(self, files_address:list, transform=None):
        
        self.transform=transform
        self.dirs = files_address


    def __len__(self):

        return len(self.dirs)

    def __getitem__(self, idx):
      
        image = plt.imread(self.dirs[idx])
        dir=self.dirs[idx].split("/")[2]
        label = int(dir) if dir.isdigit() else (ord(dir) - 87)
       
        if self.transform:
            image = self.transform(np.float32(image))

        return image, label
```   

## Define nueral network model 

in this section we use Alexnet model with following structure : 

![7](https://user-images.githubusercontent.com/67091916/219330414-4829f6e3-5600-4f10-be2c-62114a17b09d.png)


## Loss and accuracy on train and validation set 

