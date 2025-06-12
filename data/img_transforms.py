from torchvision.transforms import *
from PIL import Image
import random
import math
import torchvision.transforms.functional as F

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC

class ResizeWithEqualScale(object):
    """
    Resize an image with equal scale as the original image.

    Args:
        height (int): resized height.
        width (int): resized width.
        interpolation: interpolation manner.
        fill_color (tuple): color for padding.
    """
    def __init__(self, height, width, interpolation=Image.BICUBIC, fill_color=(0,0,0)):
        self.height = height
        self.width = width
        self.interpolation = interpolation #Image.BILINEAR
        self.fill_color = fill_color

    def __call__(self, img):
        width, height = img.size
        if self.height / self.width >= height / width:
            height = int(self.width * (height / width))
            width = self.width
        else:
            width = int(self.height * (width / height))
            height = self.height 

        resized_img = img.resize((width, height), self.interpolation)
        new_img = Image.new('RGB', (self.width, self.height), self.fill_color)
        new_img.paste(resized_img, (int((self.width - width) / 2), int((self.height - height) / 2)))
        #new_img = resized_img.resize((self.width, self.height), self.interpolation)    
        #show image in an interative window
        #new_img.show()
        #img.show()

        return new_img


class RandomCroping(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, p=0.5, interpolation=Image.BILINEAR):
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        width, height = img.size
        if random.uniform(0, 1) >= self.p:
            return img
        
        new_width, new_height = int(round(width * 1.125)), int(round(height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - width
        y_maxrange = new_height - height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + width, y1 + height))

        return croped_img


class RandomErasing(object):
    """ 
    Randomly selects a rectangle region in an image and erases its pixels.

    Reference:
        Zhong et al. Random Erasing Data Augmentation. arxiv: 1708.04896, 2017.

    Args:
        probability: The probability that the Random Erasing operation will be performed.
        sl: Minimum proportion of erased area against input image.
        sh: Maximum proportion of erased area against input image.
        r1: Minimum aspect ratio of erased area.
        mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img
    
class RandomResizedCropWithPosition:
    def __init__(self, size, p=0.5,scale=(0.4, 1.0), ratio=(3. / 4., 2)):
        """
        Custom RandomResizedCrop that also returns the relative position of the cropped region.

        Args:
            size (int or tuple): Target size of the output image (H, W).
            scale (tuple): Range of size of the origin size to crop.
            ratio (tuple): Aspect ratio range of the crop.
        """
        self.p = p
        self.scale=scale
        self.ratio=ratio
        self.size=size
        
        self.parts_area=(0.33,0.66)
        self.whole_area=(0.7,1)

    def __call__(self, img):
        """
        With a probability, first increase image size to (1 + 1/8), and then perform random crop.
        Apply RandomResizedCrop and determine the position of the cropped region.

        Args:
            img (PIL.Image): Input image.

        Returns:
            PIL.Image: Cropped image.
            str: Relative position description.
        """
        # if random.uniform(0, 1) >= self.p:
        #     return img, "whole"
        img_width, img_height = img.size
        
        aspect_ratio_original=img_width/img_height
        
        region = random.choice(["left", "right", "upper", "lower","center"])
        
        # if region == "center":
        #     target_area = random.uniform(0.6,1) * img_width * img_height
        #     aspect_ratio_new = aspect_ratio_original*random.uniform((3. / 4., 4/3))
        # else:   
        #     target_area = random.uniform(0.3,0.8) * img_width * img_height
        #     aspect_ratio_new = aspect_ratio_original*random.uniform(*self.ratio)
        # w = min(int(round((target_area * aspect_ratio_new) ** 0.5)),img_width)
        # h = min(int(round((target_area / aspect_ratio_new) ** 0.5)),img_height)

        # Set crop coordinates
        if region == "left":
            w=int(img_width*random.uniform(*self.parts_area))
            h=int(img_height*random.uniform(*self.whole_area))
            left = int(img_width*random.uniform(0,0.1))
            top=random.randint(0, img_height - h)
        elif region == "right":
            w=int(img_width*random.uniform(*self.parts_area))
            h=int(img_height*random.uniform(*self.whole_area))
            left = img_width-w-int(img_width*random.uniform(0,0.1))
            top=random.randint(0, img_height - h)
        elif region == "upper":
            h=int(img_height*random.uniform(*self.parts_area))
            w=int(img_width*random.uniform(*self.whole_area))
            left = random.randint(0, img_width - w)
            top=int(img_height*random.uniform(0,0.1))
        elif region == "lower":
            h=int(img_height*random.uniform(*self.parts_area))
            w=int(img_width*random.uniform(*self.whole_area))
            left = random.randint(0, img_width - w)
            top=img_height-h-int(img_width*random.uniform(0,0.1))
        else:
            w = int(img_width*random.uniform(*self.whole_area))
            h = int(img_height*random.uniform(*self.whole_area))
            left = random.randint(0, img_width - w)
            top = random.randint(0, img_height - h)


        # Determine position
        #position = self._determine_position(x, y, w, h, img_width, img_height)

        # Apply crop
        #img = F.resized_crop(img, y, x, h, w, (img_height,img_width))
        img = F.crop(img, top, left , h, w)
        #crop.show()
        #img.show()
        return img, region

   
    
# Custom Wrapper for Tracking Applied Transformations and Parameters with Descriptions
class TrackableTransform:
    def __init__(self, transform, name):
        self.transform = transform
        self.name = name  # Store transformation name

    def __call__(self, img):
        description = ""
        if isinstance(self.transform, transforms.RandomHorizontalFlip):
            if random.random() < self.transform.p:
                img = F.hflip(img)
                return img, (self.name, {"flipped": True}, "") #"The image is mirrored horizontally."
            return img, (self.name, {"flipped": False}, "")

        elif isinstance(self.transform, transforms.ColorJitter):
            brightness_factor = random.uniform(self.transform.brightness[0], self.transform.brightness[1])
            contrast_factor = random.uniform(self.transform.contrast[0], self.transform.contrast[1])
            saturation_factor = random.uniform(self.transform.saturation[0], self.transform.saturation[1])
            hue_factor = random.uniform(self.transform.hue[0], self.transform.hue[1])

            img = F.adjust_brightness(img, brightness_factor)
            img = F.adjust_contrast(img, contrast_factor)
            img = F.adjust_saturation(img, saturation_factor)
            img = F.adjust_hue(img, hue_factor)

            # Generate description
            desc_parts = []
            # if brightness_factor < 1:
            #     desc_parts.append("The image appears darker.")
            # elif brightness_factor > 1:
            #     desc_parts.append("The image appears brighter.")
            # if contrast_factor < 1:
            #     desc_parts.append("The contrast is reduced.")
            # elif contrast_factor > 1:
            #     desc_parts.append("The contrast is enhanced.")
            # if saturation_factor < 1:
            #     desc_parts.append("The colors look faded.")
            # elif saturation_factor > 1:
            #     desc_parts.append("The colors appear more vibrant.")

            return img, (self.name, {
                "brightness": brightness_factor,
                "contrast": contrast_factor,
                "saturation": saturation_factor,
                "hue": hue_factor,
            }, " ".join(desc_parts))

        elif isinstance(self.transform, transforms.RandomGrayscale):
            if random.random() < self.transform.p:
                img = F.rgb_to_grayscale(img, num_output_channels=3)
                return img, (self.name, {"grayscale": True}, "")#The image is converted to grayscale.
            return img, (self.name, {"grayscale": False}, "")

        elif isinstance(self.transform, transforms.RandomApply):
            if random.random() < self.transform.p:
                img = self.transform.transforms[0](img)
                return img, (self.name, {"applied": True}, "")#The image is blurry due to Gaussian blur.
            return img, (self.name, {"applied": False}, "")

        elif isinstance(self.transform, RandomResizedCropWithPosition):
            if random.random() < self.transform.p:
                img,position = self.transform(img)
                return img, (self.name, {}, f"The image shows the {position} part of the person.")
            return img, (self.name, {}, "The image shows the whole part of the person.")

        elif isinstance(self.transform, transforms.RandomErasing):
            if random.random() < self.transform.p:
                img = self.transform(img)
                return img, (self.name, {"applied": True}, "A large occlusion appears on the image.")
            return img, (self.name, {"applied": False}, "")

        else:
            img = self.transform(img)  # Apply transform normally
            return img, (self.name, {}, "")

# Custom Compose Function to Track Transformations
class TrackableCompose:
    def __init__(self, transforms):
        self.transforms = transforms  # Store list of transformations

    def __call__(self, img):
        applied_transforms = []  # Store applied transformations, parameters, and descriptions
        for transform in self.transforms:
            if isinstance(transform, TrackableTransform):  # If it's a trackable transformation
                img, (_, _, description) = transform(img)
                applied_transforms.append(description)  # Store transformation name, parameters, and description
            else:
                img = transform(img)  # Apply normally if not trackable
        descriptions=''.join(applied_transforms)
        return img, descriptions  # Return final image & list of applied transformations

# Define Transformations with Tracking
augmentation_transforms = TrackableCompose([
    TrackableTransform(transforms.RandomHorizontalFlip(p=0.5), "RandomHorizontalFlip"),
    TrackableTransform(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), "ColorJitter"),
    TrackableTransform(transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3))], p=0.5), "GaussianBlur"),
    TrackableTransform(transforms.RandomGrayscale(p=0.2), "RandomGrayscale"),
    TrackableTransform(RandomResizedCropWithPosition(size=(224, 224), scale=(0.5, 1.0)), "RandomResizedCrop"),
    #TrackableTransform(transforms.RandomErasing(p=0.3), "RandomErasing"),
])