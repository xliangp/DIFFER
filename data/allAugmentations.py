import itertools

a=["","The image appears darker.","The image appears brighter."]
b=["","The contrast is reduced.","The contrast is enhanced."]
c=["","The colors look faded.","The colors appear more vibrant."]
d=["","The image is converted to grayscale."]
e=["","The image is blurry due to Gaussian blur."]

positions=["upper","lower","left","right","whole"]
f=[f"The image shows the {position} part of the person." for position in positions]+[""]
possible_combinations=[]
# the output can take zero or one of each of the above augmentations
for combination in itertools.product(f):
    possible_combinations.append(combination)
    #print(combination)
totalNum=len(a)*len(b)*len(c)*len(d)*len(e)*len(f)
print(totalNum)
        
print(len(possible_combinations))
#writing to a file
with open('data/allAugmentations_part.txt','w') as f:
    for i,combination in enumerate(possible_combinations):
        #f.write(f"Augmentation {i+1}\n")
        for item in combination:
            f.write(f"{item}")
        f.write("\n")
     
    