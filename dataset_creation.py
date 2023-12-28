import os, csv
from skimage import io, transform
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


directory = 'annotation'
output_file = 'face_landmarks.csv'
adjusted_output = 'face_landmarks_adj.csv'
num_landmarks = 194
images_dir = "helen_1"
save_dir = "helen1_new"

######### Create the global file #########


print("Creating the global file...")
# Open the output file in write mode
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)

    header = ['image_name']
    for i in range(num_landmarks):
        header += ['part_{}_x'.format(i), 'part_{}_y'.format(i)]

    writer.writerow(header)

    # Iterate over every file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            # Read the CSV file
            with open(os.path.join(directory, filename), 'r') as csv_file:
                reader = csv.reader(csv_file)
                # Write the contents to the output file on the same line
                entry = []
                for row in reader:
                    entry += row
                
                # convert to string and remove white space
                entry = [str(x).strip() for x in entry]
                

                writer.writerow(entry)

                
print("Done!")
print("Keep only the existing images and rescalle the landmarks...")
######### Keep only the existing images #########


landmarks_frame = pd.read_csv(output_file)
to_remove = []

print("Lenght of the dataset at the beginning:" , len(landmarks_frame))

print("Removing images that do not exist...")

for i in tqdm(range(len(landmarks_frame))):
    img_name = landmarks_frame.iloc[i, 0] + ".jpg"
    if img_name not in os.listdir(images_dir):
        to_remove.append(i)
    else:
        image = io.imread(images_dir + "/" +  img_name)
        size = image.shape
        image = transform.resize(image, (256, 256)) 

        im = Image.fromarray((image * 255).astype(np.uint8))

        im.save(save_dir + "/" + img_name)
        

        landmarks_frame.iloc[i, 1::2] = landmarks_frame.iloc[i, 1::2] * (256/size[1])
        landmarks_frame.iloc[i, 2::2] = landmarks_frame.iloc[i, 2::2] * (256/size[0])


landmarks_frame = landmarks_frame.drop(to_remove)
    
print("Lenght at the end: ", len(landmarks_frame))

landmarks_frame.to_csv(adjusted_output, index=False)


