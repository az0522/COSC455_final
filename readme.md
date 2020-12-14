Amazon Rainforest Image De-hazing and Classification

File: a.dehaze.py:  
"""  
This file loads all images under 'train_all_jpg' folder and de-haze them.  
The de-hazed images are stored under 'dehazed_all' folder.  

This program implement single image de-hazing using "dark channel prior".  
https://github.com/He-Zhang/image_dehaze  
"""  
"""  
the 'train_all_jpg" folder is around 700 MB, it is not included in the submission of the final project.  
'train_jpg/' is available at:  

the 'dehazed_all" folder is around 1 GB, it is not included in the submission of the final project.  
'dehazed_all' is available at:  
"""  


File: b.build_data.py:
"""
This file is used to import de-hazed images under folder 'train_jpg' folder
and import truth file 'train_v2.csv' to create the single .npz file which
contains the de-hazed images and their corresponding labels.
"""
"""
the 'dehazed_all" folder is around 1 GB , it is not included in the submission of the final project.
'dehazed_all' is available at:

the 'img_dataset.npz' is around 1.5 GB, it is not included in the submission of the final project
'img_dataset.npz' is available at:
"""


File: c.build_model.py:
"""
This file is used to load the .npz file and use the .npz file to create the prediction model.
The prediction model is saved as 'f.model.h5'
!!! It takes hours to build the prediction model. !!!
"""
"""
the 'img_dataset.npz' is around 1.5 GB, it is not included in the submission of the final project
'img_dataset.npz' is available at:

the 'f.model.h5' is included in the submission.
"""


File: d.model_test.py:
"""
This file loads the .npz file and evaluate the cnn model.
!!! The evaluation takes hours to finish !!!
"""
"""
the 'img_dataset.npz' is around 1.5 GB, it is not included in the submission of the final project
'img_dataset.npz' is available at:
"""

File: e.predict.py:
"""
This file loads the prediction model (.h5 file), candidate image, label_file(train_v2.csv)
and uses the prediction model to label the candidate image.
"""
