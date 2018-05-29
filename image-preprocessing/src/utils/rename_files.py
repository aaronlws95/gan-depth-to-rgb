__author__ = 'QiYE'
import os

for dataset in ['vassileios','Qi','Caner','Guillermo','seung','ShanxinV7','Patrick','Xinghao','sara']:
    path = 'F:/megahand/full/annotation/%s/ego'%dataset
    files = os.listdir(path)

    for file in files:
        title,ext=os.path.splitext(file)
        name=title.split('_')
        new_name=name[:2]
        os.rename(os.path.join(path, file), os.path.join(path, new_name[0]+'_'+name[1]+ext))
        # i = i+1

