# WSI_handling
Code snippets for handling digital pathology pyramidal slide images. Currently only works with annotation XMLs from Aperio ImageScope and image formats supported by Openslide.

# Useage
xml_fname=r'/mnt/data/home/pjl54/UPenn_prostate/20698.xml'
img_fname=r'/mnt/data/home/pjl54/UPenn_prostate/20698.svs'
desired_mpp = 0.25
wh = (1024,1024)

w = wsi(img_fname,xml_fname)
desired_mpp = 1

roi, mask = w.get_annotated_region(desired_mpp,'green',0)
roi, mask = w.get_annotated_region(desired_mpp,'green',0,mask_out_roi=False)

x = np.random.randint(1000, high=17000, size=None, dtype='l')
y = np.random.randint(25000, high=26000, size=None, dtype='l')
coords = (x,y)

tile = w.get_tile(desired_mpp,coords,(1000,1000))
tile_location = w.show_tile_location(desired_mpp,coords,(1000,1000))
tile_mask = w.mask_out_region(desired_mpp,coords,(1000,1000),'red')
