
# coding: utf-8

# In[1]:


import xml.etree.ElementTree as ET
import numpy as np
import cv2

import PIL
from PIL import Image,ImageDraw
import operator

from shapely.geometry import Polygon
import openslide

import matplotlib.pyplot as plt
import matplotlib as mpl


# In[173]:


class wsi(dict):
    
    def __init__(self,img_fname=None,xml_fname=None):
        self["img_fname"] = img_fname
        self["xml_fname"] = xml_fname
        
        if img_fname is not None:
            self["osh"] = openslide.OpenSlide(img_fname)
            self["mpp"] = float(self["osh"].properties['openslide.mpp-x'])
            self["downsamples"] = self["osh"].level_downsamples
            self["mpps"] = [round(ds*self["mpp"],2) for ds in self["downsamples"]]            
            
            self["img_dims"] = self["osh"].level_dimensions
    
    def color_ref_match(self,colors_to_use):    
        
        if isinstance(colors_to_use,str):
            colors_to_use = colors_to_use.lower()
        else:
            colors_to_use = [color.lower() for color in colors_to_use]
        
        color_ref = [(65535,1,'yellow'),(65280,2,'green'),(255,3,'red'),(16711680,4,'blue'),(16711808,5,'purple')]    
        if colors_to_use is not None:
            color_map = [c for c in color_ref if c[2] in colors_to_use]
        else:
            color_map = color_ref        

        return color_map
    
    def get_points(self,colors_to_use=None):        
        
        color_map = self.color_ref_match(colors_to_use)    

        # create element tree object
        tree = ET.parse(self["xml_fname"])

        # get root element
        root = tree.getroot()        

        map_idx = []
        points = []

        for annotation in root.findall('Annotation'):        
            line_color = int(annotation.get('LineColor'))        
            mapped_idx = [item[1] for item in color_map if item[0] == line_color]

            if(len(mapped_idx) is not 0):
                mapped_idx = mapped_idx[0]
                for regions in annotation.findall('Regions'):
                    for annCount, region in enumerate(regions.findall('Region')):                                
                        map_idx.append(mapped_idx)

                        for vertices in region.findall('Vertices'):
                            points.append([None] * len(vertices.findall('Vertex')))                    
                            for k, vertex in enumerate(vertices.findall('Vertex')):
                                points[-1][k] = (int(float(vertex.get('X'))), int(float(vertex.get('Y'))))                                                                            

        return points, map_idx
        
    def get_coord_at_mpp(self,coordinate,output_mpp,input_mpp=None):
        
        if input_mpp is None:
            input_mpp = self["mpp"]
                   
        coordinate = int(coordinate * input_mpp / output_mpp)
        
        return coordinate
    
    def get_layer_for_mpp(self,desired_mpp,wh=None):
        
        diff_mpps = [x for x in [mpp - float(desired_mpp) for mpp in self["mpps"]]]    
        valid_layers = [(index,diff_mpp) for index,diff_mpp in enumerate(diff_mpps) if diff_mpp>=0]
        valid_diff_mpps = [v[1] for v in valid_layers]
        valid_layers= [v[0] for v in valid_layers]
        if len(valid_layers) == 0:
            print('Warning: desired_mpp is lower than minimum image MPP of ' + min(self["mpps"]))
            target_layer = diff_mpps.index(min([v[1] for v in valid_layers])) 
        else:
            target_layer = valid_layers[valid_diff_mpps.index(min(valid_diff_mpps))]
                
        layer_scale = desired_mpp / self["mpps"][target_layer]        
        
        if wh is not None:
            wh = [int(float(dimension) * layer_scale) for dimension in wh]            
        
        return target_layer, layer_scale, wh
            
    def read_region(self,coords,target_layer,wh):
        img = self["osh"].read_region(coords,target_layer,wh)
        
        return img        

    def mask_out_annotation(self,desired_mpp=None,colors_to_use=None):        
    
        points, map_idx = self.get_points(colors_to_use)

        if img_fname is not None:
            resize_factor = self["mpp"] / desired_mpp                
        else:
            resize_factor = 1

        for k, pointSet in enumerate(points):
            points[k] = [(int(p[0] * resize_factor), int(p[1] * resize_factor)) for p in pointSet]

        img = Image.new('L', (int(img_dim[0][0] * resize_factor), int(img_dim[0][1] * resize_factor)), 0)

        for annCount, pointSet in enumerate(points):        
            ImageDraw.Draw(img).polygon(pointSet, fill=map_idx[annCount])

        mask = np.array(img)    

        return mask, resize_factor

    def mask_out_region(self,desired_mpp,coords,wh,colors_to_use=None):
    
        points, map_idx = self.get_points(colors_to_use)

        if img_fname is not None:
            resize_factor = self["mpp"] / desired_mpp                
        else:
            resize_factor = 1

        # this rounding may de-align the mask and RGB image
        for k, pointSet in enumerate(points):
            points[k] = [(int(p[0] * resize_factor), int(p[1] * resize_factor)) for p in pointSet]

        coords = tuple([int(c * resize_factor) for c in coords])

        polygon = np.array([[coords[0],coords[1]],[coords[0]+wh[0],coords[1]],[coords[0]+wh[0],coords[1]+wh[1]],[coords[0],coords[1]+wh[1]]])

        new_points = []
        for k, point_set in enumerate(points):                

            if(all(mpl.path.Path(np.array(point_set)).contains_points(polygon))):
                new_points.append(polygon)

                region_point_set = [point for index,point in enumerate(point_set) if point[0]>coords[0] and point[0]<(coords[0]+wh[0])
                                       and point[1]>coords[1] and point[1]<(coords[1]+wh[1])]

                new_points.append(region_point_set)

        mask = Image.new('L', (wh[0], wh[1]), 0)

        for k, pointSet in enumerate(points):
            points[k] = [(int(p[0] - coords[0]), int(p[1] - coords[1])) for p in pointSet]

        for annCount, pointSet in enumerate(points):        
            ImageDraw.Draw(mask).polygon(pointSet, fill=map_idx[annCount])        
        
        return mask
    
    def get_tile(self,desired_mpp,coords,wh,wh_at_base=False):        
        
        if wh_at_base:
            wh = [self.get_coord_at_mpp(dimension,output_mpp=desired_mpp) for dimension in wh]            
        
        target_layer, _, scaled_wh = self.get_layer_for_mpp(desired_mpp,wh)
        
        img_image = self.read_region(coords,target_layer,scaled_wh)

        interp_method=PIL.Image.NEAREST
        img = img_image.resize(wh, resample=interp_method)

        return img

    def show_tile_location(self,desired_mpp,coords,wh,wsi_mpp=8):            
        
        target_layer, layer_scale, scaled_wh = self.get_layer_for_mpp(desired_mpp,wh)
        wsi_target_layer, wsi_layer_scale, wsi_scaled_wh = self.get_layer_for_mpp(wsi_mpp,wh)
                
        rect_coords = tuple([self.get_coord_at_mpp(c,wsi_mpp) for c in coords])        
        wsi_scaled_wh = tuple([self.get_coord_at_mpp(dimension,wsi_mpp,input_mpp=desired_mpp) for dimension in wh])        
        
        wsi_image = self.read_region((0,0),wsi_target_layer,self["img_dims"][wsi_target_layer])        
                    
        tile_rect = ImageDraw.Draw(wsi_image)
        outer_point = tuple(map(operator.add,rect_coords,wsi_scaled_wh))        
        tile_rect.rectangle((rect_coords,outer_point),outline="#000000",width=10)

        return wsi_image
    
    def get_annotated_region(self,desired_mpp,colors_to_use,annotation_idx,mask_out_roi=True):
            
        points, _ = self.get_points(colors_to_use)
        
        poly_list = [Polygon(point_set) for point_set in points]
        
        
        if annotation_idx is 'largest':
            areas = [poly.area for poly in poly_list]
            annotation_idx = areas.index(max(areas))
        
        bounding_box = poly_list[annotation_idx].bounds

        coords = tuple([int(bounding_box[0]),int(bounding_box[1])])
        wh = tuple([int(bounding_box[2]-bounding_box[0]),int(bounding_box[3]-bounding_box[1])])
        
        img = self.get_tile(desired_mpp,coords,wh,wh_at_base=True)
        
        wh = [self.get_coord_at_mpp(dimension,output_mpp=desired_mpp) for dimension in wh]            
        mask = self.mask_out_region(desired_mpp,coords,wh,colors_to_use=colors_to_use)
        
        if(mask_out_roi):
            background = Image.new('L', img.size, color=255)        
            img = PIL.Image.composite(background,img,mask.point(lambda p: p == 0 and 255))

        return img, mask


# In[99]:


colors_to_use = ('green','yelLLw','red')
colors_to_use = ('green')
print(type(colors_to_use))
print(isinstance(colors_to_use,str))
colors_to_use = [color.lower() for color in colors_to_use]
print(colors_to_use)


# In[174]:


xml_fname=r'/mnt/data/home/pjl54/UPenn_prostate/20698.xml'
img_fname=r'/mnt/data/home/pjl54/UPenn_prostate/20698.svs'
desired_mpp = 0.25
wh = (1024,1024)

w = wsi(img_fname,xml_fname)


# In[177]:


roi, mask = w.get_annotated_region(1,'green',0,mask_out_roi=True)

fig, ax = plt.subplots(1,2,figsize=(20,20));

mask.putpixel((0,0),0)
# mask[0][1]=1
# mask[0][2]=2
# mask[0][3]=3
# mask[0][4]=4

ax[0].imshow(roi)
ax[1].imshow(mask)


# In[157]:


plt.imshow(mask.point(lambda p: p > 0 and 255))


# In[4]:


desired_mpp = 1
coords = (9512,25596)
wh = (1024,1024)

x = np.random.randint(1000, high=17000, size=None, dtype='l')
y = np.random.randint(25000, high=26000, size=None, dtype='l')
coords = (x,y)


fig, ax = plt.subplots(1,3,figsize=(20,20));

mask = w.mask_out_region(1,coords,(1000,1000),'red')
mask[0][0]=0
mask[0][1]=1
mask[0][2]=2
mask[0][3]=3
mask[0][4]=4

ax[0].imshow(w.get_tile(1,coords,(1000,1000)))
ax[1].imshow(w.show_tile_location(1,coords,(1000,1000)))
ax[2].imshow(mask)


# In[33]:




