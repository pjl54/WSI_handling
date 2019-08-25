
# coding: utf-8

# In[1]:


import PIL
from PIL import Image,ImageDraw
import operator

from shapely.geometry import Polygon
import openslide

import matplotlib.pyplot as plt


# In[174]:


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
    
    def color_ref_match(colors_to_use):    
        color_ref = [(65535,1,'yellow'),(65280,2,'green'),(255,3,'red'),(16711680,4,'blue'),(16711808,5,'purple')]    
        if colors_to_use is not None:
            color_map = [c for c in color_ref if c[2] in colors_to_use]
        else:
            color_map = color_ref        

        return color_map
    
    def get_points(xmlfile,colors_to_use=None):        

        color_map = color_ref_match(colors_to_use)    

        # create element tree object
        tree = ET.parse(xmlfile)

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

    def mask_out_annotation(xml_fname,img_fname=None,desired_mpp=None,colors_to_use=None):        
    
        points, map_idx = get_points(xml_fname,colors_to_use)

        if img_fname is not None:
            osh_img = openslide.OpenSlide(img_fname)
            img_dim = osh_img.level_dimensions
            img_mpp = round(float(osh_img.properties['openslide.mpp-x']),2)
            resize_factor = img_mpp / desired_mpp                
        else:
            resize_factor = 1

        for k, pointSet in enumerate(points):
            points[k] = [(int(p[0] * resize_factor), int(p[1] * resize_factor)) for p in pointSet]

        img = Image.new('L', (int(img_dim[0][0] * resize_factor), int(img_dim[0][1] * resize_factor)), 0)

        for annCount, pointSet in enumerate(points):        
            ImageDraw.Draw(img).polygon(pointSet, fill=map_idx[annCount])

        mask = np.array(img)    

        return mask, resize_factor

    # coords is relative to the lowest-MPP layer, while wh is relative to desired_mpp
    def get_tile(self,desired_mpp,coords,wh):

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
                
        print(rect_coords)        
        tile_rect = ImageDraw.Draw(wsi_image)
        outer_point = tuple(map(operator.add,rect_coords,wsi_scaled_wh))
        print(outer_point)
        tile_rect.rectangle((rect_coords,outer_point),outline="#000000",width=10)

        return wsi_image


# In[175]:


xml_fname=r'/mnt/data/home/pjl54/UPenn_prostate/20698.xml'
img_fname=r'/mnt/data/home/pjl54/UPenn_prostate/20698.svs'
desired_mpp = 0.25
wh = (1024,1024)

w = wsi(img_fname,xml_fname)


# In[153]:


print(w["mpps"])


# In[186]:


img = w.show_tile_location(1,(9512,25596),(1000,1000))
plt.imshow(img);


# In[187]:


img = w.show_tile_location(0.25,(9512,25596),(4000,4000))
plt.imshow(img);

