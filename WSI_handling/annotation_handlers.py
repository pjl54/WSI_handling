import json
import xml.etree.ElementTree as ET
import numpy as np

def color_ref_match_xml(colors_to_use,custom_colors):    
    """Given a string or list of strings corresponding to colors to use, returns the hexcodes of those colors"""

    color_ref = [(65535,1,'yellow'),(65280,2,'green'),(255,3,'red'),(16711680,4,'blue'),(16711808,5,'purple'),(np.nan,6,'other')] + custom_colors

    if colors_to_use is not None:

        if isinstance(colors_to_use,str):
            colors_to_use = colors_to_use.lower()
        else:
            colors_to_use = [color.lower() for color in colors_to_use]

        color_map = [c for c in color_ref if c[2] in colors_to_use]
    else:
        color_map = color_ref        

    return color_map

def get_points_xml(wsi_object,colors_to_use,custom_colors): 
    """Given a set of annotation colors, parses the xml file to get those annotations as lists of verticies"""
    color_map = color_ref_match_xml(colors_to_use,custom_colors)    

    color_key = ''.join([k[2] for k in color_map])
    # we can store the points for this combination to speed up getting it later
    if color_key in wsi_object["stored_points"]:
        return wsi_object["stored_points"][color_key]["points"].copy(), wsi_object["stored_points"][color_key]["map_idx"].copy()
    else:
        full_map = color_ref_match_xml(None,custom_colors)

        # create element tree object
        tree = ET.parse(wsi_object["annotation_fname"])

        # get root element
        root = tree.getroot()        

        map_idx = []
        points = []

        for annotation in root.findall('Annotation'):        
            line_color = int(annotation.get('LineColor'))        
            mapped_idx = [item[1] for item in color_map if item[0] == line_color]

            if(not mapped_idx and not [item[1] for item in full_map if item[0] == line_color]):
                if('other' in [item[2] for item in color_map]):
                    mapped_idx = [item[1] for item in color_map if item[2] == 'other']                    

            if(mapped_idx):
                if(isinstance(mapped_idx,list)):
                    mapped_idx = mapped_idx[0]

                for regions in annotation.findall('Regions'):
                    for annCount, region in enumerate(regions.findall('Region')):                                
                        map_idx.append(mapped_idx)

                        for vertices in region.findall('Vertices'):
                            points.append([None] * len(vertices.findall('Vertex')))                    
                            for k, vertex in enumerate(vertices.findall('Vertex')):
                                points[-1][k] = (int(float(vertex.get('X'))), int(float(vertex.get('Y'))))                                                                            

        sort_order = [x[1] for x in color_map]
        new_order = []
        for x in sort_order:
            new_order.extend([index for index, v in enumerate(map_idx) if v == x])

        points = [points[x] for x in new_order]
        map_idx = [map_idx[x] for x in new_order]

        wsi_object["stored_points"][color_key] = []
        wsi_object["stored_points"][color_key] = {'points':points.copy(),'map_idx':map_idx.copy()}           

        return points, map_idx

def get_points_json(wsi_object,colors_to_use):
    
    colors_to_use = [c.lower() for c in colors_to_use]
    
    """Given a set of annotation types, parses the json file to get those annotations as lists of verticies"""
    color_key = ''.join([k for k in colors_to_use])
    
    mapper = list(range(1,len(colors_to_use)+1))
    map_idx = []
    
    # we can store the points for this combination to speed up getting it later
    if color_key in wsi_object["stored_points"]:
        return wsi_object["stored_points"][color_key]["points"].copy(), wsi_object["stored_points"][color_key]["map_idx"].copy()
    else:
        with open(wsi_object['annotation_fname']) as f:
            json_anno = json.load(f)
            points = []
            for idx,color in enumerate(colors_to_use):
                for anno in json_anno:
                    
                    if 'classification' not in anno['properties']:
                        anno['properties']['classification'] = dict()
                    if 'name' not in anno['properties']['classification']:
                        anno['properties']['classification']['name'] = 'null'

                    if anno['properties']['classification']['name'].lower() == color:

                        geom_type = anno['geometry']['type']
                        coordinates = anno['geometry']['coordinates']

                        if geom_type == 'MultiPolygon':
                            for roi in coordinates:
                                for sub_roi in roi:
                                    points.append([(coord[0], coord[1]) for coord in sub_roi])
                                    map_idx.append(mapper[idx])
                        elif geom_type == 'Polygon':
                            for coords in coordinates:
                                points.append([(coord[0], coord[1]) for coord in coords])
                                map_idx.append(mapper[idx])
                        elif geom_type == 'LineString':            
                            points.append([(coord[0], coord[1]) for coord in coords])                        
                            map_idx.append(mapper[idx])


            wsi_object["stored_points"][color_key] = []
            wsi_object["stored_points"][color_key] = {'points':points.copy(),'map_idx':map_idx.copy()}           

        return points, map_idx

