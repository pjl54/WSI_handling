# WSI_handling
Code for handling digital pathology pyramidal whole slide images (WSIs). Currently works with annotation XMLs from Aperio ImageScope or annotation json's from QuPath and image formats supported by Openslide.

Supports getting a tile from a WSI at the desired micron-per-pixel (mpp), getting either the whole WSI or an annotated region, generating a mask image for either a tile or the WSIs, and showing the location of a tile on the WSI.

# Annotation format
Example annotations are provided in ./example_annotations

XML annotations must follow the AperioImagescope format:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Annotations>
<Annotation LineColor="65280">
<Regions>
<Region>
<Vertices>
<Vertex X="56657.4765625" Y="78147.3984375"/>
<Vertex X="56657.4765625" Y="78147.3984375"/>
<Vertex X="56664.46875" Y="78147.3984375"/>
</Region>
</Regions>
</Annotation>
</Annotations>
```

With more \<Annotation\> or \<Region\> blocks for additional annotations.

json annotations must follow QuPath's json export format, which is GeoJSON:

```
[
  {
    "type": "Feature",
    "id": "PathAnnotationObject",
    "geometry": {
      "type": "Polygon",
      "coordinates": [
        [
          [76793.51, 4613.02],
          [76651.56, 4684],
          [76580.59, 4684],
          [76580.59, 4754.97]                   
        ]
      ]
    },
    "properties": {
      "classification": {
        "name": "Tumor",
        "colorRGB": -3670016
      },
      "isLocked": true,
      "measurements": []
    }
]
```

# Installation

pip install WSI_handling

# Usage
See https://github.com/pjl54/WSI_handling/blob/master/wsi_demo.ipynb
