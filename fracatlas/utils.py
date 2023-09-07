from pathlib import Path
from xml.etree import ElementTree as ET


def find_bounding_box(pascal_voc_path, target_size=(256, 256)):
    xml = ET.parse(pascal_voc_path)
    root = xml.getroot()
    size = root.find("size")
    (width, height) = (int(size.find("width").text), int(size.find("height").text))
    for object in root.findall("object"):
        bndbox = object.find("bndbox")

        # extract bounding box
        (xmin, ymin, xmax, ymax) = (
            int(bndbox.find("xmin").text),
            int(bndbox.find("ymin").text),
            int(bndbox.find("xmax").text),
            int(bndbox.find("ymax").text),
        )

        # convert to target_size coordinates
        (xmin, ymin, xmax, ymax) = (
            (xmin * target_size[0]) / width,
            (ymin * target_size[1]) / height,
            (xmax * target_size[0]) / width,
            (ymax * target_size[1]) / height,
        )

        # convert to xywh
        (x, y, w, h) = (
            (xmin + xmax) / 2,
            (ymin + ymax) / 2,
            xmax - xmin,
            ymax - ymin,
        )

        yield (x, y, w, h)


def find_bbox(image_path: Path, pascal_voc_directory, target_size=(256, 256)):
    xml_path = pascal_voc_directory / image_path.with_suffix(".xml").name

    return list(find_bounding_box(xml_path, target_size))
