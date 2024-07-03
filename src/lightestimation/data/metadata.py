from collections import OrderedDict
import csv
from dataclasses import asdict, dataclass, field, fields
import datetime
import json
import os
from typing import ClassVar, List

@dataclass
class SingleRenderMetadata():
    date_format: ClassVar[str] = "%Y-%m-%d_%H-%M-%S"
    
    render_settings_path: str|None
    metahuman_name: str # Name describing 
    cubemap_index: int
    cubemap_path: str|None
    cubemap_angle: float|None
    exposure_value: float|None
    facial_expression: str|None
    focal_length: float|None
    head_rotation_x: float|None
    head_rotation_y: float|None
    head_rotation_z: float|None
    actual_head_rotation_x: float|None
    actual_head_rotation_y: float|None
    actual_head_rotation_z: float|None
    body_translation_x: float|None
    body_translation_y: float|None
    body_translation_z: float|None
    render_start_datetime: str|None
    
    def to_json(self) -> str:
        """Dump Metadata to 
        """
        return json.dumps(self.__dict__, default=json_dumper, indent=4)
    
    @classmethod
    def to_csv(cls, filename, list_of_render_metadata) -> None:
        with open(filename, "w", newline='') as f:
            flds = [fld.name for fld in fields(cls)]
            w = csv.DictWriter(f, flds, delimiter=';')
            w.writeheader()
            w.writerows([asdict(prop) for prop in list_of_render_metadata])
                
    @classmethod
    def list_from_csv(cls, filename) -> List['SingleRenderMetadata']:
        with open(filename, "r", newline='') as f:
            flds = [fld.name for fld in fields(cls)]
            r = csv.DictReader(f, flds, delimiter=';')
            metadatas = [cls(**row) for row in r] # type: ignore
            return metadatas[1:]
    
    @classmethod
    def get_filename(cls, date) -> str:
        return datetime.datetime.strftime(date, cls.date_format)+".csv"
    
def json_dumper(obj) -> dict:
    try:
        return obj.to_json()
    except:
        return obj.__dict__
    