"""Segmentation for lifelog data in different hierarchies.
This is a post-processing step before getting the answers from the retrieved data.
It doesn't have to be continous.
"""

from datetime import datetime
from enum import Enum
from typing import List

from fastapi import Model


class LifelogData(Model):
    image: str
    # Time-based features
    timestamp: str
    datetime: datetime
    # Location-based features
    location: str
    city: str
    country: str
    continent: str


class SegmentationType(Enum):
    TIME = "time"
    LOCATION = "location"


class TimeInterval(Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"
    LOCATION = "location"
    HOUR = "hour"


class LocationInterval(Enum):
    CONTINENT = "continent"
    COUNTRY = "country"
    CITY = "city"
    LOCATION = "location"


def segment_by_time(data: List[LifelogData], time_interval: TimeInterval):
    segments = []
    current_segment = []
    for image in data:
        if not current_segment:
            current_segment.append(image)
            continue
        if time_interval == TimeInterval.DAY:
            if image.datetime.day != current_segment[-1].datetime.day:
                segments.append(current_segment)
                current_segment = []
        elif time_interval == TimeInterval.WEEK:
            if image.datetime.week != current_segment[-1].datetime.week:
                segments.append(current_segment)
                current_segment = []
        elif time_interval == TimeInterval.MONTH:
            if image.datetime.month != current_segment[-1].datetime.month:
                segments.append(current_segment)
                current_segment = []
        elif time_interval == TimeInterval.YEAR:
            if image.datetime.year != current_segment[-1].datetime.year:
                segments.append(current_segment)
                current_segment = []
        elif time_interval == TimeInterval.HOUR:
            if image.datetime.hour != current_segment[-1].datetime.hour:
                segments.append(current_segment)
                current_segment = []
        else:
            raise ValueError(f"Invalid time interval: {time_interval}")
    if current_segment:
        segments.append(current_segment)
    return segments


def segment_by_location(data: List[LifelogData], location_interval: LocationInterval):
    segments = []
    current_segment = []

    for image in data:
        if not current_segment:
            current_segment.append(image)
            continue
        if location_interval == LocationInterval.CONTINENT:
            if image.continent != current_segment[-1].continent:
                segments.append(current_segment)
                current_segment = []
        elif location_interval == LocationInterval.COUNTRY:
            if image.country != current_segment[-1].country:
                segments.append(current_segment)
                current_segment = []
        elif location_interval == LocationInterval.CITY:
            if image.city != current_segment[-1].city:
                segments.append(current_segment)
                current_segment = []
        elif location_interval == LocationInterval.LOCATION:
            if image.location != current_segment[-1].location:
                segments.append(current_segment)
                current_segment = []
        else:
            raise ValueError(f"Invalid location interval: {location_interval}")
    if current_segment:
        segments.append(current_segment)

    return segments
