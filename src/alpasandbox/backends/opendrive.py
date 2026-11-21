"""Generate simple, continuous OpenDRIVE roads for CARLA scenarios."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from ..scenario import Scenario


def generate_opendrive(
    scenario: Scenario,
    lane_width: float = 3.5,
    segment_length: float = 5.0,
    margin: float = 40.0,
) -> str:
    """Build a two-lane OpenDRIVE road following the scenario's sine curve."""

    length = max(
        200.0,
        scenario.duration * scenario.ego_speed + margin,
    )

    amplitude = float(scenario.road_amplitude)
    wavelength = max(1.0, float(scenario.road_wavelength))
    k = 2.0 * math.pi / wavelength

    def point(s: float) -> tuple[float, float, float]:
        x = s
        y = amplitude * math.sin(k * s) if amplitude else 0.0
        heading = math.atan(amplitude * k * math.cos(k * s)) if amplitude else 0.0
        return x, y, heading

    root = ET.Element("OpenDRIVE")
    ET.SubElement(
        root,
        "header",
        revMajor="1",
        revMinor="4",
        name=f"alpasandbox_{scenario.name}",
        version="1.00",
        date="2026-11-19",
        north="0",
        south="0",
        east=f"{length:.3f}",
        west="0",
    )

    header = root.find("header")
    ET.SubElement(header, "geoReference").text = (   # silences CARLA's "cannot parse georeference"
        "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

    road = ET.SubElement(
        root,
        "road",
        name=scenario.name,
        length=f"{length:.6f}",
        id="1",
        junction="-1",
    )
    ET.SubElement(road, "link")
    plan = ET.SubElement(road, "planView")

    count = math.ceil(length / segment_length)
    samples = [min(i * segment_length, length) for i in range(count + 1)]
    sampled_points = [point(s) for s in samples]

    segment_lengths = [
        math.hypot(
            sampled_points[i + 1][0] - sampled_points[i][0],
            sampled_points[i + 1][1] - sampled_points[i][1],
        )
        for i in range(count)
    ]

    road_s = [0.0]
    for segment_len in segment_lengths:
        road_s.append(road_s[-1] + segment_len)

    road.set("length", f"{road_s[-1]:.6f}")

    for i in range(count):
        x0, y0, h0 = sampled_points[i]
        x1, y1, h1 = sampled_points[i + 1]
        s0 = road_s[i]
        arc_length = segment_lengths[i]

        dx, dy = x1 - x0, y1 - y0
        ch, sh = math.cos(h0), math.sin(h0)

        # Transform the endpoint into the segment's local coordinates.
        px = ch * dx + sh * dy
        py = -sh * dx + ch * dy

        ds = arc_length
        dh = h1 - h0

        geometry = ET.SubElement(
            plan,
            "geometry",
            s=f"{s0:.6f}",
            x=f"{x0:.9f}",
            y=f"{y0:.9f}",
            hdg=f"{h0:.12f}",
            length=f"{arc_length:.9f}",
        )

        if abs(amplitude) < 1e-9:
            ET.SubElement(geometry, "line")
            continue

        # Cubic Hermite curve with matching endpoint positions and tangents.
        m0x, m0y = ds, 0.0
        m1x = ds * math.cos(dh)
        m1y = ds * math.sin(dh)

        cu = 3.0 * px - 2.0 * m0x - m1x
        du = -2.0 * px + m0x + m1x
        cv = 3.0 * py - 2.0 * m0y - m1y
        dv = -2.0 * py + m0y + m1y

        ET.SubElement(
            geometry,
            "paramPoly3",
            aU="0",
            bU=f"{m0x:.12f}",
            cU=f"{cu:.12f}",
            dU=f"{du:.12f}",
            aV="0",
            bV=f"{m0y:.12f}",
            cV=f"{cv:.12f}",
            dV=f"{dv:.12f}",
            pRange="normalized",
        )

    lanes = ET.SubElement(road, "lanes")
    ET.SubElement(lanes, "laneOffset", s="0", a="0", b="0", c="0", d="0")
    section = ET.SubElement(lanes, "laneSection", s="0")

    left = ET.SubElement(section, "left")
    center = ET.SubElement(section, "center")
    right = ET.SubElement(section, "right")

    for parent, lane_id in ((left, "1"), (right, "-1")):
        lane = ET.SubElement(
            parent,
            "lane",
            id=lane_id,
            type="driving",
            level="false",
        )
        ET.SubElement(lane, "link")
        ET.SubElement(
            lane,
            "width",
            sOffset="0",
            a=str(lane_width),
            b="0",
            c="0",
            d="0",
        )
        ET.SubElement(
            lane,
            "roadMark",
            sOffset="0",
            type="solid",
            weight="standard",
            color="standard",
            width="0.15",
            laneChange="none",
        )

    center_lane = ET.SubElement(
        center,
        "lane",
        id="0",
        type="none",
        level="false",
    )
    ET.SubElement(center_lane, "link")
    ET.SubElement(
        center_lane,
        "roadMark",
        sOffset="0",
        type="broken",
        weight="standard",
        color="standard",
        width="0.15",
        laneChange="none",
    )

    ET.SubElement(road, "objects")
    ET.SubElement(road, "signals")

    return ET.tostring(root, encoding="unicode")
