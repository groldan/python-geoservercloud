"""
ImageMosaic acceptance test for an externally managed granule index.

Production-scale pattern: the PostGIS index table is created and populated
outside GeoServer (gdaltindex, ogr2ogr, or SQL), the store is created once
with UseExistingSchema=true, and granules are added or removed with plain
SQL afterwards. GeoServer reads the index live: new rows appear on all pods
with no REST calls. REST harvest MUST NOT be used on such a store (it is a
silent no-op when UseExistingSchema=true).
"""

import tempfile
import zipfile
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import Connection
from sqlalchemy.sql import text

from geoservercloud import GeoServerCloud

GRANULE_BASE = "https://test-data-cog-public.s3.amazonaws.com/public"
SCHEMA = "extidx"
COVERAGE = "land_shallow_external"


def _make_config_zip(tmp_path: Path, config: dict) -> bytes:
    indexer_content = f"""Cog=true
CogRangeReader=it.geosolutions.imageioimpl.plugins.cog.HttpRangeReader
Schema=*the_geom:Polygon,location:String
CanBeEmpty=true
UseExistingSchema=true
Name={COVERAGE}"""
    datastore_content = f"""SPI=org.geotools.data.postgis.PostgisNGDataStoreFactory
host={config['db']['pg_host']['docker']}
port={config['db']['pg_port']['docker']}
database={config['db']['pg_db']}
schema={SCHEMA}
user={config['db']['pg_user']}
passwd={config['db']['pg_password']}
Loose\\ bbox=true
Estimated\\ extends=false
validate\\ connections=true
preparedStatements=false
"""
    indexer_file = tmp_path / "indexer.properties"
    indexer_file.write_text(indexer_content)
    datastore_file = tmp_path / "datastore.properties"
    datastore_file.write_text(datastore_content)

    zip_file = tmp_path / f"{COVERAGE}.zip"
    with zipfile.ZipFile(zip_file, "w") as zf:
        zf.write(indexer_file, "indexer.properties")
        zf.write(datastore_file, "datastore.properties")
    return zip_file.read_bytes()


@pytest.fixture
def external_index(db_session: Connection) -> Generator[None, None, None]:
    """Create and populate the granule index table outside of GeoServer,
    the way a production ingestion pipeline would (gdaltindex, ogr2ogr,
    or plain SQL), before the store is ever created."""
    db_session.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
    db_session.execute(
        text(
            f"CREATE TABLE {SCHEMA}.{COVERAGE} ("
            "fid serial PRIMARY KEY, "
            "the_geom geometry(Polygon, 4326), "
            "location varchar)"
        )
    )
    db_session.execute(
        text(f"CREATE INDEX ON {SCHEMA}.{COVERAGE} USING GIST (the_geom)")
    )
    granules = {
        "land_shallow_topo_21600_NE_cog.tif": (0, 0, 180, 90),
        "land_shallow_topo_21600_NW_cog.tif": (-180, 0, 0, 90),
        "land_shallow_topo_21600_SE_cog.tif": (0, -90, 180, 0),
    }
    for name, (minx, miny, maxx, maxy) in granules.items():
        db_session.execute(
            text(
                f"INSERT INTO {SCHEMA}.{COVERAGE} (the_geom, location) VALUES ("
                f"ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326), "
                f"'{GRANULE_BASE}/{name}')"
            )
        )
    db_session.commit()
    yield
    db_session.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
    db_session.commit()


@pytest.mark.cog
@pytest.mark.db
def test_imagemosaic_over_external_index(
    geoserver_factory,
    config: dict,
    external_index: None,
    db_session: Connection,
):
    workspace = "external_index"
    geoserver: GeoServerCloud = geoserver_factory(workspace)

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_data = _make_config_zip(Path(tmp_dir), config)

    content, status = geoserver.create_imagemosaic_store_from_properties_zip(
        workspace_name=workspace, coveragestore_name=COVERAGE, properties_zip=zip_data
    )
    assert status == 201, f"store creation over populated index failed: {content}"

    content, status = geoserver.get_coverages(workspace, COVERAGE)
    assert status == 200
    assert content[0].get("name") == COVERAGE

    content, status = geoserver.create_coverage(
        workspace_name=workspace,
        coveragestore_name=COVERAGE,
        coverage_name=COVERAGE,
        title="External index mosaic",
    )
    assert status == 201

    wms_response = geoserver.get_map(
        layers=[f"{workspace}:{COVERAGE}"],
        bbox=(-180, -90, 180, 90),
        size=(256, 128),
        srs="EPSG:4326",
        format="image/png",
    )._response
    assert wms_response.status_code == 200, f"WMS failed: {wms_response.text}"
    assert wms_response.headers.get("content-type").startswith("image/png")

    # A granule added with plain SQL must appear with no REST call and no
    # reader reset: this is the whole point of the externally managed index.
    sw_quadrant_bbox = (-180, -90, 0, 0)
    sw_quadrant_size = (128, 64)

    # CanBeEmpty=true makes the mosaic return a blank tile (still 200,
    # image/png) for a region with no granules; status and content-type
    # alone therefore cannot tell a rendered granule from an empty tile.
    # Capture this blank-tile baseline before the insert to compare against
    # afterwards.
    baseline_response = geoserver.get_map(
        layers=[f"{workspace}:{COVERAGE}"],
        bbox=sw_quadrant_bbox,
        size=sw_quadrant_size,
        srs="EPSG:4326",
        format="image/png",
    )._response
    assert baseline_response.status_code == 200, f"WMS failed: {baseline_response.text}"
    assert baseline_response.headers.get("content-type").startswith("image/png")

    db_session.execute(
        text(
            f"INSERT INTO {SCHEMA}.{COVERAGE} (the_geom, location) VALUES ("
            "ST_MakeEnvelope(-180, -90, 0, 0, 4326), "
            f"'{GRANULE_BASE}/land_shallow_topo_21600_SW_cog.tif')"
        )
    )
    db_session.commit()

    wms_response = geoserver.get_map(
        layers=[f"{workspace}:{COVERAGE}"],
        bbox=sw_quadrant_bbox,
        size=sw_quadrant_size,
        srs="EPSG:4326",
        format="image/png",
    )._response
    assert (
        wms_response.status_code == 200
    ), f"WMS after SQL insert failed: {wms_response.text}"
    assert wms_response.headers.get("content-type").startswith("image/png")
    assert wms_response.content != baseline_response.content, (
        "WMS response identical to the pre-insert blank tile: "
        "the SQL-inserted granule was not picked up"
    )
