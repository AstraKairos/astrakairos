"""
    In order to achieve the necessary speed for processing large catalogs
    such as the WDSS (in this case, the WDSS specifically), some kind of
    fast local database builder is required. This script builds that database
    based of the downloaded WDSS catalog files, and converts them into an SQLite
    database for fast querying.
"""