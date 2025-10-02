"""
Gaia Data Pre-fetcher: Download all Gaia data for WDSS catalog in advance

This script queries Gaia once for all source IDs in the WDSS catalog and caches
the results locally. Subsequent analyses use only local data (no Gaia queries).

For 2.4M systems with ~4.8M unique Gaia IDs:
- Query time: ~2-3 hours (one-time cost)
- Analysis time after: instant (no network delays)
- Disk space: ~500 MB for cache table
"""
import asyncio
import sqlite3
import json
import time
from typing import Set, List, Dict, Any
from astroquery.gaia import Gaia
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


class GaiaDataPrefetcher:
    """Pre-fetch and cache Gaia data for massive batch analysis"""
    
    def __init__(self, wdss_db_path: str, gaia_table: str = 'gaiadr3.gaia_source'):
        self.wdss_db_path = wdss_db_path
        self.gaia_table = gaia_table
        self.batch_size = 10000  # Gaia can handle large IN clauses
        
    def extract_all_gaia_ids(self) -> Set[str]:
        """Extract all unique Gaia source IDs from WDSS catalog"""
        log.info(f"Extracting Gaia IDs from {self.wdss_db_path}...")
        
        conn = sqlite3.connect(self.wdss_db_path)
        cursor = conn.cursor()
        
        # Check if gaia_source_ids column exists
        cursor.execute("PRAGMA table_info(wdss_summary)")
        columns = [row[1] for row in cursor.fetchall()]
        
        gaia_ids = set()
        
        if 'gaia_source_ids' in columns:
            # New format: JSON mapping
            cursor.execute("SELECT gaia_source_ids FROM wdss_summary WHERE gaia_source_ids IS NOT NULL")
            for (json_str,) in cursor:
                try:
                    gaia_map = json.loads(json_str)
                    for component, gaia_id in gaia_map.items():
                        if gaia_id and str(gaia_id).strip().isdigit():
                            gaia_ids.add(str(gaia_id).strip())
                except:
                    pass
        
        # Also check individual columns if they exist
        for col in ['gaia_id_primary', 'gaia_id_secondary']:
            if col in columns:
                cursor.execute(f"SELECT {col} FROM wdss_summary WHERE {col} IS NOT NULL")
                for (gaia_id,) in cursor:
                    if gaia_id and str(gaia_id).strip().isdigit():
                        gaia_ids.add(str(gaia_id).strip())
        
        conn.close()
        
        log.info(f"Found {len(gaia_ids):,} unique Gaia source IDs")
        return gaia_ids
    
    def create_cache_table(self):
        """Create table to store cached Gaia data"""
        log.info("Creating gaia_cache table...")
        
        conn = sqlite3.connect(self.wdss_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gaia_cache (
                source_id TEXT PRIMARY KEY,
                ra REAL,
                dec REAL,
                parallax REAL,
                parallax_error REAL,
                pmra REAL,
                pmra_error REAL,
                pmdec REAL,
                pmdec_error REAL,
                ra_error REAL,
                dec_error REAL,
                ra_dec_corr REAL,
                ra_parallax_corr REAL,
                ra_pmra_corr REAL,
                ra_pmdec_corr REAL,
                dec_parallax_corr REAL,
                dec_pmra_corr REAL,
                dec_pmdec_corr REAL,
                parallax_pmra_corr REAL,
                parallax_pmdec_corr REAL,
                pmra_pmdec_corr REAL,
                phot_g_mean_mag REAL,
                bp_rp REAL,
                ruwe REAL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_gaia_cache_source_id ON gaia_cache(source_id)")
        
        conn.commit()
        conn.close()
        
        log.info("Cache table ready")
    
    def get_cached_ids(self) -> Set[str]:
        """Get IDs that are already in cache"""
        conn = sqlite3.connect(self.wdss_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT source_id FROM gaia_cache")
            cached = {row[0] for row in cursor}
            log.info(f"Found {len(cached):,} IDs already cached")
            return cached
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return set()
        finally:
            conn.close()
    
    async def fetch_batch(self, source_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch a batch of Gaia data"""
        if not source_ids:
            return []
        
        source_ids_str = ','.join(source_ids)
        
        query = f"""
        SELECT source_id, ra, dec, parallax, parallax_error,
               pmra, pmra_error, pmdec, pmdec_error,
               ra_error, dec_error, 
               ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr,
               dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr,
               parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr,
               phot_g_mean_mag, bp_rp, ruwe
        FROM {self.gaia_table}
        WHERE source_id IN ({source_ids_str})
        """
        
        log.debug(f"Querying batch of {len(source_ids)} IDs...")
        job = Gaia.launch_job_async(query)
        results = job.get_results()
        
        # Convert to list of dicts
        data = []
        for row in results:
            data.append({
                'source_id': str(row['source_id']),
                'ra': float(row['ra']) if row['ra'] is not None else None,
                'dec': float(row['dec']) if row['dec'] is not None else None,
                'parallax': float(row['parallax']) if row['parallax'] is not None else None,
                'parallax_error': float(row['parallax_error']) if row['parallax_error'] is not None else None,
                'pmra': float(row['pmra']) if row['pmra'] is not None else None,
                'pmra_error': float(row['pmra_error']) if row['pmra_error'] is not None else None,
                'pmdec': float(row['pmdec']) if row['pmdec'] is not None else None,
                'pmdec_error': float(row['pmdec_error']) if row['pmdec_error'] is not None else None,
                'ra_error': float(row['ra_error']) if row['ra_error'] is not None else None,
                'dec_error': float(row['dec_error']) if row['dec_error'] is not None else None,
                'ra_dec_corr': float(row['ra_dec_corr']) if row['ra_dec_corr'] is not None else None,
                'ra_parallax_corr': float(row['ra_parallax_corr']) if row['ra_parallax_corr'] is not None else None,
                'ra_pmra_corr': float(row['ra_pmra_corr']) if row['ra_pmra_corr'] is not None else None,
                'ra_pmdec_corr': float(row['ra_pmdec_corr']) if row['ra_pmdec_corr'] is not None else None,
                'dec_parallax_corr': float(row['dec_parallax_corr']) if row['dec_parallax_corr'] is not None else None,
                'dec_pmra_corr': float(row['dec_pmra_corr']) if row['dec_pmra_corr'] is not None else None,
                'dec_pmdec_corr': float(row['dec_pmdec_corr']) if row['dec_pmdec_corr'] is not None else None,
                'parallax_pmra_corr': float(row['parallax_pmra_corr']) if row['parallax_pmra_corr'] is not None else None,
                'parallax_pmdec_corr': float(row['parallax_pmdec_corr']) if row['parallax_pmdec_corr'] is not None else None,
                'pmra_pmdec_corr': float(row['pmra_pmdec_corr']) if row['pmra_pmdec_corr'] is not None else None,
                'phot_g_mean_mag': float(row['phot_g_mean_mag']) if row['phot_g_mean_mag'] is not None else None,
                'bp_rp': float(row['bp_rp']) if row['bp_rp'] is not None else None,
                'ruwe': float(row['ruwe']) if row['ruwe'] is not None else None,
            })
        
        return data
    
    def save_batch_to_cache(self, data: List[Dict[str, Any]]):
        """Save batch of Gaia data to cache"""
        if not data:
            return
        
        conn = sqlite3.connect(self.wdss_db_path)
        cursor = conn.cursor()
        
        cursor.executemany("""
            INSERT OR REPLACE INTO gaia_cache 
            (source_id, ra, dec, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error,
             ra_error, dec_error, ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr,
             dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr, parallax_pmra_corr, 
             parallax_pmdec_corr, pmra_pmdec_corr, phot_g_mean_mag, bp_rp, ruwe)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [(
            d['source_id'], d['ra'], d['dec'], d['parallax'], d['parallax_error'],
            d['pmra'], d['pmra_error'], d['pmdec'], d['pmdec_error'],
            d['ra_error'], d['dec_error'], d['ra_dec_corr'], d['ra_parallax_corr'],
            d['ra_pmra_corr'], d['ra_pmdec_corr'], d['dec_parallax_corr'], d['dec_pmra_corr'],
            d['dec_pmdec_corr'], d['parallax_pmra_corr'], d['parallax_pmdec_corr'],
            d['pmra_pmdec_corr'], d['phot_g_mean_mag'], d['bp_rp'], d['ruwe']
        ) for d in data])
        
        conn.commit()
        conn.close()
    
    async def prefetch_all(self, resume: bool = True):
        """Main method: prefetch all Gaia data"""
        log.info("="*80)
        log.info("GAIA DATA PRE-FETCHER")
        log.info("="*80)
        
        # Extract all Gaia IDs
        all_ids = self.extract_all_gaia_ids()
        
        if not all_ids:
            log.error("No Gaia IDs found in WDSS catalog!")
            return
        
        # Create cache table
        self.create_cache_table()
        
        # Check what's already cached
        if resume:
            cached_ids = self.get_cached_ids()
            remaining_ids = all_ids - cached_ids
            log.info(f"Resuming: {len(remaining_ids):,} IDs remaining to fetch")
        else:
            remaining_ids = all_ids
            log.info(f"Starting fresh: {len(remaining_ids):,} IDs to fetch")
        
        if not remaining_ids:
            log.info("All IDs already cached!")
            return
        
        # Split into batches
        id_list = sorted(remaining_ids)
        batches = [id_list[i:i+self.batch_size] for i in range(0, len(id_list), self.batch_size)]
        
        log.info(f"Will process {len(batches)} batches of up to {self.batch_size} IDs each")
        log.info(f"Estimated time: {len(batches) * 3 / 60:.1f} minutes")
        
        # Process batches
        start_time = time.time()
        for i, batch in enumerate(batches, 1):
            batch_start = time.time()
            
            try:
                data = await self.fetch_batch(batch)
                self.save_batch_to_cache(data)
                
                batch_time = time.time() - batch_start
                elapsed = time.time() - start_time
                remaining = (len(batches) - i) * batch_time
                
                log.info(f"Batch {i}/{len(batches)}: {len(data)} IDs cached "
                        f"({batch_time:.1f}s) | Elapsed: {elapsed/60:.1f}min | "
                        f"ETA: {remaining/60:.1f}min")
                
            except Exception as e:
                log.error(f"Batch {i} failed: {e}")
                # Continue with next batch
        
        total_time = time.time() - start_time
        log.info("="*80)
        log.info(f"PRE-FETCH COMPLETE in {total_time/60:.1f} minutes")
        log.info(f"Total IDs cached: {len(all_ids):,}")
        log.info("="*80)


async def main():
    """Run the pre-fetcher"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pre-fetch Gaia data for WDSS catalog')
    parser.add_argument('database', help='Path to WDSS SQLite database')
    parser.add_argument('--batch-size', type=int, default=10000, 
                       help='Number of IDs per batch (default: 10000)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh (ignore existing cache)')
    
    args = parser.parse_args()
    
    prefetcher = GaiaDataPrefetcher(args.database)
    prefetcher.batch_size = args.batch_size
    
    await prefetcher.prefetch_all(resume=not args.no_resume)


if __name__ == '__main__':
    asyncio.run(main())
