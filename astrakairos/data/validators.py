"""
Hybrid validation strategies combining local cache with online queries.

This module provides intelligent validation that first checks pre-computed local
physicality assessments before falling back to expensive online Gaia queries.
This dramatically improves performance for large-scale analysis while maintaining
scientific accuracy.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from .source import PhysicalityValidator, PhysicalityAssessment, PhysicalityLabel, ValidationMethod
from .local_source import LocalDataSource
from .gaia_source import GaiaValidator

log = logging.getLogger(__name__)


class HybridValidator(PhysicalityValidator):
    """
    A hybrid validator that first checks a local, pre-computed cache of high-confidence
    binaries before falling back to a live online query.
    
    This approach provides:
    1. **Performance**: Instant validation for systems in the El-Badry catalog
    2. **Completeness**: Full Gaia validation for systems not in the cache
    3. **Scientific Rigor**: Gold-standard assessments where available
    
    The local cache is populated from the El-Badry et al. (2021) catalog, which
    contains 1.3 million high-confidence binary systems with detailed physicality
    assessments based on Gaia astrometry.
    
    References:
        El-Badry et al. (2021), MNRAS, 506, 2269-2295
        "A catalogue of stellar multiplicity among bright stellar systems"
    """

    def __init__(self, data_source: LocalDataSource, online_validator: Optional[GaiaValidator] = None):
        """
        Initialize hybrid validator.
        
        Args:
            data_source: Local database with El-Badry cross-match data
            online_validator: Optional Gaia validator for fallback queries
        """
        self.data_source = data_source
        self.online_validator = online_validator
        
        log.info("Initialized hybrid validator with local cache + online fallback")

    async def validate_physicality(self, wds_summary: Dict[str, Any], **kwargs) -> Optional[PhysicalityAssessment]:
        """
        Validate physicality using hybrid approach.
        
        Strategy:
        1. Check local cache (El-Badry catalog) for instant assessment
        2. If not cached, fall back to online Gaia query (if available)
        3. Return appropriate assessment with method attribution
        
        Args:
            wds_summary: System summary data containing wds_id
            **kwargs: Additional arguments passed to online validator
            
        Returns:
            PhysicalityAssessment with appropriate confidence and method
        """
        wds_id = wds_summary.get('wds_id')
        if not wds_id:
            log.warning("No wds_id provided for physicality validation")
            return None

        # Step 1: Check the pre-computed local cache
        local_assessment = await self.data_source.get_precomputed_physicality(wds_id)
        
        if local_assessment:
            log.debug(f"Validation for {wds_id} found in local El-Badry cache")
            return local_assessment

        # Step 2: If not in cache, fall back to the online validator
        if self.online_validator:
            log.debug(f"No local validation for {wds_id}. Falling back to online Gaia query")
            return await self.online_validator.validate_physicality(wds_summary, **kwargs)
        else:
            # No online validator available, return unknown status
            log.debug(f"No validation available for {wds_id} (not in cache, no online validator)")
            return PhysicalityAssessment(
                label=PhysicalityLabel.UNKNOWN,
                confidence=0.0,
                p_value=None,
                method=ValidationMethod.INSUFFICIENT_DATA,
                parallax_consistency=None,
                proper_motion_consistency=None,
                gaia_source_id_primary=None,
                gaia_source_id_secondary=None,
                validation_date=datetime.now().isoformat(),
                search_radius_arcsec=0.0,
                significance_thresholds={},
                retry_attempts=0
            )

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the local validation cache.
        
        Returns:
            Dictionary with cache coverage and El-Badry statistics
        """
        try:
            stats = self.data_source.get_catalog_statistics()
            if stats:
                # Add cache-specific statistics
                cache_stats = {
                    'total_systems': stats.get(f'{self.data_source.summary_table}_count', 0),
                    'has_online_fallback': self.online_validator is not None,
                    'cache_type': 'El-Badry et al. (2021)',
                    'validation_strategy': 'hybrid'
                }
                
                # Try to get El-Badry specific counts
                try:
                    cursor = self.data_source.conn.execute(
                        f"SELECT COUNT(*) FROM {self.data_source.summary_table} WHERE in_el_badry_catalog = 1"
                    )
                    cache_stats['cached_systems'] = cursor.fetchone()[0]
                    cache_stats['cache_coverage_percent'] = (
                        cache_stats['cached_systems'] / cache_stats['total_systems'] * 100
                        if cache_stats['total_systems'] > 0 else 0.0
                    )
                except Exception:
                    cache_stats['cached_systems'] = 'unknown'
                    cache_stats['cache_coverage_percent'] = 'unknown'
                
                return cache_stats
            else:
                return {'error': 'Unable to retrieve cache statistics'}
                
        except Exception as e:
            log.error(f"Failed to get cache statistics: {e}")
            return {'error': str(e)}
