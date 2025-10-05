"""
Hybrid validation strategies combining local cache with online queries.

This module provides intelligent validation that first checks pre-computed local
physicality assessments before falling back to online Gaia queries.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from .source import (
    PhysicalityValidator, PhysicalityAssessment, PhysicalityLabel, 
    ValidationMethod, InvalidInputError, CacheStatsError
)
from .local_source import LocalDataSource
from .gaia_source import GaiaValidator
from ..config import (
    VALIDATOR_CACHE_CATALOG_NAME,
    VALIDATOR_CACHE_STRATEGY_NAME,
    VALIDATOR_CACHE_UNAVAILABLE_VALUE
)

log = logging.getLogger(__name__)


class HybridValidator(PhysicalityValidator):
    """
    A hybrid validator that first checks a local, pre-computed cache of high-confidence
    binaries before falling back to a live online query.
    
    This approach provides:
    1. Performance improvement by checking local cache first
    2. Completeness through Gaia validation for systems not in cache
    3. High-confidence assessments where available
    
    The local cache is populated from the El-Badry et al. (2021) catalog, which
    contains 1.3 million high-confidence binary systems with detailed physicality
    assessments based on Gaia astrometry.
    
    References:
        El-Badry et al. (2021), MNRAS, 506, 2269-2295
        "A catalogue of stellar multiplicity among bright stellar systems"
    """

    def __init__(self, data_source: LocalDataSource, online_validator: GaiaValidator = None):
        """
        Initialize hybrid validator.
        
        Args:
            data_source: Local database with El-Badry cross-match data
            online_validator: Optional Gaia validator for fallback queries
        """
        self.data_source = data_source
        self.online_validator = online_validator
        
        log.info("Initialized hybrid validator with local cache + online fallback")

    async def validate_physicality(self, system_data: Dict[str, Any], **kwargs) -> PhysicalityAssessment:
        """
        Validate physicality using hybrid approach.
        
        Strategy:
        1. Check local cache (El-Badry catalog) for instant assessment
        2. If not cached, fall back to online Gaia query (if available)
        3. Return appropriate assessment with method attribution
        
        Args:
            system_data: System summary data containing wds_id
            **kwargs: Additional arguments passed to online validator
            
        Returns:
            PhysicalityAssessment with appropriate confidence and method
            
        Raises:
            InvalidInputError: If wds_id is missing from system_data
            PhysicalityValidationError: If validation fails due to external errors
        """
        wds_id = system_data.get('wds_id')
        if not wds_id:
            raise InvalidInputError("Missing required field 'wds_id' in system_data")

        # Get wdss_id if available for accurate validation of multiple component systems
        wdss_id = system_data.get('wdss_id')

        # Step 1: Check the pre-computed local cache
        # Pass wdss_id to ensure correct validation for triple/quadruple systems
        local_assessment = await self.data_source.get_precomputed_physicality(wds_id, wdss_id)
        
        if local_assessment:
            identifier = wdss_id if wdss_id else wds_id
            log.debug(f"Validation for {identifier} found in local El-Badry cache")
            return local_assessment

        # Step 2: If not in cache, fall back to the online validator
        if self.online_validator:
            log.debug(f"No local validation for {wds_id}. Falling back to online Gaia query")
            return await self.online_validator.validate_physicality(system_data, **kwargs)
        else:
            # No online validator available, return insufficient data status
            log.debug(f"No validation available for {wds_id} (not in cache, no online validator)")
            return {
                'label': PhysicalityLabel.INSUFFICIENT_DATA,
                'confidence': 0.0,
                'p_value': None,
                'method': None,
                'parallax_consistency': None,
                'proper_motion_consistency': None,
                'gaia_source_id_primary': None,
                'gaia_source_id_secondary': None,
                'validation_date': datetime.now().isoformat(),
                'search_radius_arcsec': 0.0,
                'significance_thresholds': {},
                'retry_attempts': 0
            }

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the local validation cache.
        
        Returns:
            Dictionary with cache coverage and El-Badry statistics
            
        Raises:
            CacheStatsError: If cache statistics cannot be retrieved
        """
        try:
            stats = self.data_source.get_catalog_statistics()
            if not stats:
                raise CacheStatsError("Unable to retrieve catalog statistics from data source")
                
            # Add cache-specific statistics
            cache_stats = {
                'total_systems': stats.get(f'{self.data_source.summary_table}_count', 0),
                'has_online_fallback': self.online_validator is not None,
                'cache_type': VALIDATOR_CACHE_CATALOG_NAME,
                'validation_strategy': VALIDATOR_CACHE_STRATEGY_NAME
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
            except Exception as e:
                log.warning(f"Could not retrieve El-Badry specific counts: {e}")
                cache_stats['cached_systems'] = VALIDATOR_CACHE_UNAVAILABLE_VALUE
                cache_stats['cache_coverage_percent'] = VALIDATOR_CACHE_UNAVAILABLE_VALUE
            
            return cache_stats
            
        except Exception as e:
            log.error(f"Failed to get cache statistics: {e}")
            raise CacheStatsError(f"Failed to retrieve cache statistics: {e}") from e
